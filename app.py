import os
import datetime
import threading
import time
import requests
import torch
import gc
import json
from flask import Flask, render_template, request, jsonify, url_for
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusion3Pipeline, 
    EulerAncestralDiscreteScheduler
)

# --- НАСТРОЙКИ ---
IMAGE_FOLDER = 'outputs'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
DEFAULT_NEGATIVE = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name"

app = Flask(__name__)
app.static_folder = 'static'

# Создаем папки
static_img_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
os.makedirs(static_img_dir, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Глобальные переменные состояния
pipe = None
current_model_name = None
current_model_type = None
model_state = {
    'status': 'initializing',
    'progress': 0,
    'message': 'Проверка файлов...'
}

downloads_state = {}

# --- ФУНКЦИИ МОДЕЛЕЙ И ЗАГРУЗКИ ---

def save_model_meta(filename, model_type):
    if model_type == 'auto': return
    meta_path = os.path.join(MODELS_FOLDER, 'meta.json')
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            pass
    meta[filename] = model_type
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)

def get_model_meta(filename):
    meta_path = os.path.join(MODELS_FOLDER, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return meta.get(filename, 'auto')
        except Exception:
            pass
    return 'auto'

def download_file(url, filename, is_main_init=False, api_key=None):
    filepath = os.path.join(MODELS_FOLDER, filename)
    max_retries = 3
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
            
        response = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, headers=headers, timeout=15)
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    if is_main_init:
                        model_state['message'] = f'Сервер занят. Попытка {attempt + 2}/{max_retries} через {wait_time}с...'
                    else:
                        downloads_state[filename] = {'progress': 0, 'status': 'downloading', 'error': f'Ожидание сервера ({wait_time}с)...'}
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 4):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        if is_main_init:
                            model_state['progress'] = progress
                        else:
                            downloads_state[filename] = {'progress': progress, 'status': 'downloading', 'error': ''}
                            
        if not is_main_init:
            downloads_state[filename] = {'progress': 100, 'status': 'completed', 'error': ''}
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            error_msg = "Требуется API ключ (ошибка 401)"
        elif "404" in error_msg:
            error_msg = "Файл не найден (ошибка 404)"
            
        if is_main_init:
            model_state['status'] = 'awaiting_setup'
            model_state['message'] = f'Ошибка: {error_msg}. Попробуйте снова.'
        else:
            downloads_state[filename] = {'progress': 0, 'status': 'error', 'error': error_msg}
            
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def load_model_into_vram(filename):
    global pipe, model_state, current_model_name, current_model_type
    
    filepath = os.path.join(MODELS_FOLDER, filename)
    if not os.path.exists(filepath):
        model_state['status'] = 'error'
        model_state['message'] = f'Файл {filename} не найден!'
        return False

    model_state['status'] = 'loading'
    model_state['progress'] = 100
    model_state['message'] = f'Определение типа и загрузка модели {filename} в VRAM...'
    
    try:
        if pipe is not None:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SD 3.5 требует bfloat16 для стабильной работы, если поддерживается картой
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
        
        explicit_type = get_model_meta(filename)
        
        is_probably_sd3 = False
        is_probably_sdxl = False
        is_probably_sd15 = False
        
        if explicit_type == 'sd3':
            is_probably_sd3 = True
        elif explicit_type == 'sdxl':
            is_probably_sdxl = True
        elif explicit_type == 'sd15':
            is_probably_sd15 = True
        else:
            file_size = os.path.getsize(filepath)
            filename_lower = filename.lower()
            is_probably_sd3 = "sd3" in filename_lower or "3.5" in filename_lower
            is_probably_sdxl = file_size > 4.5 * 1024 * 1024 * 1024 and not is_probably_sd3

        def try_load_sd3():
            # Попытка 1: С явным конфигом SD 3.5
            try:
                print("Пробуем загрузить как SD 3.5 Medium (с конфигом)...")
                return StableDiffusion3Pipeline.from_single_file(
                    filepath,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    config="stabilityai/stable-diffusion-3.5-medium"
                ), "SD 3.5"
            except Exception as e1:
                print(f"Загрузка с конфигом 3.5 не удалась ({e1}). Пробуем стандартный метод...")
                # Попытка 2: Без явного конфига (автоопределение)
                return StableDiffusion3Pipeline.from_single_file(
                    filepath,
                    torch_dtype=dtype,
                    use_safetensors=True
                ), "SD 3 / 3.5"

        def try_load_sdxl():
            try:
                return StableDiffusionXLPipeline.from_single_file(
                    filepath, 
                    torch_dtype=dtype, 
                    use_safetensors=True,
                    config="stabilityai/stable-diffusion-xl-base-1.0"
                ), "SDXL"
            except Exception:
                return StableDiffusionXLPipeline.from_single_file(
                    filepath, torch_dtype=dtype, use_safetensors=True
                ), "SDXL"

        def try_load_sd15():
            try:
                return StableDiffusionPipeline.from_single_file(
                    filepath, 
                    torch_dtype=dtype, 
                    use_safetensors=True,
                    config="runwayml/stable-diffusion-v1-5"
                ), "SD 1.5 / Стандартная"
            except Exception:
                return StableDiffusionPipeline.from_single_file(
                    filepath, torch_dtype=dtype, use_safetensors=True
                ), "SD 1.5 / Стандартная"

        if is_probably_sd3:
            attempts = [try_load_sd3, try_load_sdxl, try_load_sd15]
        elif is_probably_sdxl:
            attempts = [try_load_sdxl, try_load_sd3, try_load_sd15]
        elif is_probably_sd15:
            attempts = [try_load_sd15, try_load_sdxl, try_load_sd3]
        else:
            attempts = [try_load_sd15, try_load_sdxl, try_load_sd3]

        temp_pipe = None
        loaded_type = ""
        error_logs = []

        for attempt_func in attempts:
            try:
                temp_pipe, loaded_type = attempt_func()
                break
            except Exception as e:
                error_str = str(e).split('\n')[0] # Берем только первую строчку ошибки для компактности
                error_logs.append(f"{attempt_func.__name__}: {error_str}")
                print(f"Не удалось загрузить через {attempt_func.__name__}: {e}")
                continue

        if temp_pipe is None:
            # Улучшенная диагностика ошибок для пользователя
            filename_lower = filename.lower()
            if "fp8" in filename_lower or "scaled" in filename_lower:
                raise Exception("Это FP8-модель формата ComfyUI. Текущая библиотека не поддерживает этот специфичный формат. Скачайте официальную стандартную версию (FP16).")
            else:
                # Показываем реальную системную ошибку
                raise Exception(f"Архитектура не распознана. Ошибка: {error_logs[0][:150]}...")

        if "SD 3" not in loaded_type:
            temp_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(temp_pipe.scheduler.config)
            
        temp_pipe.to(device)
        
        if device == "cuda":
            temp_pipe.enable_model_cpu_offload()
            
        pipe = temp_pipe
        current_model_name = filename
        current_model_type = loaded_type
        model_state['status'] = 'ready'
        model_state['message'] = f'Модель [{loaded_type}] готова!'
        return True
    except Exception as e:
        model_state['status'] = 'error'
        model_state['message'] = f'Ошибка: {str(e)}'
        return False

def startup_check():
    global model_state
    models_list = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.safetensors')]
    
    if not models_list:
        model_state['status'] = 'awaiting_setup'
        model_state['message'] = 'Выберите модель для первого запуска'
    else:
        load_model_into_vram(models_list[0])

threading.Thread(target=startup_check, daemon=True).start()

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# --- ОСНОВНЫЕ РОУТЫ ---
@app.route('/')
def gallery():
    image_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
    images = []
    if os.path.exists(image_dir):
        files = os.listdir(image_dir)
        for f in files:
            filepath = os.path.join(image_dir, f)
            if os.path.isfile(filepath) and allowed_file(f):
                stats = os.stat(filepath)
                images.append({
                    'name': f,
                    'url': f"{IMAGE_FOLDER}/{f}",
                    'size': f"{stats.st_size / 1024:.1f} KB",
                    'date': datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%d.%m.%Y %H:%M"),
                    'timestamp': stats.st_mtime
                })
    images.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('index.html', images=images)

@app.route('/generator')
def generator_page():
    return render_template('generator.html')

@app.route('/status')
def get_status():
    return jsonify({
        'state': model_state,
        'current_model': current_model_name,
        'current_model_type': current_model_type
    })

# --- API МЕНЕДЖЕРА МОДЕЛЕЙ ---
@app.route('/api/models', methods=['GET'])
def list_models():
    models = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.safetensors')]
    return jsonify({'models': models, 'current': current_model_name})

@app.route('/api/models/download', methods=['POST'])
def api_download_model():
    data = request.json
    url = data.get('url')
    filename = data.get('filename')
    api_key = data.get('api_key')
    is_init = data.get('is_init', False)
    model_type = data.get('model_type', 'auto')
    
    if not url or not filename:
        return jsonify({'success': False, 'error': 'Укажите URL и имя файла'}), 400
        
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
        
    save_model_meta(filename, model_type)
        
    if is_init:
        model_state['status'] = 'downloading'
        model_state['message'] = f'Скачивание {filename}...'
        model_state['progress'] = 0
        
        def init_download_wrapper():
            success = download_file(url, filename, is_main_init=True, api_key=api_key)
            if success:
                load_model_into_vram(filename)

        threading.Thread(target=init_download_wrapper, daemon=True).start()
    else:
        downloads_state[filename] = {'progress': 0, 'status': 'starting', 'error': ''}
        threading.Thread(target=download_file, args=(url, filename, False, api_key), daemon=True).start()
        
    return jsonify({'success': True, 'message': 'Загрузка начата'})

@app.route('/api/models/downloads_status', methods=['GET'])
def api_downloads_status():
    return jsonify(downloads_state)

@app.route('/api/models/load', methods=['POST'])
def api_load_model():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'error': 'Укажите имя файла'}), 400
        
    threading.Thread(target=load_model_into_vram, args=(filename,), daemon=True).start()
    return jsonify({'success': True, 'message': 'Инициализирована загрузка модели'})

# --- РОУТ ГЕНЕРАЦИИ ---
@app.route('/generate', methods=['POST'])
def generate_art():
    if model_state['status'] != 'ready' or pipe is None:
        return jsonify({'success': False, 'error': 'Нейросеть еще не готова.'}), 503

    data = request.json
    user_prompt = data.get('prompt', '').strip()
    user_negative = data.get('negative_prompt', '').strip()
    ratio = data.get('ratio', '9:16')
    style = data.get('style', 'Anime')
    steps = int(data.get('steps', 28))
    guidance_scale = float(data.get('guidance_scale', 7.0))
    seed = int(data.get('seed', -1))

    if not user_prompt:
        return jsonify({'success': False, 'error': 'Промпт пустой'}), 400

    if current_model_type and ("SD 3" in current_model_type or current_model_type == "SDXL"):
        dims = {"1:1": (1024, 1024), "16:9": (1216, 680), "9:16": (832, 1216)}
    else:
        dims = {"1:1": (512, 512), "16:9": (768, 432), "9:16": (432, 768)}

    width, height = dims.get(ratio, dims["1:1"])

    full_prompt = f"masterpiece, best quality, {style}, {user_prompt}"
    final_negative = f"{DEFAULT_NEGATIVE}, {user_negative}" if user_negative else DEFAULT_NEGATIVE

    generator = None
    if seed != -1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = {
        "prompt": full_prompt,
        "negative_prompt": final_negative,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "generator": generator
    }

    if current_model_type and "SD 3" in current_model_type:
        kwargs["max_sequence_length"] = 512

    try:
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gen_{timestamp}.png"
        file_path = os.path.join(app.static_folder, IMAGE_FOLDER, filename)
        image.save(file_path)

        return jsonify({'success': True, 'url': url_for('static', filename=f"{IMAGE_FOLDER}/{filename}")})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
