import os
import datetime
import threading
import requests
import torch
from flask import Flask, render_template, request, jsonify, url_for
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# --- НАСТРОЙКИ ---
IMAGE_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
MODEL_URL = "https://civitai.com/api/download/models/2615702?type=Model&format=SafeTensor&size=pruned&fp=fp16"
MODEL_FILENAME = "HassakuXL_Illustrious.safetensors"
DEFAULT_NEGATIVE = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name"

app = Flask(__name__)
app.static_folder = 'static'

# Создаем папку для сохранения картинок
static_img_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
os.makedirs(static_img_dir, exist_ok=True)

# Глобальные переменные состояния
pipe = None
model_state = {
    'status': 'initializing',
    'progress': 0,
    'message': 'Сервер запущен. Проверка модели...'
}

# --- ФОНОВАЯ ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ---
def download_model():
    global model_state
    model_state['status'] = 'downloading'
    model_state['message'] = 'Скачивание модели Hassaku XL (около 6.5 ГБ)...'
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        with open(MODEL_FILENAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 4):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        model_state['progress'] = progress
    except Exception as e:
        model_state['status'] = 'error'
        model_state['message'] = f'Ошибка скачивания: {str(e)}'
        return False
    return True

def init_model_thread():
    global pipe, model_state
    
    # 1. Проверяем и скачиваем, если нужно
    if not os.path.exists(MODEL_FILENAME):
        success = download_model()
        if not success: 
            return
            
    # 2. Загружаем в память
    model_state['status'] = 'loading'
    model_state['progress'] = 100
    model_state['message'] = 'Загрузка весов модели в видеокарту (VRAM)...'
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temp_pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_FILENAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        temp_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(temp_pipe.scheduler.config)
        temp_pipe.to(device)
        
        if device == "cuda":
            temp_pipe.enable_model_cpu_offload() # Экономия видеопамяти
            
        pipe = temp_pipe
        model_state['status'] = 'ready'
        model_state['message'] = 'Нейросеть готова к работе!'
    except Exception as e:
        model_state['status'] = 'error'
        model_state['message'] = f'Ошибка загрузки: {str(e)}'

# Запускаем инициализацию в фоновом потоке
threading.Thread(target=init_model_thread, daemon=True).start()

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# --- РОУТЫ ---
@app.route('/status')
def get_status():
    return jsonify(model_state)

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

@app.route('/generate', methods=['POST'])
def generate_art():
    if model_state['status'] != 'ready' or pipe is None:
        return jsonify({'success': False, 'error': 'Нейросеть еще не готова. Дождитесь загрузки.'}), 503

    data = request.json
    user_prompt = data.get('prompt', '').strip()
    ratio = data.get('ratio', '9:16')
    style = data.get('style', 'Anime')

    if not user_prompt:
        return jsonify({'success': False, 'error': 'Промпт пустой'}), 400

    dims = {"1:1": (1024, 1024), "16:9": (1216, 680), "9:16": (832, 1216)}
    width, height = dims.get(ratio, (832, 1216))

    full_prompt = f"masterpiece, best quality, {style}, {user_prompt}"

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=full_prompt,
                negative_prompt=DEFAULT_NEGATIVE,
                num_inference_steps=28,
                guidance_scale=7.0,
                width=width,
                height=height
            ).images[0]

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gen_{timestamp}.png"
        file_path = os.path.join(app.static_folder, IMAGE_FOLDER, filename)
        image.save(file_path)

        return jsonify({'success': True, 'url': url_for('static', filename=f"{IMAGE_FOLDER}/{filename}")})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
