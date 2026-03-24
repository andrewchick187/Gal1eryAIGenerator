import os
import datetime
import torch
from flask import Flask, render_template, request, jsonify, url_for
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# --- НАСТРОЙКИ ---
IMAGE_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
MODEL_FILENAME = "HassakuXL_Illustrious.safetensors"
DEFAULT_NEGATIVE = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name"

app = Flask(__name__)
app.static_folder = 'static'

# Создаем папку для сохранения картинок
static_img_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
os.makedirs(static_img_dir, exist_ok=True)

# --- ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТИ ---
print("🚀 Загрузка нейросети (это может занять время)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_FILENAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    print(f"✅ Модель успешно загружена на {device.upper()}!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    pipe = None

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# --- РОУТЫ (СТРАНИЦЫ) ---

@app.route('/')
def gallery():
    # Галерея всех сгенерированных изображений
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
    # Сортировка от новых к старым
    images.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('index.html', images=images)

@app.route('/generator')
def generator_page():
    return render_template('generator.html')

@app.route('/generate', methods=['POST'])
def generate_art():
    if pipe is None:
        return jsonify({'success': False, 'error': 'Модель не загружена. Проверьте логи сервера.'}), 500

    data = request.json
    user_prompt = data.get('prompt', '').strip()
    ratio = data.get('ratio', '9:16')
    style = data.get('style', 'Anime')

    if not user_prompt:
        return jsonify({'success': False, 'error': 'Промпт пустой'}), 400

    # Настройка размеров (из нового файла)
    dims = {"1:1": (1024, 1024), "16:9": (1216, 680), "9:16": (832, 1216)}
    width, height = dims.get(ratio, (832, 1216))

    # Формируем финальный промпт
    full_prompt = f"masterpiece, best quality, {style}, {user_prompt}"

    try:
        # Используем inference_mode для экономии памяти и ускорения
        with torch.inference_mode():
            image = pipe(
                prompt=full_prompt,
                negative_prompt=DEFAULT_NEGATIVE,
                num_inference_steps=28,
                guidance_scale=7.0,
                width=width,
                height=height
            ).images[0]

        # Сохранение на диск (вместо Base64)
        filename = f"gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(app.static_folder, IMAGE_FOLDER, filename)
        image.save(file_path)

        # Возвращаем URL до файла
        return jsonify({'success': True, 'url': url_for('static', filename=f"{IMAGE_FOLDER}/{filename}")})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# --- ЗАПУСК СЕРВЕРА ---
if __name__ == '__main__':
    print("🌍 Запуск локального сервера...")
    # Запускаем строго на локальном хосте. Туннель Cloudflare запустим отдельно!
    app.run(debug=False, host='127.0.0.1', port=5000)
