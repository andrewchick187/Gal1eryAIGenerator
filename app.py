import os
import datetime
import torch
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# --- НАСТРОЙКИ ---
IMAGE_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
MODEL_FILENAME = "HassakuXL_Illustrious.safetensors"

app = Flask(__name__)
app.static_folder = 'static'

# Создаем папки
static_img_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
os.makedirs(static_img_dir, exist_ok=True)

# --- ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТИ ---
print("🚀 Загрузка нейросети (это может занять время)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Загружаем модель (убедитесь, что файл лежит в папке с проектом)
    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_FILENAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    print(f"✅ Модель загружена на {device}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    pipe = None

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def gallery():
    image_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
    search_query = request.args.get('q', '').strip().lower()
    sort_by = request.args.get('sort', 'date_desc')
    images = []
    
    if os.path.exists(image_dir):
        files = os.listdir(image_dir)
        for f in files:
            filepath = os.path.join(image_dir, f)
            if os.path.isfile(filepath) and allowed_file(f):
                if search_query and search_query not in f.lower(): continue
                stats = os.stat(filepath)
                images.append({
                    'name': f,
                    'url': f"{IMAGE_FOLDER}/{f}",
                    'size': f"{stats.st_size / 1024:.1f} KB",
                    'date': datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%d.%m.%Y %H:%M"),
                    'timestamp': stats.st_mtime
                })

    if sort_by == 'date_desc': images.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_by == 'date_asc': images.sort(key=lambda x: x['timestamp'])
    elif sort_by == 'name': images.sort(key=lambda x: x['name'])

    return render_template('index.html', images=images, search_query=search_query, sort_by=sort_by)

@app.route('/generator')
def generator_page():
    return render_template('generator.html')

@app.route('/generate', methods=['POST'])
def generate_art():
    if pipe is None:
        return jsonify({'success': False, 'error': 'Модель не загружена'}), 500

    data = request.json
    prompt = data.get('prompt', '')
    style = data.get('style', 'Реализм')
    ratio = data.get('ratio', '1:1')

    if not prompt:
        return jsonify({'success': False, 'error': 'Промпт пустой'}), 400

    # Настройка размеров
    dims = {"1:1": (1024, 1024), "16:9": (1216, 640), "9:16": (640, 1216)}
    width, height = dims.get(ratio, (1024, 1024))

    # Стилизация промпта
    styles_map = {
        "Аниме": "masterpiece, best quality, anime style, highres, illustrious",
        "3D Рендер": "masterpiece, best quality, unreal engine 5, octane render, 4k",
        "Масло": "oil painting, textured brush strokes, masterpiece",
        "Реализм": "photorealistic, ultra highres, sharp focus, 8k"
    }
    full_prompt = f"{styles_map.get(style, '')}, {prompt}"
    negative_prompt = "low quality, worst quality, bad anatomy, bad hands, text, error, blurry"

    try:
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            width=width,
            height=height
        ).images[0]

        # Сохранение
        filename = f"gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(app.static_folder, IMAGE_FOLDER, filename)
        image.save(file_path)

        return jsonify({'success': True, 'url': url_for('static', filename=f"{IMAGE_FOLDER}/{filename}")})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    # ... (код загрузки из вашего исходника остается без изменений)
    pass

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000) # debug=False лучше для GPU