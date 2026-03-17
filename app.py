import os
import torch
import base64
import datetime
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cloudflared import _run_cloudflared  # Библиотека для туннеля
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
app = Flask(__name__)

# --- Настройки модели (без изменений) ---
OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_FILENAME = "HassakuXL_Illustrious.safetensors"
DEFAULT_NEGATIVE = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name"

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("🚀 Инициализация Hassaku XL...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_FILENAME,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        print("✅ Модель готова!")

try:
    load_model()
except Exception as e:
    print(f"❌ Ошибка: {e}")

@app.route('/')
def index():
    return render_template('generator.html')

@app.route('/generate', methods=['POST'])
def generate():
    if pipe is None: return jsonify({'success': False, 'error': 'Модель не загружена'})
    data = request.json
    user_prompt = data.get('prompt', '').strip()
    ratio = data.get('ratio', '9:16')
    style = data.get('style', 'Anime')

    dims = {"1:1": (1024, 1024), "16:9": (1216, 680), "9:16": (832, 1216)}
    width, height = dims.get(ratio, (832, 1216))

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=f"masterpiece, best quality, {style}, {user_prompt}",
                negative_prompt=DEFAULT_NEGATIVE,
                num_inference_steps=28,
                guidance_scale=7.0,
                width=width, height=height
            ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'success': True, 'url': f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# --- Блок запуска туннеля и Flask ---
if __name__ == '__main__':
    print("🌍 Запуск туннеля Cloudflare...")
    # Эта функция автоматически запустит cloudflared и выведет ссылку в консоль
    _run_cloudflared(5000) 
    app.run(host='0.0.0.0', port=5000)
