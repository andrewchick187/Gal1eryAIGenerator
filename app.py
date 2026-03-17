import os
import torch
import base64
import datetime
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

app = Flask(__name__)

# --- Настройки модели ---
MODEL_FILENAME = "HassakuXL_Illustrious.safetensors"
# Базовый негативный промпт для качества
DEFAULT_NEGATIVE = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name"

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("🚀 Инициализация Hassaku XL в GPU...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_FILENAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        print("✅ Модель готова к работе!")

# Пробуем загрузить модель сразу
try:
    load_model()
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}. Убедитесь, что файл {MODEL_FILENAME} на месте.")

@app.route('/')
def index():
    return render_template('generator.html')

@app.route('/generate', methods=['POST'])
def generate():
    if pipe is None:
        return jsonify({'success': False, 'error': 'Модель не загружена'})

    data = request.json
    user_prompt = data.get('prompt', '').strip()
    ratio = data.get('ratio', '1:1')
    style = data.get('style', 'Anime')

    if not user_prompt:
        return jsonify({'success': False, 'error': 'Промпт пустой'})

    # Настройка размеров на основе вашего кода (SDXL любит большие разрешения)
    dimensions = {
        "1:1": (1024, 1024),
        "16:9": (1216, 680),
        "9:16": (832, 1216) # Как в вашем коде Colab
    }
    width, height = dimensions.get(ratio, (832, 1216))

    # Формируем полный промпт
    full_prompt = f"masterpiece, best quality, {style}, {user_prompt}"

    try:
        # Генерация (логика из вашего Colab)
        with torch.inference_mode():
            image = pipe(
                prompt=full_prompt,
                negative_prompt=DEFAULT_NEGATIVE,
                num_inference_steps=28,
                guidance_scale=7.0,
                width=width,
                height=height
            ).images[0]

        # Конвертируем в Base64, чтобы отправить сразу на страницу без сохранения мусора на диске
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True, 
            'url': f"data:image/png;base64,{img_base64}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
