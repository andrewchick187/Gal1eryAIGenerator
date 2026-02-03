import os
import datetime
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# Настройки
IMAGE_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}

app = Flask(__name__)
app.static_folder = 'static'

# Создаем папку, если её нет
static_img_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
if not os.path.exists(static_img_dir):
    os.makedirs(static_img_dir)

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def gallery():
    image_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
    
    search_query = request.args.get('q', '').strip().lower()
    sort_by = request.args.get('sort', 'date_desc')

    images = []
    
    try:
        files = os.listdir(image_dir)
    except FileNotFoundError:
        files = []

    for f in files:
        filepath = os.path.join(image_dir, f)
        if os.path.isfile(filepath) and allowed_file(f):
            # Фильтрация поиска
            if search_query and search_query not in f.lower():
                continue
            
            # Получаем метаданные
            stats = os.stat(filepath)
            size_kb = stats.st_size / 1024
            mod_time = datetime.datetime.fromtimestamp(stats.st_mtime)
            
            images.append({
                'name': f,
                'url': f"{IMAGE_FOLDER}/{f}",
                'size': f"{size_kb:.1f} KB",
                'date': mod_time.strftime("%d.%m.%Y %H:%M"),
                'timestamp': stats.st_mtime
            })

    # Сортировка
    if sort_by == 'date_desc':
        images.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_by == 'date_asc':
        images.sort(key=lambda x: x['timestamp'])
    elif sort_by == 'name':
        images.sort(key=lambda x: x['name'])

    return render_template(
        'index.html', 
        images=images, 
        search_query=search_query, 
        sort_by=sort_by
    )

# --- НОВЫЙ МАРШРУТ ГЕНЕРАТОРА ---
@app.route('/generator')
def generator_page():
    return render_template('generator.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Нет файла'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_dir = os.path.join(app.static_folder, IMAGE_FOLDER)
        file_path = os.path.join(image_dir, filename)

        counter = 1
        original_filename = filename
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            file_path = os.path.join(image_dir, filename)
            counter += 1

        file.save(file_path)
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Формат не поддерживается'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)