from flask import Flask, request, render_template, jsonify
import os
from modules.dataset_handler import DatasetHandler
from modules.model_trainer import ModelTrainer
from modules.evaluator import Evaluator
from modules.predictor import Predictor

app = Flask(__name__)

# Константи
UPLOAD_FOLDER = 'static/uploads/'
DATASET_DIR = 'architectural_styles'
MODEL_PATH = 'models/mymodel.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ініціалізація компонентів
dataset_handler = DatasetHandler(dataset_dir=DATASET_DIR, upload_folder=UPLOAD_FOLDER)
model_trainer = ModelTrainer(dataset_path=DATASET_DIR, model_path=MODEL_PATH)
evaluator = Evaluator()
predictor = Predictor(model_path=MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не завантажено'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не вибрано'}), 400

    file_path = dataset_handler.handle_dataset_upload(file)
    if file_path:
        result = predictor.predict_image(file_path)
        if 'error' not in result:
            conclusion = predictor.classification_conclusion(result)
            style_info = predictor.get_style_info(result['class'])
            return jsonify({
                'prediction': result['class'],
                'confidence': result['confidence'],
                'conclusion': conclusion,
                'style_info': style_info,
                'all_probabilities': result['all_probabilities'],
                'image_path': file_path
            })
        return jsonify({'error': result['error']}), 400
    return jsonify({'error': 'Неприпустимий формат файлу'}), 400

@app.route('/train', methods=['POST'])
def train():
    epochs = int(request.form.get('epochs', 10))
    batch_size = int(request.form.get('batch_size', 32))
    trainable_layers = int(request.form.get('trainable_layers', 0))
    history = model_trainer.train(epochs, batch_size, trainable_layers)
    evaluator.plot_training_history(history)
    train_gen, val_gen = dataset_handler.load_dataset(batch_size=batch_size)
    evaluator.generate_confusion_matrix(model_trainer.model, val_gen)
    metrics = evaluator.compute_metrics(model_trainer.model, val_gen)
    model_trainer.view_logs()
    return render_template('index.html', message=f'Модель успішно навчена. Метрики: {metrics}')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'dataset' not in request.files:
        return render_template('index.html', error='Датасет не завантажено')
    file = request.files['dataset']
    if file and file.filename.endswith('.zip'):
        file_path = dataset_handler.handle_dataset_upload(file)
        if file_path:
            return render_template('index.html', message='Датасет завантажено')
    return render_template('index.html', error='Неприпустимий формат датасету')

@app.route('/dataset_stats', methods=['GET'])
def dataset_stats():
    stats = dataset_handler.get_dataset_stats()
    return jsonify(stats)

@app.route('/balance_dataset', methods=['POST'])
def balance_dataset():
    target_count = request.form.get('target_count', type=int)
    stats = dataset_handler.balance_dataset(target_count)
    return jsonify(stats)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    train_gen, val_gen = dataset_handler.load_dataset(batch_size=32)
    evaluator.generate_confusion_matrix(model_trainer.model, val_gen)
    report, matrix = evaluator.evaluate(model_trainer.model, val_gen)
    metrics = evaluator.compute_metrics(model_trainer.model, val_gen)
    return render_template('index.html', evaluation_report=f"Report:\n{report}\nMetrics:\n{metrics}", message="Модель оцінена")

if __name__ == '__main__':
    app.run(debug=True)