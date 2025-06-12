import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self):
        self.history = None

    def evaluate(self, model, generator):
        """Оцінка моделі."""
        predictions = model.predict(generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = generator.classes
        report = classification_report(y_true, y_pred, target_names=list(generator.class_indices.keys()))
        matrix = confusion_matrix(y_true, y_pred)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)
        return report, matrix

    def plot_training_history(self, history):
        """Виведення графіків точності та втрат."""
        self.history = history

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('static/training_history.png')

    def generate_confusion_matrix(self, model, generator):
        """Формування матриці сплутування."""
        predictions = model.predict(generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = generator.classes
        matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(generator.class_indices))
        plt.xticks(tick_marks, list(generator.class_indices.keys()), rotation=45)
        plt.yticks(tick_marks, list(generator.class_indices.keys()))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('static/confusion_matrix.png')
        return matrix

    def compute_metrics(self, model, generator):
        """Обчислення статистичних метрик."""
        predictions = model.predict(generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = generator.classes

        report = classification_report(y_true, y_pred, target_names=list(generator.class_indices.keys()), output_dict=True)
        metrics = {
            'overall_accuracy': report['accuracy'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg']
        }
        print("Detailed Metrics:", metrics)
        return metrics