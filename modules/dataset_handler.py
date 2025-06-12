import os
import zipfile
import numpy as np
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from werkzeug.utils import secure_filename

class DatasetHandler:
    def __init__(self, dataset_dir='dataset', upload_folder='static/uploads/'):
        self.dataset_dir = dataset_dir
        self.upload_folder = upload_folder
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.upload_folder, exist_ok=True)

    def allowed_file(self, filename):
        """Перевірка дозволеного розширення"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

    def handle_dataset_upload(self, file):
        if file and (self.allowed_file(file.filename) or file.filename.endswith('.zip')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.upload_folder, filename)
            file.save(file_path)
            if file.filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.dataset_dir)
            return file_path
        return None

    def get_dataset_stats(self):
        """Виведення статистики по датасету."""
        if not os.path.exists(self.dataset_dir):
            return {"error": "Датасет не знайдено"}

        class_counts = defaultdict(int)
        total_images = 0

        for class_name in os.listdir(self.dataset_dir):
            class_path = os.path.join(self.dataset_dir, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = num_images
                total_images += num_images

        return {
            "class_counts": dict(class_counts),
            "total_images": total_images,
            "num_classes": len(class_counts)
        }

    def balance_dataset(self, target_count=None):
        """Балансування датасету: аугментація та обрізання."""
        stats = self.get_dataset_stats()
        if "error" in stats:
            return stats

        class_counts = stats["class_counts"]
        total_images = stats["total_images"]
        num_classes = stats["num_classes"]

        if target_count is None:
            target_count = total_images // num_classes

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        for class_name, count in class_counts.items():
            class_path = os.path.join(self.dataset_dir, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if count < target_count:
                num_to_generate = target_count - count
                for img_name in images[:num_to_generate]:
                    img_path = os.path.join(class_path, img_name)
                    img = load_img(img_path)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)
                    i = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=class_path, save_prefix='aug', save_format='jpg'):
                        i += 1
                        if i >= 1:
                            break
            elif count > target_count:
                num_to_remove = count - target_count
                for img_name in images[:num_to_remove]:
                    img_path = os.path.join(class_path, img_name)
                    os.remove(img_path)

        return self.get_dataset_stats()

    def load_dataset(self, batch_size=32, validation_split=0.15):
        """Завантаження та підготовка датасету."""
        train_ds = image_dataset_from_directory(
            self.dataset_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size
        )
        val_ds = image_dataset_from_directory(
            self.dataset_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size
        )
        return train_ds, val_ds