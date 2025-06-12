import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import logging

class ModelTrainer:
    def __init__(self, dataset_path='dataset', model_path='models/mymodel.h5', img_size=224):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler('training.log'))

    def preprocess_dataset(self):
        """Попередня обробка датасету."""
        self.logger.info("Початок попередньої обробки датасету...")
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            fill_mode='nearest'
        )
        train_gen = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            subset='training',
            class_mode='categorical'
        )
        val_gen = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            subset='validation',
            class_mode='categorical'
        )
        self.logger.info("Датасет підготовлено.")
        return train_gen, val_gen

    def build_model(self, trainable_layers=0):
        """Побудова моделі з вибором стратегії навчання."""
        self.logger.info("Побудова моделі...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.img_size, self.img_size, 3))

        if trainable_layers > 0:
            self.logger.info(f"Використання fine-tuning для {trainable_layers} шарів.")
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        else:
            self.logger.info("Використання feature extraction (всі шари заморожені).")
            for layer in base_model.layers:
                layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        output = Dense(24, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, epochs, batch_size, trainable_layers=0):
        """Навчання моделі."""
        self.logger.info(f"Початок навчання з epochs={epochs}, batch_size={batch_size}, trainable_layers={trainable_layers}")
        train_gen, val_gen = self.preprocess_dataset()

        if self.model is None:
            self.build_model(trainable_layers)

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        self.logger.info(f"Модель збережено за шляхом {self.model_path}")
        return history

    def view_logs(self):
        """Перегляд логів навчання."""
        self.logger.info("Перегляд логів навчання...")
        with open('training.log', 'r') as f:
            print(f.read())