import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

ARCHITECTURE_CLASSES = [
    'Achaemenid architecture', 'American Foursquare architecture', 'American craftsman style',
    'Ancient Egyptian architecture', 'Art Deco architecture', 'Art Nouveau architecture',
    'Baroque architecture', 'Bauhaus architecture', 'Beaux-Arts architecture',
    'Byzantine architecture', 'Chicago school architecture', 'Colonial architecture',
    'Deconstructivism', 'Edwardian architecture', 'Georgian architecture',
    'Greek Revival architecture', 'International style', 'Novelty architecture',
    'Palladian architecture', 'Postmodern architecture', 'Queen Anne architecture',
    'Romanesque architecture', 'Russian Revival architecture', 'Tudor Revival architecture'
]

ARCHITECTURE_INFO = {
    'Achaemenid_architecture': 'Походить із Стародавньої Персії (550–330 рр. до н.е.), характеризується величними палацами, масивними колонами та складними рельєфами, як-от у Персеполі.',
    'American_Foursquare_architecture': 'Американський стиль 20-го століття (1890-ті–1930-ті роки), відомий простими прямокутними дизайнами з чотирма кімнатами на кожному поверсі та акцентом на функціональність.',
    'American_craftsman_style': 'Виник у ранньому 20 столітті, підкреслює ручну обробку, натуральні матеріали та низькі схили дахів, часто використовується в бунгало.',
    'Ancient_Egyptian_architecture': 'Відомий монументальними спорудами, такими як піраміди та храми (приблизно 3100–30 рр. до н.е.), виконаними з масивних кам’яних блоків із символічними мотивами.',
    'Art_Deco_architecture': 'Стиль 1920–1930-х років із сміливими геометричними формами, яскравими кольорами та розкішними матеріалами, як-от хмарочоси, наприклад, Chrysler Building.',
    'Art_Nouveau_architecture': 'Популярний наприкінці 19 – на початку 20 століття, характеризується плавними органічними формами та складною роботою з заліза, як у творах Гауді.',
    'Baroque_architecture': 'Європейський стиль 17–18 століть із драматичними, оздобленими дизайнами, вигнутими формами та величчю, наприклад, у Базиліці Святого Петра.',
    'Bauhaus_architecture': 'Модерністський рух 20 століття (1919–1933), який зосереджується на мінімалізмі, функціональності та інтеграції мистецтва та технологій.',
    'Beaux-Arts_architecture': 'Стиль кінця 19 століття з класичною симетрією, складними деталями та величчю, часто використовується в громадських будівлях, таких як музеї.',
    'Byzantine_architecture': 'Починаючи з 5 століття, відомий куполами, мозаїками та централізованими планами, як-от у Софії Константинопольській у Стамбулі.',
    'Chicago_school_architecture': 'Стиль кінця 19 століття, який започаткував дизайн хмарочосів зі сталевими рамами та великими вікнами, як-от Home Insurance Building.',
    'Colonial_architecture': 'Відображає європейські стилі, адаптовані для колоній (16–19 століття), часто з простими симетричними дизайнами та місцевими матеріалами.',
    'Deconstructivism': 'Стиль кінця 20 століття з фрагментованими, нелінійними формами, який кидає виклик традиційній архітектурі, як-от музей Гуггенхайма в Більбао.',
    'Edwardian_architecture': 'Британський стиль початку 20 століття (1901–1910), що поєднує класичні елементи з легшими та елегантнішими дизайнами.',
    'Georgian_architecture': 'Стиль 18 століття з симетрією, цегляною кладкою та класичними пропорціями, поширений у Великобританії та Америці.',
    'Greek_Revival_architecture': 'Стиль 19 століття, натхненний Стародавньою Грецією, з колонами, симетрією та фронтонами, як-от у Капітолії США.',
    'International_style': 'Модерністський стиль 20 століття (1920-ті–1970-ті) з чистими лініями, склом і сталлю, де форма випливає з функції.',
    'Novelty_architecture': 'Будівлі, створені у формі об’єктів або форм (наприклад, гарячий собачок у вигляді гарячого собачки), часто для комерційних цілей.',
    'Palladian_architecture': 'Натхненний 16-столітнім архітектором Андреа Палладіо, з симетрією, класичними ордерами та віллами, як-от Вілла Ротонда.',
    'Postmodern_architecture': 'Реакція на модернізм наприкінці 20 століття з грайливими, еклектичними дизайнами, як-от будівля в Портленді.',
    'Queen_Anne_architecture': 'Стиль кінця 19 століття з асиметричними дизайнами, башточками та декоративними елементами, популярний у вікторіанських будинках.',
    'Romanesque_architecture': 'Середньовічний стиль (10–12 століття) з товстими стінами, заокругленими арками та масивними формами, як у багатьох європейських церквах.',
    'Russian_Revival_architecture': 'Стиль 19 століття, що поєднує традиційні російські елементи, як-от цибулинні куполи, із сучасним будівництвом, як у Соборі Василя Блаженного.',
    'Tudor_Revival_architecture': 'Стиль 19–20 століть, що імітує англійську тудорську архітектуру, з напівдерев’яними конструкціями, крутими дахами та оздобленими димоходами.'
}

class Predictor:
    def __init__(self, model_path, img_size=224):
        self.model = load_model(model_path)
        self.img_size = img_size

    def load_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Помилка завантаження зображення: {e}")
            return None

    def validate_image(self, img):
        if img is None:
            return False, "Зображення не завантажено."
        if img.size[0] < 10 or img.size[1] < 10:
            return False, "Зображення занадто маленьке."
        if img.mode != 'RGB':
            return False, "Зображення має бути у форматі RGB."
        return True, "Зображення коректне."

    def preprocess_image(self, img):
        try:
            img = img.resize((self.img_size, self.img_size))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Помилка обробки зображення: {e}")
            return None

    def predict_image(self, image_path):
        img = self.load_image(image_path)
        is_valid, validation_message = self.validate_image(img)
        if not is_valid:
            return {'error': validation_message}

        img_array = self.preprocess_image(img)
        if img_array is None:
            return {'error': 'Помилка при обробці зображення'}

        predictions = self.model.predict(img_array)
        predicted_class = ARCHITECTURE_CLASSES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        all_probabilities = [(class_name, prob * 100) for class_name, prob in zip(ARCHITECTURE_CLASSES, predictions[0])]

        return {
            'class': predicted_class,
            'confidence': f'{confidence:.2f}%',
            'all_probabilities': all_probabilities
        }

    def classification_conclusion(self, prediction_result):
        if 'error' in prediction_result:
            return f"Класифікація не виконана: {prediction_result['error']}"

        predicted_class = prediction_result['class']
        confidence = float(prediction_result['confidence'].replace('%', ''))

        conclusion = f"Зображення класифіковано як стиль '{predicted_class}' з впевненістю {confidence:.2f}%.\n"
        if confidence >= 80:
            conclusion += "Висока впевненість: результат є дуже надійним."
        elif confidence >= 50:
            conclusion += "Середня впевненість: результат ймовірний, але варто перевірити."
        else:
            conclusion += "Низька впевненість: результат ненадійний, рекомендується повторна перевірка."
        return conclusion

    def get_style_info(self, predicted_class):
        style_key = predicted_class.replace(' ', '_')
        return ARCHITECTURE_INFO.get(style_key, "Інформація про цей стиль відсутня.")