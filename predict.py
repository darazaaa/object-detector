import cv2
import numpy as np
import joblib
import os

# Загружаем модель
model = joblib.load('model.pkl')

# Папка с классами (нужна для расшифровки меток)
classes = os.listdir('data')  # ['cup', ...]

# Загружаем новое изображение
img_path = 'test.jpeg'  # ← положи сюда новое фото меня формат на jpg и т.п в зависимости от формата фото
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img_flat = img.flatten().reshape(1, -1)

# Предсказание
pred = model.predict(img_flat)
print("Предсказанный класс:", classes[pred[0]])
