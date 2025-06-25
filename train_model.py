# 🔹 Шаг 6–10: Загрузка данных, обучение и сохранение модели

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Папка с изображениями
data_dir = 'data'
classes = os.listdir(data_dir)  # ['cup']
X = []
y = []

# Загружаем изображения
for label, class_name in enumerate(classes):
    folder_path = os.path.join(data_dir, class_name)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

# Делим на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучаем модель
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Проверяем точность
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Точность модели:", acc)

# Сохраняем модель
joblib.dump(model, 'model.pkl')
print("Модель сохранена в файл model.pkl ✅")
