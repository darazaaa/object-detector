import cv2
import numpy as np
import joblib
import os

# Загрузка модели
model = joblib.load('model.pkl')
classes = os.listdir('data')  # ['cup', 'book', ...]

# Запуск камеры
cap = cv2.VideoCapture(0)  # 0 — по умолчанию первая камера

print("📷 Запущена веб-камера. Нажми 'Esc' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Центрируем область интереса (центр кадра)
    h, w = frame.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    size = 200  # размер области (можно увеличить)

    x1 = max(cx - size//2, 0)
    y1 = max(cy - size//2, 0)
    x2 = min(cx + size//2, w)
    y2 = min(cy + size//2, h)

    roi = frame[y1:y2, x1:x2]

    # Преобразование
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    flat = resized.flatten().reshape(1, -1)

    # Предсказание
    pred = model.predict(flat)[0]
    label = classes[pred]

    # Показываем рамку и метку
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f'{label}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("📦 Object Detector", frame)

    # Esc для выхода
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
