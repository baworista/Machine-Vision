import cv2
import numpy as np

# Wczytanie filmu
cap = cv2.VideoCapture('pliki/movingball.mp4')

# Utworzenie obiektu do zapisu filmu
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('nowePliki/tracked_movingball.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja do HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    params = {
        'LowH': 0,
        'LowS': 100,
        'LowV': 100,
        'UpH': 10,
        'UpS': 255,
        'UpV': 255,
        'KernelSize': 101
    }

    # Zakres kolorów
    lower_color = np.array([params['LowH'], params['LowS'], params['LowV']])
    upper_color = np.array([params['UpH'], params['UpS'], params['UpV']])

    # Maska
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Operacje morfologiczne
    kernel = np.ones((101, 101), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Znalezienie kontorow
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Wyszukiwanie centrum
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Dodanie znacznika w postaci kropki w miejscu środka masy
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

    # Zapisanie zmodyfikowanej klatki do nowego filmu
    out.write(frame)

    # Wyświetlenie klatki z naniesionymi znacznikami
    cv2.imshow('Tracked Moving Ball', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie obiektów
cap.release()
out.release()
cv2.destroyAllWindows()