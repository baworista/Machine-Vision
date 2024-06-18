import cv2
import numpy as np

# Wczytanie obrazu okularów, który zostanie nałożony na twarz
glasses = cv2.imread('glasses.png', -1)

# Funkcja do nakładania obrazów z kanałem alfa
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Nakładanie obrazu z przezroczystością"""
    # Obliczanie wymiarów obrazu nakładanego
    h, w = img_overlay.shape[:2]

    # Wybieranie obszaru w tle do modyfikacji
    overlay_image = img[y:y+h, x:x+w]

    # Przekształcanie maski alfa na skale od 0 do 1
    alpha = alpha_mask / 255.0

    # Nakładanie obrazu
    for c in range(0, 3):
        overlay_image[:, :, c] = (alpha * img_overlay[:, :, c] +
                                  (1 - alpha) * overlay_image[:, :, c])

    img[y:y+h, x:x+w] = overlay_image

# Ładowanie modelu do detekcji twarzy z OpenCV dnn
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

while True:
    # Odczytywanie ramki z kamery
    ret, frame = cap.read()

    # Przygotowanie obrazu do detekcji twarzy
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Detekcja twarzy
    net.setInput(blob)
    detections = net.forward()

    # Przetwarzanie wykrytych twarzy
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Skalowanie obrazu okularów do szerokości twarzy
            face_width = x1 - x
            aspect_ratio = glasses.shape[1] / glasses.shape[0]  # szerokość/wysokość
            glasses_height = int(face_width / aspect_ratio)

            glasses_resized = cv2.resize(glasses, (face_width, glasses_height))
            alpha_mask = glasses_resized[:, :, 3]
            glasses_resized = glasses_resized[:, :, 0:3]

            # Nakładanie obrazu okularów na twarz
            y_offset = y + int((y1 - y) / 5)  # Wyżej niż wcześniej
            overlay_image_alpha(frame, glasses_resized, x, y_offset, alpha_mask)

    # Wyświetlanie obrazu
    cv2.imshow('Face with Glasses', frame)

    # Wyjście po naciśnięciu klawisza 'q' lub 'ESC'
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 to kod ASCII dla klawisza ESC
        break

# Zwolnienie kamery i zamknięcie okienek
cap.release()
cv2.destroyAllWindows()
