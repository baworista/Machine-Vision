import cv2
import numpy as np

# Funkcja przetwarzająca obraz dla różnych parametrów
def process_frame(frame_num):
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Ustaw numer klatki
    ret, frame = cap.read()
    if ret:
        # Konwersja do HSV
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Zakres kolorów
        lower_color = np.array([params['LowH'], params['LowS'], params['LowV']])
        upper_color = np.array([params['UpH'], params['UpS'], params['UpV']])

        # Maska
        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        #  Operacje morfologiczne
        kernel = np.ones((params['KernelSize'], params['KernelSize']), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Znalezienie kontorow
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Wyszukiwanie centrum
        image_with_dots = frame.copy()
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(image_with_dots, (cx, cy), 5, (255, 255, 255), -1)

        # Wyświetlenie obrazu oraz maski
        cv2.imshow('image', cv2.hconcat([image_with_dots, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

# Funkcja zmieniająca wartość parametru LowH
def change_LowH(value):
    params['LowH'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Funkcja zmieniająca wartość parametru LowS
def change_LowS(value):
    params['LowS'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Funkcja zmieniająca wartość parametru LowV
def change_LowV(value):
    params['LowV'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Funkcja zmieniająca wartość parametru UpH
def change_UpH(value):
    params['UpH'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Funkcja zmieniająca wartość parametru UpS
def change_UpS(value):
    params['UpS'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Funkcja zmieniająca wartość parametru UpV
def change_UpV(value):
    params['UpV'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Funkcja zmieniająca rozmiar jądra operacji morfologicznych
def change_kernel_size(value):
    params['KernelSize'] = value
    process_frame(cv2.getTrackbarPos('Frame', 'Tracked Moving Ball'))

# Wczytanie filmu
cap = cv2.VideoCapture('pliki/blue_ball.mp4')

# Czerwona kula
# params = {
#     'LowH': 0,
#     'LowS': 85,
#     'LowV': 100,
#     'UpH': 10,
#     'UpS': 255,
#     'UpV': 255,
#     'KernelSize': 101
# }

# Niebieska kula
params = {
    'LowH': 116,
    'LowS': 88,
    'LowV': 50,
    'UpH': 145,
    'UpS': 255,
    'UpV': 255,
    'KernelSize': 101
}


# Wybór numeru klatki
frame_num = 0
cv2.namedWindow('Tracked Moving Ball')
cv2.createTrackbar('Frame', 'Tracked Moving Ball', frame_num, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, lambda x: process_frame(x))

# Ustawienie suwaków dla parametrów HSV i rozmiaru jądra
cv2.createTrackbar('LowH', 'Tracked Moving Ball', params['LowH'], 255, change_LowH)
cv2.createTrackbar('LowS', 'Tracked Moving Ball', params['LowS'], 255, change_LowS)
cv2.createTrackbar('LowV', 'Tracked Moving Ball', params['LowV'], 255, change_LowV)
cv2.createTrackbar('UpH', 'Tracked Moving Ball', params['UpH'], 255, change_UpH)
cv2.createTrackbar('UpS', 'Tracked Moving Ball', params['UpS'], 255, change_UpS)
cv2.createTrackbar('UpV', 'Tracked Moving Ball', params['UpV'], 255, change_UpV)
cv2.createTrackbar('Kernel Size', 'Tracked Moving Ball', params['KernelSize'], 200, change_kernel_size)

# Wywołanie funkcji do przetwarzania pierwszej klatki
process_frame(frame_num)

# Pętla wyświetlająca wideo w nieskończoność
while True:
    key = cv2.waitKey(1)
    if key == 27: # Esc lub zamknięcie okna
        break

# Zwolnienie obiektów
cap.release()
cv2.destroyAllWindows()
