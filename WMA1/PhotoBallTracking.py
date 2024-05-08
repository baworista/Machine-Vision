import cv2
import numpy as np

def change_kernel_size(value):
    global kernel_size, kernel
    kernel_size = value
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    process_image(params)

kernel_size = 101

# Funkcja przetwarzająca obraz dla różnych parametrów
def process_image(params):
    global hsv_image, mask, image

    # Zaktualizuj parametry
    lower_color = np.array([params['LowH'], params['LowS'], params['LowV']])
    upper_color = np.array([params['UpH'], params['UpS'], params['UpV']])

    # Utwórz maskę koloru piłki
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Poprawa jakości obrazu przez zastosowanie operacji morfologicznych
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Znajdź kontury obiektów na obrazie
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nanieś znaczniki na obrazie
    image_with_dots = image.copy()
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(image_with_dots, (cx, cy), 5, (255, 255, 255), -1)

    # Wyświetl przetworzony obraz
    cv2.imshow('image', cv2.hconcat([image_with_dots, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))


# Funkcja zmieniająca wartość parametru LowH
def change_LowH(value):
    params['LowH'] = value
    process_image(params)


# Funkcja zmieniająca wartość parametru LowS
def change_LowS(value):
    params['LowS'] = value
    process_image(params)


# Funkcja zmieniająca wartość parametru LowV
def change_LowV(value):
    params['LowV'] = value
    process_image(params)


# Funkcja zmieniająca wartość parametru UpH
def change_UpH(value):
    params['UpH'] = value
    process_image(params)


# Funkcja zmieniająca wartość parametru UpS
def change_UpS(value):
    params['UpS'] = value
    process_image(params)


# Funkcja zmieniająca wartość parametru UpV
def change_UpV(value):
    params['UpV'] = value
    process_image(params)


# Wczytaj obraz
image = cv2.imread('pliki/ball3.png')

# Konwersja obrazu do przestrzeni kolorów HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Początkowe parametry
params = {
    'LowH': 0,
    'LowS': 100,
    'LowV': 150,
    'UpH': 10,
    'UpS': 255,
    'UpV': 255
}

# Przetwórz obraz z początkowymi parametrami
process_image(params)

# Ustawienie suwaków dla parametrów HSV
cv2.namedWindow('Center of the object')
cv2.createTrackbar('LowH', 'Center of the object', params['LowH'], 255, change_LowH)
cv2.createTrackbar('LowS', 'Center of the object', params['LowS'], 255, change_LowS)
cv2.createTrackbar('LowV', 'Center of the object', params['LowV'], 255, change_LowV)
cv2.createTrackbar('UpH', 'Center of the object', params['UpH'], 255, change_UpH)
cv2.createTrackbar('UpS', 'Center of the object', params['UpS'], 255, change_UpS)
cv2.createTrackbar('UpV', 'Center of the object', params['UpV'], 255, change_UpV)
cv2.createTrackbar('Kernel Size', 'Center of the object', kernel_size, 200, change_kernel_size)

# Pętla wyświetlająca obraz w nieskończoność
while True:
    key = cv2.waitKey(1)
    if key == 27:  # Esc - wyjście z pętli
        break

cv2.destroyAllWindows()
