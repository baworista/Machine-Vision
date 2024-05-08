import cv2 as cv
import numpy as np
import os


def is_inside(x, y, x0, y0, x1, y1):
    """Sprawdza, czy punkt (x, y) znajduje się wewnątrz prostokąta o rogach (x0, y0) i (x1, y1)."""
    return x0 < x < x1 and y0 < y < y1


def detect_circles(img):
    """Wykrywa okręgi na podstawie obrazu img."""
    # Rozmycie Gaussa
    blurred_img = cv.GaussianBlur(img, (3, 3), 0)

    # Filtr medianowy
    blurred_img = cv.medianBlur(blurred_img, 3)

    # Konwertacja obrazu do skali szarości
    gray_blurred = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)

    # Wyszukiwanie okregow, param1 i param2 odpowiadaja czułości detekcji i minimalnej odległości między centrami okręgów
    circles = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, 20,
                              param1=140, param2=30, minRadius=20, maxRadius=40)
    return circles


def detect_lines(img):
    """Wykrywa linie na podstawie obrazu img."""
    # Konwertacja obrazu do skali szarości
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detekcja krawędzi
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    #Parametry minLineLength i maxLineGap kontrolują minimalną długość linii i maksymalny odstęp między segmentami linii
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    return lines


def calculate_bounding_box(lines):
    """Oblicza prostokątny obszar ograniczający na podstawie wykrytych linii."""
    x0, y0, x1, y1 = lines[0][0][0], lines[0][0][1], lines[0][0][0], lines[0][0][1]
    for line in lines:
        x0 = min(x0, line[0][0])
        x1 = max(x1, line[0][2])
        y0 = min(y0, line[0][1])
        y1 = max(y1, line[0][3])
    return x0, y0, x1, y1


def draw_and_count_circles(img, circles, bounding_box):
    """Rysuje okręgi na obrazie img, oznaczając je kolorem na podstawie ich wielkości oraz pozycji."""
    x0, y0, x1, y1 = bounding_box
    sum_inside, count_inside = 0, 0
    sum_outside, count_outside = 0, 0

    for i in circles[0, :]:
        x, y, radius = int(i[0]), int(i[1]), int(i[2])

        # Sprawdzamy rozmiar i ustawiamy nominal monety
        if radius >= max(circles[0, :, 2]) - 3:
            radius_colour = (255, 0, 0)
            zm = 5
        else:
            radius_colour = (0, 255, 255)
            zm = 0.05

        # Zmiana koloru w zaleznosci od tego czy moneta na tace
        center_colour = (0, 255, 0) if is_inside(x, y, x0, y0, x1, y1) else (0, 0, 255)
        cv.circle(img, (x, y), radius, radius_colour, 2)
        cv.circle(img, (x, y), 2, center_colour, 3)

        # Rachujemy ilosc monet na i poza taca
        if is_inside(i[0], i[1], x0, y0, x1, y1):
            sum_inside += zm
            count_inside += 1
        else:
            sum_outside += zm
            count_outside += 1

    return sum_inside, count_inside, sum_outside, count_outside


def display_text(img, sum_inside, count_inside, sum_outside, count_outside):
    """Wyświetla tekst na obrazie img zawierający informacje o liczbie monet oraz ich sumie."""
    text_y_offset = 80
    total_sum = sum_inside + sum_outside
    cv.putText(img, f'Monet na tace: {count_inside}, Suma: {round(sum_inside, 2)}', (10, img.shape[0] - text_y_offset),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(img, f'Monet poza taca: {count_outside}, Suma: {round(sum_outside, 2)}',
               (10, img.shape[0] - text_y_offset + 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(img, f'Cala suma monet: {round(total_sum, 2)}', (10, img.shape[0] - text_y_offset + 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def calculate_tray_area(bounding_box):
    """Oblicza powierzchnię tacy na podstawie współrzędnych jej ograniczeń."""
    x0, y0, x1, y1 = bounding_box
    tray_width = x1 - x0
    tray_height = y1 - y0
    tray_area = tray_width * tray_height
    return tray_area


def detect_coins_with_text_below(img, window_name):
    """Wykrywa monety na obrazie i wyświetla informacje o nich na dole okna."""
    circles = detect_circles(img)
    lines = detect_lines(img)

    if circles is None or lines is None:
        return

    bounding_box = calculate_bounding_box(lines)
    tray_area = calculate_tray_area(bounding_box)

    sum_inside, count_inside, sum_outside, count_outside = draw_and_count_circles(img, circles, bounding_box)

    # Rysowanie krawędzi tacy
    x0, y0, x1, y1 = bounding_box
    cv.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)

    img_with_text = display_text(img, sum_inside, count_inside, sum_outside, count_outside)

    # Wyliczenie rozmiaru tekstu dla dopasowania na zdjęciu
    text_size, _ = cv.getTextSize("Text", cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    # Resize dla dopasowania zdjęcia oraz tekstu
    cv.resizeWindow(window_name, img_with_text.shape[1], img_with_text.shape[0] + 140)

    cv.imshow(window_name, img_with_text)

    # Wyświetlanie powierzchni tacy
    tray_area_text = f'Powierzchnia tacy: {tray_area} px^2'
    print(tray_area_text)



def main():
    """Główna funkcja programu, wczytuje obrazy z folderu i uruchamia proces detekcji."""
    folder_dir = 'pliki'
    window_name = 'Image'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1200, 800)

    image_files = [f for f in os.listdir(folder_dir) if f.endswith('.jpg')]
    current_image_index = 0

    while True:
        image_file = image_files[current_image_index]
        image_path = os.path.join(folder_dir, image_file)
        print('Processing image:', image_path)
        img = cv.imread(image_path, cv.IMREAD_COLOR)

        detect_coins_with_text_below(img, window_name)
        key = cv.waitKey(0)

        if key == 27:
            break
        elif 49 <= key <= 56:
            current_image_index = key - 49
            current_image_index %= len(image_files)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
