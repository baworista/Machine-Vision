import cv2 as cv

image = None  # Zmienna globalna przechowująca obraz z orb()
image2 = None  # Zmienna globalna przechowująca obraz z orb()


def sift3(image, image2, k=3):
    gimg1 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    gimg1 = cv.medianBlur(gimg1, ksize=k)
    gimg2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gimg2 = cv.medianBlur(gimg2, ksize=k)
    siftobject = cv.SIFT_create()
    keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2, None)
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv.drawMatches(
        image2, keypoints_1, image, keypoints_2, matches, image, flags=2)
    cv.imshow('obrazek', matched_img)


def orb(image, images, k=3):
    gimg1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gimg1 = cv.medianBlur(gimg1, ksize=k)
    orb = cv.ORB_create()

    best_matches = []
    best_matched_img = None
    best_matched_index = -1

    for idx, img in enumerate(images):
        gimg2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gimg2 = cv.medianBlur(gimg2, ksize=k)

        keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)

        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > len(best_matches):
            best_matches = matches
            best_matched_img = img
            best_matched_index = idx

    matched_img = cv.drawMatches(
        best_matched_img, keypoints_1, image, keypoints_2, best_matches, image, flags=2)

    cv.imshow('Dopasowanie', matched_img)


def main(method='sift', k=3):
    global image, image2

    # Initialize FlannBasedMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Ścieżki do zdjęć elementu i filmu
    image_paths = [
        "data/img1.jpeg",
        "data/img2.jpeg",
    ]
    video_path = "video.mp4"

    # Wyciągnięcie cech przy pomocy SIFT
    images = [cv.imread(path) for path in image_paths]

    image = images[0]  # Wybierz pierwszy obraz z unlabeled_faces jako image
    cap = cv.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image2 = frame  # Ustaw bieżącą klatkę jako image2

        if method == 'sift':
            sift3(frame, image, k=k)  # Wywołaj funkcję sift3() dla bieżącej klatki i obrazu image z parametrem k
        elif method == 'orb':
            orb(frame, images, k=k)  # Wywołaj funkcję orb() dla bieżącej klatki i obrazu image2 z parametrem k

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Wywołanie głównej funkcji, można zmieniać metodę ('sift' lub 'orb') oraz parametr k
    main(method='orb', k=5)