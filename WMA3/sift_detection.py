import cv2 as cv
import numpy as np
import os

def load_images(folder_path):
    """
    Wczytuje obrazy z określonej ścieżki folderu i zwraca listę obrazów w skali szarości oraz odpowiadających im kluczowych punktów i deskryptorów.
    """
    sift = cv.SIFT_create()
    keypoints_and_descriptors = []
    images = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_and_descriptors.append((kp, des))
        images.append(gray)

    return keypoints_and_descriptors, images

def find_best_match(kp_frame, des_frame, keypoints_and_descriptors):
    """
    Znajduje najlepsze dopasowanie dla obecnego klatki wideo.
    """
    bf = cv.BFMatcher()
    matches = [bf.knnMatch(des, des_frame, k=2) for _, des in keypoints_and_descriptors]

    good_matches = [[m[0] for m in match if len(m) == 2 and m[0].distance < 0.45 * m[1].distance] for match in matches]

    best_matches_count = 0
    best_matched_image_index = 0

    for idx, good_match in enumerate(good_matches):
        if len(good_match) > best_matches_count:
            best_matches_count = len(good_match)
            best_matched_image_index = idx

    return best_matched_image_index, good_matches[best_matched_image_index]

def main():
    cap = cv.VideoCapture('video.mp4')
    MIN_MATCH_COUNT = 5
    folder_path = 'data/'

    keypoints_and_descriptors, images = load_images(folder_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp_frame, des_frame = cv.SIFT_create().detectAndCompute(gray_frame, None)

        best_matched_image_index, very_good = find_best_match(kp_frame, des_frame, keypoints_and_descriptors)
        best_matched_image = images[best_matched_image_index]

        if len(very_good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints_and_descriptors[best_matched_image_index][0][m.queryIdx].pt for m in very_good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in very_good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = best_matched_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            if M is not None:
                dst = cv.perspectiveTransform(pts, M)
                frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                print("Matches found - {}/{}".format(len(very_good), MIN_MATCH_COUNT))
            else:
                print("Error: Unable to find valid homography matrix!")
                matchesMask = None

            print("Matches found - {}/{}".format(len(very_good), MIN_MATCH_COUNT))
        else:
            print("Not enough matches found - {}/{}".format(len(very_good), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(255, 50, 200),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        img = cv.drawMatches(best_matched_image, keypoints_and_descriptors[best_matched_image_index][0], frame, kp_frame, very_good, None, **draw_params)

        cv.imshow('frame', img)
        if cv.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
