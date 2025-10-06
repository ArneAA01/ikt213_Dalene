import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.png', cv2.IMREAD_GRAYSCALE)

def preprocess(img):
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

def orb_bf_match(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good_matches, match_img

def sift_flann_match(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good_matches, match_img

img1_pre = preprocess(img1)
img2_pre = preprocess(img2)

orb_matches, orb_img = orb_bf_match(img1_pre, img2_pre)
sift_matches, sift_img = sift_flann_match(img1_pre, img2_pre)

print(f"[ORB + BFMatcher] Good matches: {len(orb_matches)}")
print(f"[SIFT + FLANN]   Good matches: {len(sift_matches)}")

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title("ORB + BFMatcher")
plt.imshow(orb_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("SIFT + FLANN")
plt.imshow(sift_img)
plt.axis('off')

plt.tight_layout()
plt.show()
