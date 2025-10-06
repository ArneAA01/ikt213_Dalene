import argparse
import cv2
import numpy as np
from PIL import Image
import os

def harris_corner_detection(reference_image_path):
    img_color = cv2.imread(reference_image_path)
    if img_color is None:
        raise FileNotFoundError(f"Could not read {reference_image_path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img_marked = img_color.copy()
    img_marked[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite("harris.png", img_marked)

def align_with_sift(image_to_align_path, reference_image_path, max_features, good_match_percent):
    im1 = cv2.imread(image_to_align_path, cv2.IMREAD_COLOR)
    im2 = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        raise FileNotFoundError("Input images not found")
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps1, des1 = sift.detectAndCompute(im1_gray, None)
    kps2, des2 = sift.detectAndCompute(im2_gray, None)
    if des1 is None or des2 is None or len(kps1) == 0 or len(kps2) == 0:
        raise ValueError("No features detected")
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches_knn:
        if m.distance < good_match_percent * n.distance:
            good.append(m)
    if len(good) < 4:
        raise ValueError("Not enough good matches to compute homography")
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography computation failed")
    h, w = im2.shape[:2]
    im1_aligned = cv2.warpPerspective(im1, H, (w, h))
    cv2.imwrite("aligned.png", im1_aligned)
    matches_mask = mask.ravel().tolist()
    good_inliers = [good[i] for i in range(len(good)) if matches_mask[i] == 1]
    if len(good_inliers) == 0:
        good_inliers = good
    vis = cv2.drawMatches(im1, kps1, im2, kps2, good_inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matches.png", vis)

def make_pdf():
    pages = []
    for name in ["harris.png", "aligned.png", "matches.png"]:
        if not os.path.exists(name):
            continue
        im = Image.open(name).convert("RGB")
        pages.append(im)
    if len(pages) == 0:
        raise FileNotFoundError("No images to add to PDF")
    first, rest = pages[0], pages[1:]
    first.save("output.pdf", save_all=True, append_images=rest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harris", type=str, help="Path to reference image")
    parser.add_argument("--align", nargs=2, metavar=("image_to_align", "reference_image"), help="Align image_to_align to reference_image")
    parser.add_argument("--max_features", type=int, default=10)
    parser.add_argument("--good_match_percent", type=float, default=0.7)
    parser.add_argument("--make-pdf", action="store_true")
    args = parser.parse_args()
    if args.harris:
        harris_corner_detection(args.harris)
    if args.align:
        align_with_sift(args.align[0], args.align[1], args.max_features, args.good_match_percent)
    if args.make_pdf:
        make_pdf()

if __name__ == "__main__":
    main()
