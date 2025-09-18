
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def _read_first_found(filenames):
    for name in filenames:
        p = os.path.join(BASE_DIR, name)
        img = cv2.imread(p)
        if img is not None:
            return img, p
        p2 = os.path.join(OUT_DIR, name)
        img = cv2.imread(p2)
        if img is not None:
            return img, p2
    return None, None

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None.")
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = _ensure_gray(blurred)
    sobel64 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel64)
    cv2.imwrite(os.path.join(OUT_DIR, "lambo_sobel.png"), sobel)
    return sobel

def canny_edge_detection(image: np.ndarray, threshold_1: int, threshold_2: int) -> np.ndarray:

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = _ensure_gray(blurred)
    edges = cv2.Canny(gray, threshold_1, threshold_2)
    cv2.imwrite(os.path.join(OUT_DIR, "lambo_canny.png"), edges)
    return edges


def template_match(image: np.ndarray, template: np.ndarray) -> np.ndarray:

    gray_img = _ensure_gray(image)
    gray_tpl = _ensure_gray(template)

    res = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    draw = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = gray_tpl.shape

    for pt in zip(*loc[::-1]):
        top_left = (int(pt[0]), int(pt[1]))
        bottom_right = (int(pt[0] + w), int(pt[1] + h))
        cv2.rectangle(draw, top_left, bottom_right, (0, 0, 255), 2)  # red

    cv2.imwrite(os.path.join(OUT_DIR, "shapes_matched.png"), draw)
    return draw

def resize(image: np.ndarray, scale_factor: int, up_or_down: str) -> np.ndarray:

    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")

    out = image.copy()
    if up_or_down.lower() == "up":
        for _ in range(scale_factor):
            out = cv2.pyrUp(out)
        fn = f"lambo_resized_up_{scale_factor}.png"
    elif up_or_down.lower() == "down":
        for _ in range(scale_factor):
            out = cv2.pyrDown(out)
        fn = f"lambo_resized_down_{scale_factor}.png"
    else:
        raise ValueError("up_or_down must be 'up' or 'down'.")

    cv2.imwrite(os.path.join(OUT_DIR, fn), out)
    return out

# ---------- Runner ----------
if __name__ == "__main__":
    lambo_img, lambo_path = _read_first_found(["lambo.png"])
    if lambo_img is None:
        raise FileNotFoundError("Could not find lambo.png (looked in project root and /outputs).")

    shapes_img, _ = _read_first_found(["shapes.png"])
    template_img, _ = _read_first_found(["shapes_template.jpg"])

    sobel_edge_detection(lambo_img)

    canny_edge_detection(lambo_img, 50, 50)

    if shapes_img is not None and template_img is not None:
        template_match(shapes_img, template_img)

    resize(lambo_img, 2, "up")
    resize(lambo_img, 2, "down")

    print(f"Done. Saved outputs in: {OUT_DIR}")
