import cv2
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

def save(name: str, img: np.ndarray):
    p = OUT / name
    cv2.imwrite(str(p), img)
    print(f"Saved -> {p}")


def padding(image: np.ndarray, border_width: int) -> np.ndarray:
    return cv2.copyMakeBorder(
        image, border_width, border_width, border_width, border_width,
        borderType=cv2.BORDER_REFLECT
    )

def crop(image: np.ndarray, x_0: int, x_1: int, y_0: int, y_1: int) -> np.ndarray:
    return image[y_0:y_1, x_0:x_1]
def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def copy(image: np.ndarray, emptyPictureArray: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    emptyPictureArray[:h, :w, :] = image
    return emptyPictureArray

def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image: np.ndarray, emptyPictureArray: np.ndarray, hue: int) -> np.ndarray:
    shifted = image.astype(np.int16) + int(hue)
    shifted = np.clip(shifted, 0, 255)
    emptyPictureArray[:image.shape[0], :image.shape[1], :] = shifted.astype(np.uint8)
    return emptyPictureArray

def smoothing(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, ksize=(15, 15), sigmaX=0)

def rotation(image: np.ndarray, rotation_angle: int) -> np.ndarray:
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle must be 90 or 180")

if __name__ == "__main__":
    img_path = HERE / "lena.png"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}. Make sure lena.png is in {HERE}")

    h, w = img.shape[:2]
    empty = np.zeros((h, w, 3), dtype=np.uint8)

    padded = padding(img, border_width=100)
    save("01_padding_reflect_100.png", padded)

    cropped = crop(img, x_0=80, x_1=w-130, y_0=80, y_1=h-130)
    save("02_cropped.png", cropped)

    resized = resize(img, width=200, height=200)
    save("03_resized_200x200.png", resized)

    copied = copy(img, emptyPictureArray=empty.copy())
    save("04_manual_copy.png", copied)

    gray = grayscale(img)
    save("05_grayscale.png", gray)

    hsv_img = hsv(img)
    save("06_hsv.png", hsv_img)

    shifted = hue_shifted(img, emptyPictureArray=empty.copy(), hue=50)
    save("07_color_shift_plus50.png", shifted)


    smooth = smoothing(img)
    save("08_smoothing_gaussian_15.png", smooth)

    rot90 = rotation(img, 90)
    save("09_rotated_90.png", rot90)

    rot180 = rotation(img, 180)
    save("10_rotated_180.png", rot180)

    print("All outputs saved to:", OUT)
