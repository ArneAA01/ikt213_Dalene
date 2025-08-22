import cv2

def print_image_information(image):
    height, width, channels = image.shape
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", image.size)
    print("dtype:", image.dtype)

def main():
    img = cv2.imread("lena-1.png")
    print_image_information(img)

if __name__ == "__main__":
    main()
