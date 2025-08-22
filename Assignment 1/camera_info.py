import cv2
import os

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    os.makedirs("solutions", exist_ok=True)

    out_path = os.path.join("solutions", "camera_outputs.txt")
    with open(out_path, "w") as f:
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print(f"Camera info saved in {out_path}")

if __name__ == "__main__":
    main()
