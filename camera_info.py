import cv2
import os

def main():
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Get webcam properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # Make sure solutions folder exists
    os.makedirs("solutions", exist_ok=True)

    # Save results into solutions/camera_outputs.txt
    out_path = os.path.join("solutions", "camera_outputs.txt")
    with open(out_path, "w") as f:
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print(f"Camera info saved in {out_path}")

if __name__ == "__main__":
    main()
