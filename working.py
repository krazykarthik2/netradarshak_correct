import cv2

def test_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("✅ Camera opened successfully. Capturing frame...")

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image — try a different format or resolution.")
        return

    filename = "test_image.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Image saved as {filename}")

    cap.release()

if __name__ == "__main__":
    test_camera()
this works write like this