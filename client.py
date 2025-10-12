import subprocess
import requests
import os
import cv2 as cv
import time
import sys

# Use msvcrt for non-blocking key detection on Windows
try:
    import msvcrt
except Exception:
    msvcrt = None


def take_photo(photo_path, capture):
    """Read a single frame from an already-open cv2.VideoCapture instance and save it."""
    ret, frame = capture.read()
    if ret and frame is not None:
        # ensure folder exists
        os.makedirs(os.path.dirname(photo_path), exist_ok=True)
        cv.imwrite(photo_path, frame)
        return True
    return False


def upload_photo(photo_path, server_url):
    print(f"Uploading photo to {server_url}...")
    with open(photo_path, 'rb') as photo_file:
        files = {'file': photo_file}
        response = requests.post(server_url, files=files, timeout=15)
    response.raise_for_status()
    return response.json()


def speak_caption(caption):
    print("Speaking caption...")
    try:
        # Keep the existing espeak call as a convenience; on Windows you may want to
        # replace this with gTTS or another TTS solution.
        subprocess.run(['espeak', f"{caption}"])
    except FileNotFoundError:
        print("espeak not found. Please install espeak or modify the speak_caption function to use another TTS engine.")


def user_pressed_d() -> bool:
    """Return True if the user pressed 'd' (non-blocking). Works on Windows via msvcrt."""
    if msvcrt is None:
        return False
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            key = ch.decode('utf-8')
        except Exception:
            return False
        return key.lower() == 'd'
    return False


def main():
    folder_path = "temp/images"
    photo_path = os.path.join(folder_path, "latest.jpg")
    server_base = "http://specsserver.local"  # base URL
    caption_endpoint = "/caption/en"
    ocr_endpoint = "/traffic"

    print("Client started. Press 'd' to send the next photo to /ocr/en. Press Ctrl-C to quit.")

    # Open camera once and reuse it to avoid repeated locking/releasing
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Error: could not open camera (index 0)")
        return

    try:
        while True:
            try:
                # Default endpoint
                endpoint = caption_endpoint

                # If the user pressed 'd' since last loop, use /ocr/en for this iteration
                if user_pressed_d():
                    endpoint = ocr_endpoint
                    print("'d' pressed — sending to /ocr/en for this iteration")

                ok = take_photo(photo_path, capture)
                if not ok:
                    print("Warning: failed to capture frame")
                    time.sleep(0.5)
                    continue

                url = server_base + endpoint
                response = upload_photo(photo_path, url)
                # The server returns either 'caption' or 'text' depending on endpoint
                caption = response.get("caption") or response.get("text") or response.get("message")
                print(f"Response from {endpoint}: {caption}")
                if caption:
                    speak_caption(caption)

                # Small delay to avoid hammering the camera/server
                time.sleep(1)
            except KeyboardInterrupt:
                print("Exiting client.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)
    finally:
        try:
            capture.release()
        except Exception:
            pass

if __name__ == "__main__":
    main()
