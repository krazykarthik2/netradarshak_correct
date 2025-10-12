import subprocess
import requests
import os
import cv2 as cv
import time
import sys
import threading
import io
import tempfile

# Try to import RPi.GPIO; if not available (e.g., running on desktop), provide a dummy
try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None

# TTS helpers - try gTTS+playsound then fall back to espeak
try:
    from gtts import gTTS
    from playsound import playsound
except Exception:
    gTTS = None
    playsound = None


def speak(text: str):
    """Speak the provided text. Try gTTS+playsound, fall back to espeak CLI, else print."""
    if not text:
        return
    try:
        if gTTS is not None and playsound is not None:
            tts = gTTS(text=text, lang='en')
            fd, path = tempfile.mkstemp(suffix='.mp3')
            os.close(fd)
            try:
                tts.save(path)
                playsound(path)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
            return
    except Exception:
        pass

    # Fallback to espeak
    try:
        subprocess.run(['espeak', text], check=False)
        return
    except Exception:
        pass

    # Last resort: print
    print(text)


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
        response = requests.post(server_url, files=files, timeout=30)
    response.raise_for_status()
    return response.json()


class RPICaptureClient:
    def __init__(self, server_base: str, camera_index=0):
        self.server_base = server_base.rstrip('/')
        self.photo_folder = os.path.join(tempfile.gettempdir(), 'rpi_client_photos')
        os.makedirs(self.photo_folder, exist_ok=True)
        self.photo_path = os.path.join(self.photo_folder, 'latest.jpg')

        # capture and mutex to avoid concurrent captures
        try:
            self.capture = cv.VideoCapture(camera_index, cv.CAP_V4L2)
        except Exception:
            self.capture = cv.VideoCapture(camera_index)

        try:
            self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass

        if not self.capture.isOpened():
            raise RuntimeError('Could not open camera')

        self.capture_lock = threading.Lock()

    def close(self):
        try:
            self.capture.release()
        except Exception:
            pass

    def capture_and_send(self, endpoint: str):
        url = self.server_base + endpoint
        with self.capture_lock:
            ok = take_photo(self.photo_path, self.capture)
        if not ok:
            print('Failed to capture photo')
            return

        try:
            resp = upload_photo(self.photo_path, url)
            # endpoint responses may vary: try common keys
            text = resp.get('caption') or resp.get('text') or resp.get('result') or resp.get('message')
            print(f"Response from {endpoint}: {text}")
            if text:
                speak(str(text))
        except Exception as e:
            print(f"Error sending photo to {url}: {e}")


def setup_gpio(client: RPICaptureClient):
    if GPIO is None:
        print('RPi.GPIO not available; running without GPIO button handlers')
        return None

    GPIO.setmode(GPIO.BCM)

    # mapping: pin -> endpoint
    mapping = {
        5: '/ocr/en',     # GPIO5 -> OCR
        6: '/traffic',    # GPIO6 -> traffic (server expected to behave like /caption)
        13: '/search',    # GPIO13 -> search (server expected to return result)
    }

    # configure pins as inputs with pull-up (buttons to GND)
    for pin in mapping.keys():
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        except Exception:
            GPIO.setup(pin, GPIO.IN)

    # callback factory
    def make_callback(endpoint):
        def cb(channel):
            print(f'Button on GPIO {channel} pressed -> {endpoint}')
            # dispatch to background thread so callback returns quickly
            t = threading.Thread(target=client.capture_and_send, args=(endpoint,), daemon=True)
            t.start()
        return cb

    # add event detection with debounce
    for pin, endpoint in mapping.items():
        try:
            GPIO.add_event_detect(pin, GPIO.FALLING, callback=make_callback(endpoint), bouncetime=300)
        except Exception as e:
            # fallback to polling if event detect fails
            print(f'GPIO add_event_detect failed for pin {pin}: {e}')

    return mapping


def main():
    server_base = "http://specsserver.local"  # base URL (adjust if needed)

    try:
        client = RPICaptureClient(server_base)
    except Exception as e:
        print(f'Camera initialization failed: {e}')
        return

    mapping = setup_gpio(client)

    print('RPi client running. Buttons: GPIO5=/ocr, GPIO6=/traffic, GPIO13=/search. Ctrl-C to quit.')

    try:
        # If GPIO is not available, optionally poll keyboard for testing
        if GPIO is None:
            # simple loop to emulate button presses by typing keys
            while True:
                print('Type o for OCR, t for traffic, s for search, q to quit: ', end='', flush=True)
                ch = sys.stdin.read(1).strip().lower()
                if ch == 'q':
                    break
                elif ch == 'o':
                    client.capture_and_send('/ocr/en')
                elif ch == 't':
                    client.capture_and_send('/traffic')
                elif ch == 's':
                    client.capture_and_send('/search')
                else:
                    client.capture_and_send('/caption/en')
                time.sleep(0.1)
        else:
            # keep the program alive while GPIO callbacks handle presses
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print('Exiting...')
    finally:
        try:
            client.close()
        except Exception:
            pass
        if GPIO is not None:
            try:
                GPIO.cleanup()
            except Exception:
                pass


if __name__ == '__main__':
    main()
