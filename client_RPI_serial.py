"""
client_RPI_serial.py

Simple serial/keyboard-driven Raspberry Pi client for testing endpoints.

Behavior:
- Opens the camera once and uploads images in a loop to `/caption/en` by default.
- Press keys 1, 2, 3 to change the endpoint used for the *next* upload:
    1 -> /ocr/en
    2 -> /traffic
    3 -> /search
- Press q to quit.

Usage: run from a terminal on the Pi (or your desktop for testing). If RPi.GPIO
is available it will not be used; this script relies on keyboard input only.

"""

import os
import sys
import time
import threading
import select
try:
    import msvcrt
except Exception:
    msvcrt = None
import tempfile
import subprocess
import shutil

import requests
import cv2 as cv

# TTS helpers (reuse approach from other client): try gTTS then espeak
try:
    from gtts import gTTS
    from playsound import playsound
except Exception:
    gTTS = None
    playsound = None


def speak(text: str):
    if not text:
        return
    # Prefer espeak (synchronous). This blocks until speech completes.
    try:
        subprocess.run(['espeak', text], check=True)
        return
    except Exception:
        pass

    # Fallback to gTTS+playsound (also synchronous)
    try:
        if gTTS is not None and playsound is not None:
            tts = gTTS(text=text, lang='en')
            fd, path = tempfile.mkstemp(suffix='.mp3')
            os.close(fd)
            try:
                tts.save(path)
                # Try to play with an external player (blocking) if available
                player = None
                for cmd in ('mpg123', 'mpv', 'omxplayer', 'ffplay'):
                    if shutil.which(cmd):
                        player = cmd
                        break
                if player:
                    if player == 'omxplayer':
                        subprocess.run([player, path, '-o', 'local'], check=False)
                    elif player == 'ffplay':
                        subprocess.run([player, '-nodisp', '-autoexit', '-loglevel', 'quiet', path], check=False)
                    else:
                        subprocess.run([player, path], check=False)
                else:
                    # playsound typically blocks; use it as last resort
                    playsound(path)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
            return
    except Exception:
        pass

    # Last resort: print the text
    print(text)


class SerialClient:
    def __init__(self, server_base: str, camera_index=0):
        self.server_base = server_base.rstrip('/')
        self.photo_folder = os.path.join(tempfile.gettempdir(), 'rpi_serial_photos')
        os.makedirs(self.photo_folder, exist_ok=True)
        self.photo_path = os.path.join(self.photo_folder, 'latest.jpg')

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

        # Single-threaded design: no capture_lock required
        self.current_endpoint = '/caption/en'
        self._stop = False

        if not self.capture.isOpened():
            raise RuntimeError('Could not open camera')

    def close(self):
        try:
            self.capture.release()
        except Exception:
            pass

    def take_photo(self):
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return False
        cv.imwrite(self.photo_path, frame)
        return True

    def upload_and_speak(self):
        url = self.server_base + self.current_endpoint
        ok = self.take_photo()
        if not ok:
            print('Failed to capture')
            return
        try:
            # Keep the file open while requests reads it by posting inside the with-block
            with open(self.photo_path, 'rb') as f:
                resp = requests.post(url, files={'file': f}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data.get('caption') or data.get('text') or data.get('result') or data.get('message')
            print(f'Response from {self.current_endpoint}: {text}')
            if text:
                speak(str(text))
        except Exception as e:
            print(f'Upload error: {e}')
            try:
                self.close()
            except Exception:
                pass
            # Exit the whole process with non-zero so a service manager can restart it
            os._exit(1)

    def _handle_key(self, ch: str):
        if not ch:
            return
        ch = ch.strip().lower()
        if ch == 'q':
            print('Quitting...')
            self._stop = True
        elif ch == '1':
            self.current_endpoint = '/ocr/en'
            print('Next upload -> /ocr/en')
        elif ch == '2':
            self.current_endpoint = '/traffic'
            print('Next upload -> /traffic')
        elif ch == '3':
            self.current_endpoint = '/search'
            print('Next upload -> /search')
        else:
            print('Unknown key')

    def run_loop(self, interval=1.0):
        print('Starting single-threaded upload loop. Press 1=/ocr, 2=/traffic, 3=/search for next upload. q to quit.')
        use_select = hasattr(select, 'select') and msvcrt is None
        try:
            while not self._stop:
                # perform upload -> speak synchronously
                self.upload_and_speak()

                # wait for interval seconds but react to keypresses
                end_time = time.time() + interval
                while time.time() < end_time and not self._stop:
                    # On Windows, use msvcrt for non-blocking key reads
                    if msvcrt is not None:
                        if msvcrt.kbhit():
                            ch = msvcrt.getwch()
                            self._handle_key(ch)
                        else:
                            time.sleep(0.1)
                    elif use_select:
                        timeout = max(0, end_time - time.time())
                        r, _, _ = select.select([sys.stdin], [], [], min(0.1, timeout))
                        if r:
                            ch = sys.stdin.read(1)
                            self._handle_key(ch)
                    else:
                        # fallback: blocking read (will likely pause loop)
                        try:
                            ch = sys.stdin.read(1)
                            self._handle_key(ch)
                        except Exception:
                            time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f'Fatal loop error: {e}')
            try:
                self.close()
            except Exception:
                pass
            os._exit(1)

    def stop(self):
        self._stop = True


def read_key_nonblocking():
    # simple cross-platform approach: on Windows this will block; on Unix you can run in a terminal
    # For now use blocking read from stdin (user will press keys to change endpoint)
    try:
        ch = sys.stdin.read(1)
        return ch
    except Exception:
        return None


def main():
    server_base = 'http://specsserver.local'  # adjust if needed
    client = SerialClient(server_base)
    try:
        client.run_loop(interval=1.0)
    finally:
        client.close()


if __name__ == '__main__':
    main()
