"""
demo.py

Capture frames from webcam and process them with VLM models.

Controls:
  c     - capture and process in caption mode
  q     - quit

Frames are saved to temp/images and opened in Chrome.
"""
import sys
import time
import os
import subprocess

try:
    import cv2
except Exception:
    print("Error: OpenCV (cv2) is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    torch = None
    BlipProcessor = None
    BlipForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except Exception:
    AutoModelForVision2Seq = None
    AutoProcessor = None

try:
    from googletrans import Translator
except Exception:
    Translator = None

from PIL import Image
import numpy as np

# Lazy model holders
_models = {
    "blip_processor": None,
    "blip_model": None,
    "traffic_blip_processor": None,
    "traffic_blip_model": None,
    "ocr_reader": None,
    "translator": None,
}



def init_blip():
    """Initialize BLIP model for general captioning."""
    if _models["blip_model"] is None or _models["blip_processor"] is None:
        if BlipProcessor is None or BlipForConditionalGeneration is None or torch is None:
            print("BLIP not available")
            return None, None
        try:
            print("Loading BLIP model...")
            _models["blip_processor"] = BlipProcessor.from_pretrained("./vlms/final_model")
            _models["blip_model"] = BlipForConditionalGeneration.from_pretrained("./vlms/final_model")
            _models["blip_model"].eval()
            print("BLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load BLIP: {e}")
            _models["blip_processor"] = None
            _models["blip_model"] = None
    return _models["blip_processor"], _models["blip_model"]


def init_traffic_blip():
    """Initialize BLIP model for traffic analysis."""
    if _models["traffic_blip_model"] is None or _models["traffic_blip_processor"] is None:
        if AutoProcessor is None or AutoModelForVision2Seq is None or torch is None:
            print("Traffic BLIP not available")
            return None, None
        try:
            print("Loading Traffic BLIP model...")
            _models["traffic_blip_processor"] = AutoProcessor.from_pretrained("./vlms/final_model")
            _models["traffic_blip_model"] = AutoModelForVision2Seq.from_pretrained("./vlms/final_model")
            _models["traffic_blip_model"].eval()
            print("Traffic BLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load Traffic BLIP: {e}")
            _models["traffic_blip_processor"] = None
            _models["traffic_blip_model"] = None
    return _models["traffic_blip_processor"], _models["traffic_blip_model"]


def init_ocr():
    """Initialize EasyOCR reader."""
    if _models["ocr_reader"] is None:
        if easyocr is None:
            print("EasyOCR not available")
            return None
        try:
            print("Loading OCR model (this may take a while)...")
            _models["ocr_reader"] = easyocr.Reader(["en"], gpu=False)
            print("OCR model loaded successfully")
        except Exception as e:
            print(f"Failed to load OCR: {e}")
            _models["ocr_reader"] = None
    return _models["ocr_reader"]


def init_translator():
    """Initialize Google Translator."""
    if _models["translator"] is None:
        if Translator is None:
            print("Translator not available")
            return None
        try:
            _models["translator"] = Translator()
            print("Translator initialized")
        except Exception as e:
            print(f"Failed to init translator: {e}")
            _models["translator"] = None
    return _models["translator"]


def generate_caption(pil_image):
    """Generate caption for image using BLIP model."""
    processor, model = init_blip()
    if processor is None or model is None:
        return "Caption generation unavailable"
    
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def generate_traffic_caption(pil_image):
    """Generate traffic analysis caption."""
    processor, model = init_traffic_blip()
    if processor is None or model is None:
        return "Traffic analysis unavailable"
    
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def generate_ocr_text(pil_image):
    """Extract text from image using OCR."""
    reader = init_ocr()
    if reader is None:
        return "OCR unavailable"
    
    try:
        # Convert PIL to numpy array for easyocr
        img_array = np.array(pil_image)
        if len(img_array.shape) == 2:  # grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        result = reader.readtext(img_array)
        text = " ".join([t[1] for t in result])
        return text if text else "(No text found)"
    except Exception as e:
        return f"Error: {str(e)}"


def translate_text(text, target_lang="en"):
    """Translate text to target language."""
    if target_lang == "en":
        return text
    
    translator = init_translator()
    if translator is None:
        return text
    
    try:
        result = translator.translate(text, dest=target_lang)
        return result.text if result else text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def bgr_to_pil(bgr_frame):
    """Convert OpenCV BGR frame to PIL Image."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_frame_and_open_chrome(bgr_frame, output_dir='temp/images'):
    """Save frame to temp/images folder and open in Chrome."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(output_dir, f'frame_{timestamp}.jpg')
    
    # Save frame
    success = cv2.imwrite(filename, bgr_frame)
    
    if success:
        print(f"Frame saved to: {filename}")
        # Open in Chrome
        try:
            abs_path = os.path.abspath(filename)
            subprocess.Popen(['start', 'chrome',abs_path], shell=True)
            print(f"Opened in Chrome: {abs_path}")
        except Exception as e:
            print(f"Could not open in Chrome: {e}")
        return filename
    else:
        print("Failed to save frame")
        return None


def main():
    """Main webcam capture and processing loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Webcam-based VLM demo")
    parser.add_argument('--lang', default='en', help='Target language for translation (en, hi, te, etc)')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mode = "caption"  # caption, ocr, or traffic
    
    print("\n=== Webcam VLM Demo ===")
    print("Controls:")
    print("  c     - capture and process in caption mode")
    print("  q     - quit")
    print(f"Language: {args.lang}\n")
    
    try:
        while True:
            
            # Handle keyboard input
            key = -1
            try:
                user_input = input().strip().lower()
                if user_input == 'q':
                    key = ord('q')
                elif user_input == 'c':
                    key = ord('c')
                elif user_input == 'o':
                    key = ord('o')
                elif user_input == 't':
                    key = ord('t')
            except Exception:
                pass
            
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                mode = "caption"
                print(f"\n[{time.strftime('%H:%M:%S')}] Capturing in {mode} mode...")
                
                # Save frame and open in Chrome
                save_frame_and_open_chrome(frame)
                
                # Convert frame to PIL
                pil_image = bgr_to_pil(frame)
                
                try:
                    result = generate_caption(pil_image)
                    translated = translate_text(result, args.lang)
                    print(f"Caption (EN): {result}")
                    subprocess.Popen(['espeak', result])  # Optional: speak the caption
                    if args.lang != "en":
                        print(f"Caption ({args.lang.upper()}): {translated}")
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                print()
            
            elif key == ord('o'):
                mode = "ocr"
                print(f"\n[{time.strftime('%H:%M:%S')}] Capturing in {mode} mode...")
                
                # Save frame and open in Chrome
                save_frame_and_open_chrome(frame)
                
                # Convert frame to PIL
                pil_image = bgr_to_pil(frame)
                
                try:
                    result = generate_ocr_text(pil_image)
                    translated = translate_text(result, args.lang)
                    print(f"OCR (EN): {result}")
                    if args.lang != "en":
                        print(f"OCR ({args.lang.upper()}): {translated}")
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                print()
            
            elif key == ord('t'):
                mode = "traffic"
                print(f"\n[{time.strftime('%H:%M:%S')}] Capturing in {mode} mode...")
                
                # Save frame and open in Chrome
                save_frame_and_open_chrome(frame)
                
                # Convert frame to PIL
                pil_image = bgr_to_pil(frame)
                
                try:
                    result = generate_traffic_caption(pil_image)
                    print(f"Traffic Analysis: {result}")
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                print()
    
    finally:
        cap.release()
        print("Webcam closed.")


if __name__ == '__main__':
    main()