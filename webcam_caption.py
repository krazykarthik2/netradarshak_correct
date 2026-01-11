"""
webcam_caption.py

Capture frames from webcam and get captions from a local VLM model.

Usage:
  python webcam_caption.py --model_dir ./ --auto

Controls:
  c     - capture a frame and send to model (saves to temp/images, opens in Chrome)
  q     - quit

This script attempts to load a HF-style vision->text pipeline from the provided
model directory. If that fails it will attempt to POST the image to a local
HTTP server at http://localhost:8000/predict. Adjust as needed for your setup.
"""
import argparse
import io
import select
import sys
import time
import subprocess
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    print("Error: OpenCV (cv2) is required. Install with: pip install opencv-python")
    raise

from PIL import Image

def pil_from_bgr(bgr_frame):
    # Convert BGR (OpenCV) to RGB and create PIL image
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
            subprocess.Popen(['chrome', abs_path])
            print(f"Opened in Chrome: {abs_path}")
        except Exception as e:
            print(f"Could not open in Chrome: {e}")
        return filename
    else:
        print("Failed to save frame")
        return None

def try_load_transformers(model_dir, device):
    """Try to load HF-compatible vision-to-text model from model_dir.
    Returns a callable caption(image: PIL.Image) -> str or None if can't load."""
    import torch
    model_dir = str(Path(model_dir))

    # First try BLIP-specific loader (BlipForConditionalGeneration + BlipProcessor)
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        print(f"Trying to load BLIP model/processor from {model_dir} (or HF id)...")
        processor = BlipProcessor.from_pretrained(model_dir)
        model = BlipForConditionalGeneration.from_pretrained(model_dir)
        torch_device = torch.device('cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        model.to(torch_device)

        gen_kwargs = dict(max_length=64, num_beams=4)

        def caption(image: Image.Image) -> str:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = processor(images=image, return_tensors='pt').to(torch_device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            caption_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return caption_text.strip()

        return caption
    except Exception as e:
        print(f"BLIP loader failed: {e}")

    # Fallback: generic VisionEncoderDecoderModel
    try:
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        print(f"Trying to load VisionEncoderDecoderModel from {model_dir} (this may take a while)...")
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        processor = AutoProcessor.from_pretrained(model_dir)

        torch_device = torch.device('cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        model.to(torch_device)

        gen_kwargs = dict(max_length=64, num_beams=4)

        def caption(image: Image.Image) -> str:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            pixel_values = processor(images=image, return_tensors='pt').pixel_values.to(torch_device)
            with torch.no_grad():
                output_ids = model.generate(pixel_values, **gen_kwargs)
            caption_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return caption_text.strip()

        return caption
    except Exception as e:
        print(f"Transformer loader failed: {e}")
        return None

import base64
import json
import requests

def caption_via_http(image: Image.Image, url: str):
    # encode image to JPEG in-memory
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    files = {'image': ('frame.jpg', buf, 'image/jpeg')}
    try:
        r = requests.post(url, files=files, timeout=30)
        r.raise_for_status()
        data = r.json()
        # expect {'caption': 'text'} or string
        if isinstance(data, dict) and 'caption' in data:
            return data['caption']
        if isinstance(data, str):
            return data
        return str(data)
    except Exception as e:
        return f"HTTP request failed: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='.', help='Path to model directory (default: current dir)')
    parser.add_argument('--auto', action='store_true', help='Auto-capture frames every N seconds')
    parser.add_argument('--interval', type=float, default=3.0, help='Interval seconds for auto mode')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu', help='Device to run model on')
    parser.add_argument('--http_url', default='http://localhost:80/caption/en', help='Fallback HTTP endpoint for captioning')
    parser.add_argument('--max_captures', type=int, default=0, help='Stop after this many captures (0 = unlimited)')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    caption_fn = try_load_transformers(model_dir, args.device)
    if caption_fn is None:
        print("Falling back to HTTP endpoint at", args.http_url)
        caption_fn = lambda img: caption_via_http(img, args.http_url)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. If you're on Windows, ensure camera access permission is enabled.")
        sys.exit(1)

    last_auto = 0.0
    captures_done = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame from webcam')
                break

            now = time.time()
            do_capture = False
            if args.auto and (now - last_auto) >= args.interval:
                do_capture = True
                last_auto = now

            # Non-blocking keyboard input
            key = -1
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                try:
                    user_input = sys.stdin.read(1).lower()
                    if user_input == 'q':
                        key = ord('q')
                    elif user_input == 'c':
                        key = ord('c')
                except Exception:
                    pass
            
            if key == ord('q'):
                break
            if key == ord('c'):
                do_capture = True

            if do_capture:
                # Save frame and open in Chrome
                save_frame_and_open_chrome(frame)
                
                pil = pil_from_bgr(frame)
                print('\nCaptured frame, sending to model...')
                start = time.time()
                caption = caption_fn(pil)
                elapsed = time.time() - start
                print(f'Caption: {caption}')
                print(f'(in {elapsed:.2f}s)')
                captures_done += 1
                if args.max_captures > 0 and captures_done >= args.max_captures:
                    print(f'Reached --max_captures={args.max_captures}, exiting.')
                    break

    finally:
        cap.release()
        print("Webcam closed.")

if __name__ == '__main__':
    main()
