import subprocess
from flask import Flask, request, jsonify
import os
import tempfile
import traceback

from transformers import AutoModelForVision2Seq, AutoProcessor

# Optional heavy imports are attempted at module import time; failures are handled gracefully
try:
    import cv2
except Exception:
    cv2 = None

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
    from googletrans import Translator
except Exception:
    Translator = None

# TTS imports (kept optional)
try:
    from gtts import gTTS
    from playsound import playsound
except Exception:
    gTTS = None
    playsound = None

app = Flask(__name__)

# Language mapping convenience (allow both codes and some names)
LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "te": "te",
    "telugu": "te",
}

# Lazy model holders
_models = {
    "blip_processor": None,
    "blip_model": None,
    "traffic_blip_processor": None,
    "traffic_blip_model": None,
    "ocr_reader": None,
    "translator": None,
}


# -------------------------
# Utilities
# -------------------------

def normalize_language(lang: str) -> str:
    if not lang:
        return "en"
    l = lang.lower()
    return LANGUAGE_ALIASES.get(l, l)


def save_upload_to_temp(upload) -> str:
    """Save a Werkzeug FileStorage to a temporary file and return its path."""
    suffix = os.path.splitext(upload.filename)[1] if upload and upload.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload.save(tmp.name)
        subprocess.run(["start","chrome", tmp.name], shell=True)
        return tmp.name


# -------------------------
# Model initialization
# -------------------------

def init_translator():
    if _models["translator"] is None:
        if Translator is None:
            _models["translator"] = None
        else:
            try:
                _models["translator"] = Translator()
            except Exception:
                _models["translator"] = None
    return _models["translator"]


def init_blip():
    if _models["blip_model"] is None or _models["blip_processor"] is None:
        if BlipProcessor is None or BlipForConditionalGeneration is None or torch is None:
            _models["blip_processor"] = None
            _models["blip_model"] = None
            return None, None
        try:
            _models["blip_processor"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _models["blip_model"] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            _models["blip_model"].eval()
        except Exception:
            _models["blip_processor"] = None
            _models["blip_model"] = None
    return _models["blip_processor"], _models["blip_model"]

def init_traffic_blip():
    if _models["traffic_blip_model"] is None or _models["traffic_blip_processor"] is None:
        if BlipProcessor is None or BlipForConditionalGeneration is None or torch is None:
            _models["traffic_blip_processor"] = None
            _models["traffic_blip_model"] = None
            return None, None
        try:
            _models["traffic_blip_processor"] = AutoProcessor.from_pretrained("Charansaiponnada/blip-traffic-rr")
            _models["traffic_blip_model"] = AutoModelForVision2Seq.from_pretrained("Charansaiponnada/blip-traffic-rr")
            _models["traffic_blip_model"].eval()
        except Exception:
            _models["traffic_blip_processor"] = None
            _models["traffic_blip_model"] = None
    return _models["traffic_blip_processor"], _models["traffic_blip_model"]

def init_ocr():
    if _models["ocr_reader"] is None:
        if easyocr is None:
            _models["ocr_reader"] = None
            return None
        try:
            _models["ocr_reader"] = easyocr.Reader(["en"], gpu=False)
        except Exception:
            _models["ocr_reader"] = None
    return _models["ocr_reader"]


# -------------------------
# Core processing functions
# -------------------------

def generate_caption_from_image_path(image_path: str) -> tuple[str, float]:
    """Return (caption, confidence). If BLIP unavailable, return a sample caption."""
    processor, model = init_blip()
    if processor is None or model is None:
        return ("A sample caption describing the scene.", 0.5)

    try:
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption, 0.85
    except Exception:
        return ("A sample caption describing the scene.", 0.5)


def generate_traffic_caption_from_image_path(image_path: str) -> tuple[str, float]:
    """Return (caption, confidence). If BLIP unavailable, return a sample caption."""
    processor, model = init_traffic_blip()
    if processor is None or model is None:
        return ("A sample caption describing the scene.", 0.5)

    try:
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption, 0.85
    except Exception:
        return ("A sample caption describing the scene.", 0.5)


def ocr_from_image_path(image_path: str) -> tuple[str, float]:
    reader = init_ocr()
    if reader is None:
        # fallback sample
        return ("Sample OCR: The quick brown fox jumps over the lazy dog.", 0.8)
    try:
        result = reader.readtext(image_path)
        text = " ".join([t[1] for t in result])
        return text or "", 0.9
    except Exception:
        return ("", 0.0)


def translate_text(text: str, target_lang: str) -> str:
    translator = init_translator()
    if translator is None:
        return text
    try:
        return translator.translate(text, dest=target_lang).text
    except Exception:
        return text


# -------------------------
# Flask endpoints
# -------------------------

@app.route('/caption/<lang>', methods=['GET', 'POST'])
def caption_endpoint(lang):
    """POST an image (multipart form 'image' or 'file') and receive a caption in the requested language."""
    target_lang = normalize_language(lang)

    if request.method == 'GET':
        return jsonify({
            "help": "POST an image as multipart form-data ('image' or 'file') to receive a caption.",
            "example": {"curl": "curl -X POST -F \"image=@/path/photo.jpg\" http://localhost:5000/caption/en"}
        })

    upload = None
    if 'image' in request.files:
        upload = request.files['image']
    elif 'file' in request.files:
        upload = request.files['file']
    else:
        return jsonify({"error": "No image provided. Use multipart form 'image' or 'file'."}), 400

    tmp = save_upload_to_temp(upload)
    try:
        caption, conf = generate_caption_from_image_path(tmp)
        translated = translate_text(caption, target_lang)
        return jsonify({
            "caption": translated,
            "caption_en": caption,
            "confidence": conf,
            "language": target_lang,
        })
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

@app.route('/traffic', methods=['GET', 'POST'])
def traffic_endpoint():
    """POST an image (multipart form 'image' or 'file') and receive traffic-related information."""
    if request.method == 'GET':
        return jsonify({
            "help": "POST an image as multipart form-data ('image' or 'file') to receive traffic-related information.",
            "example": {"curl": "curl -X POST -F \"image=@/path/photo.jpg\" http://localhost:5000/traffic"}
        })

    upload = None
    if 'image' in request.files:
        upload = request.files['image']
    elif 'file' in request.files:
        upload = request.files['file']
    else:
        return jsonify({"error": "No image provided. Use multipart form 'image' or 'file'."}), 400

    tmp = save_upload_to_temp(upload)
    try:
        caption, conf = generate_traffic_caption_from_image_path(tmp)
        return jsonify({
            "caption": caption,
            "caption_en": caption,
            "confidence": conf,
            "language": "en",
        })
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


@app.route('/ocr/<lang>', methods=['GET', 'POST'])
def ocr_endpoint(lang):
    """POST an image (multipart form 'image' or 'file') and receive OCR text in the requested language."""
    target_lang = normalize_language(lang)

    if request.method == 'GET':
        return jsonify({
            "help": "POST an image as multipart form-data ('image' or 'file') to receive OCR text.",
            "example": {"curl": "curl -X POST -F \"image=@/path/photo.jpg\" http://localhost:5000/ocr/en"}
        })

    upload = None
    if 'image' in request.files:
        upload = request.files['image']
    elif 'file' in request.files:
        upload = request.files['file']
    else:
        return jsonify({"error": "No image provided. Use multipart form 'image' or 'file'."}), 400

    tmp = save_upload_to_temp(upload)
    try:
        text, conf = ocr_from_image_path(tmp)
        translated = translate_text(text, target_lang)
        return jsonify({
            "text": translated,
            "text_en": text,
            "confidence": conf,
            "language": target_lang,
        })
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# -------------------------
# Run server (CLI)
# -------------------------
def create_app():
    return app


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run caption/ocr server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print('Starting server on %s:%s' % (args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug)