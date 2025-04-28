#!/usr/bin/env python3
"""
classify_trash.py
-----------------
Light‑weight, headless image‑classifier for a Raspberry Pi–based smart trash‑can.
 • No web server, no database calls – just local inference.
 • Returns shell exit‑code   0 = trash / 1 = recycle   for easy motor control.

Usage
-----`````
python classify_trash.py --image /path/to/frame.jpg
# or, to fall back on a demo image:
python classify_trash.py
"""

# ---------- Standard lib imports ----------
import argparse, sys, pickle
from pathlib import Path
from datetime import datetime

# ---------- Third‑party imports ----------
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.inception import inception_v3   # ImageNet‑pretrained

# ---------- Paths & lists ----------
ROOT = Path(__file__).resolve().parent
LABEL_PATH   = ROOT / "config" / "imagenet.p"           # pickled {idx:label}
DEFAULT_IMG  = "/home/ianre/Desktop/test.png"             # demo image

# Labels we consider recyclable *out‑of‑the‑box*.
RECYCLABLE = {
    'can', 'water bottle', 'pop bottle', 'soda bottle', 'wine bottle',
    'envelope', 'mailbag', 'postbag', 'menu', 'comic book',
    'crossword puzzle', 'cardigan', 'bath towel', 'wooden spoon',
    'chain', 'cloak', 'bow', 'ruler', 'suit', 'sunglasses',
    'pajama', 'pyjama', "pj's", 'jammies'
}

# ---------- Pre‑processing ----------
preprocess = transforms.Compose([
    transforms.Resize(299),            # Inception‑v3 native input size
    transforms.CenterCrop(299),
    transforms.ToTensor(),             # [0,1]  → tensor
    transforms.Normalize(              # ImageNet per‑channel stats
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]),
])

# ---------- Model loading ----------
def load_model(device: torch.device):
    """
    Loads a pretrained Inception‑v3 (no aux‑logits) and puts it on CPU or GPU.
    For fastest Pi inference swap this for MobileNet‑v3 or EfficientNet‑Lite.
    """
    model = inception_v3(pretrained=True, aux_logits=True)
    model.eval().to(device)
    return model

# ---------- Inference core ----------
@torch.inference_mode()
def classify(img_path: Path, device: torch.device) -> tuple[int, str]:
    """
    Returns (status, label) where status=1 means recyclable, 0 otherwise.
    """
    # 1. Load image → tensor
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 2. Inference
    logits = MODEL(input_tensor)
    idx    = logits.argmax(1).item()

    # 3. Human‑readable label lookup
    with open(LABEL_PATH, "rb") as fp:
        labels = pickle.load(fp)       # ImageNet synset strings
    label = labels[idx]

    # 4. Recyclable? (simple lookup for now)
    recycle_flag = int(any(tok in label for tok in RECYCLABLE))

    return recycle_flag, label

# ---------- CLI entry‑point ----------
def main():
    parser = argparse.ArgumentParser(
        description="Classify an image as recyclable (exit‑code 1) or trash (0).")
    parser.add_argument("--image", "-i", type=Path, default=DEFAULT_IMG,
                        help=f"Path to image file [default: {DEFAULT_IMG}]")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"[Error] Could not find {args.image}")

    status, label = classify(args.image, DEVICE)
    outcome = "RECYCLE" if status else "TRASH"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Detected: {label}  →  {outcome}")

    # Exit‑code read by the robot’s control script
    sys.exit(status)          # 0 = trash, 1 = recycle


# ---------- Bootstrapping ----------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL  = load_model(DEVICE)
    main()
