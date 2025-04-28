#!/usr/bin/env python3
"""
live_trash_classifier_space.py
------------------------------
OpenCV video preview + ImageNet inference on-demand:
press the SPACEBAR to classify the current frame, 'q' or Esc to quit.
When you classify, send “R” (recycle) or “T” (trash) over serial to the Arduino.
"""

import sys, argparse
from pathlib import Path
from datetime import datetime

import cv2
import torch
from torchvision import transforms
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from PIL import Image

import serial  # pip install pyserial

# ---------- model & labels ----------
weights = Inception_V3_Weights.IMAGENET1K_V1
IMNET_LABELS = weights.meta["categories"]
model = inception_v3(weights=weights, aux_logits=True).eval().to("cpu")

# ---------- preprocessing ----------
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

RECYCLABLE = {
    'can','water bottle','pop bottle','soda bottle','wine bottle',
    'envelope','mailbag','postbag','menu','comic book','crossword puzzle',
    'cardigan','bath towel','wooden spoon','chain','cloak','bow',
    'ruler','suit','sunglasses','pajama','pyjama',"pj's",'jammies'
}

@torch.inference_mode()
def infer(pil_img: Image.Image, k: int) -> list[tuple[str, float]]:
    tensor = preprocess(pil_img).unsqueeze(0)
    out    = model(tensor)
    # handle different return types
    if isinstance(out, (tuple, list)):
        logits = out[0]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    logits = logits.squeeze(0)
    probs  = torch.softmax(logits, dim=0)
    top_p, top_i = probs.topk(k)
    return [(IMNET_LABELS[i], float(p)) for i, p in zip(top_i.tolist(), top_p.tolist())]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",    type=int,   default=0,              help="Camera index")
    ap.add_argument("--top-k",  type=int,   default=5, metavar="N", help="How many predictions")
    ap.add_argument("--port",   type=str,   default="/dev/ttyACM0",  help="Serial port")
    ap.add_argument("--baud",   type=int,   default=9600,           help="Serial baud rate")
    args = ap.parse_args()

    # open video
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit("Error: Could not open webcam")

    # open serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Opened serial {args.port} @ {args.baud}bps")
    except serial.SerialException as e:
        sys.exit(f"Error opening serial port: {e}")

    print("Press SPACE to classify. Press 'q' or Esc to quit.")
    latest_flag = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            if key == 32:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                top_k = infer(pil_img, max(1, args.top_k))

                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{ts}] Top-{len(top_k)} predictions:")
                for rank, (lbl, p) in enumerate(top_k, 1):
                    print(f"  #{rank:<2} {lbl:<25} {p*100:5.1f}%")

                # decide recycle vs trash
                latest_flag = int(any(tok in top_k[0][0] for tok in RECYCLABLE))
                if latest_flag:
                    outcome, byte = "RECYCLE", b'R'
                else:
                    outcome, byte = "TRASH",   b'T'

                print(f"Decision based on #1: {outcome}")
                ser.write(byte)
                ser.flush()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()

    sys.exit(latest_flag)

if __name__ == "__main__":
    main()
