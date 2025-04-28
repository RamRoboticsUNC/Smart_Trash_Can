#!/usr/bin/env python3
"""
live_trash_classifier_space.py
------------------------------
OpenCV video preview + ImageNet inference on‑demand:
press the SPACEBAR to classify the current frame, 'q' or Esc to quit.
Sends "R" to Arduino via serial for recyclable items and "T" for trash.
"""

# --- std‑lib ---
import sys, argparse
import time
from pathlib import Path
from datetime import datetime

# --- third‑party ---
import cv2
import torch
import serial
from torchvision import transforms
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from PIL import Image

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

# ---------- single‑frame inference ----------
@torch.inference_mode()
def infer(pil_img: Image.Image, k: int) -> list[tuple[str, float]]:
    tensor = preprocess(pil_img).unsqueeze(0)           # (1,3,299,299)
    out    = model(tensor)

    # robustly handle different return types
    if isinstance(out, (tuple, list)):          # training‑mode tuple
        logits = out[0]
    elif hasattr(out, "logits"):                # namedtuple
        logits = out.logits
    else:                                       # plain tensor
        logits = out
    logits = logits.squeeze(0)                  # (1000,)

    probs  = torch.softmax(logits, dim=0)
    top_p, top_i = probs.topk(k)
    return [(IMNET_LABELS[i], float(p)) for i, p in zip(top_i.tolist(), top_p.tolist())]

# ---------- main loop ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--top-k", type=int, default=5, metavar="N",
                    help="How many predictions to print (default 5)")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", 
                    help="Serial port for Arduino (default /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=9600,
                    help="Baud rate for serial communication (default 9600)")
    args = ap.parse_args()

    # Set up serial connection to Arduino
    arduino = None
    try:
        arduino = serial.Serial(args.port, args.baud, timeout=1)
        #print(f"Connected to Arduino on {args.port} at {args.baud} baud")
        time.sleep(2)  # Allow Arduino to reset after serial connection
    except Exception as e:
        #print(f"Warning: Could not connect to Arduino: {e}")
        print("Continuing without Arduino connection...")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit("Error: Could not open webcam")

    latest_flag = 0  # recycle/trash decision of the last inference
    print("Press SPACE to classify current frame. Press 'q' or Esc to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):          # 27 = Esc
            break

        if key == 32:                      # 32 = spacebar
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            top_k = infer(pil_img, max(1, args.top_k))

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] Top‑{len(top_k)} predictions:")
            for rank, (lbl, p) in enumerate(top_k, 1):
                print(f"  #{rank:<2} {lbl:<25} {p*100:5.1f}%")

            is_recyclable = any(tok in top_k[0][0] for tok in RECYCLABLE)
            latest_flag = int(is_recyclable)
            
            # Send command to Arduino
            if arduino and arduino.is_open:
                command = "R" if is_recyclable else "T"
                arduino.write(command.encode())
                arduino.flush()
                print(f"Sent '{command}' to Arduino")
            
            outcome = "RECYCLE" if is_recyclable else "TRASH"
            print(f"Decision based on #1: {outcome}")

    # Clean up resources
    if arduino and arduino.is_open:
        arduino.close()
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(latest_flag)      # 1 = recycle, 0 = trash


if __name__ == "__main__":
    main()