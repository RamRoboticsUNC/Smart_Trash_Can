#!/usr/bin/env python3
"""
live_trash_classifier_pi5.py
----------------------------
Pi 5, gpiozero + libgpiod (GPIODFactory)
Press SPACE -> classify current frame
    recycle  → motor forward  2 s
    trash    → motor reverse 2 s

L298N wiring (BCM):
    IN1 = 16  (forward)
    IN2 = 18  (backward)
    ENA = 22  (enable, full‑speed)

Quit with 'q' or Esc.  Exit‑code 1 = recycle, 0 = trash.
"""

# ────────────────────────── standard lib ───────────────────────────
import sys, argparse, time
from datetime import datetime

# ────────────────────────── third‑party ────────────────────────────
import cv2
import torch
from torchvision import transforms
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from PIL import Image

from gpiozero import Motor, Device
from gpiozero.pins.gpiod import GPIODFactory        # ← Pi5‑friendly backend

# ─────────────── GPIO / Motor (libgpiod backend) ──────────────────
Device.pin_factory = GPIODFactory()                 # no sudo needed

MOTOR = Motor(forward=16, backward=18, enable=22, pwm=False)

def actuate_motor(recycle_flag: int, seconds: float = 2.0) -> None:
    """Run DC motor forward (recycle) or backward (trash) for N seconds."""
    MOTOR.forward() if recycle_flag else MOTOR.backward()
    time.sleep(seconds)
    MOTOR.stop()

# ─────────────────── model, labels, transforms ────────────────────
weights = Inception_V3_Weights.IMAGENET1K_V1
LABELS  = weights.meta["categories"]                # 1000 ImageNet classes
model   = inception_v3(weights=weights, aux_logits=True).eval().to("cpu")

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

# ────────────────── single‑frame inference helper ─────────────────
@torch.inference_mode()
def infer(img: Image.Image, k: int = 5) -> list[tuple[str, float]]:
    tens   = preprocess(img).unsqueeze(0)           # (1,3,299,299)
    out    = model(tens)

    logits = (
        out[0] if isinstance(out, (tuple, list)) else
        out.logits if hasattr(out, "logits") else
        out
    ).squeeze(0)                                    # (1000,)

    probs  = torch.softmax(logits, dim=0)
    top_p, top_i = probs.topk(k)
    return [(LABELS[i], float(p)) for i, p in zip(top_i.tolist(), top_p.tolist())]

# ──────────────────────────── main loop ───────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--top-k", type=int, default=5, metavar="N",
                    help="How many predictions to print (default 5)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit("Error: Could not open webcam")

    latest_flag = 0
    print("Press SPACE to classify frame, 'q' or Esc to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame read failed")
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # quit on 'q' or Esc
            break

        if key == 32:              # 32 = SPACEBAR
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            top_k = infer(pil, max(1, args.top_k))

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] Top‑{len(top_k)} predictions:")
            for r, (lbl, p) in enumerate(top_k, 1):
                print(f"  #{r:<2}  {lbl:<25} {p*100:5.1f}%")

            latest_flag = int(any(tok in top_k[0][0] for tok in RECYCLABLE))
            print(f"Decision: {'RECYCLE' if latest_flag else 'TRASH'}")

            actuate_motor(latest_flag)

    cap.release()
    cv2.destroyAllWindows()
    MOTOR.stop()
    sys.exit(latest_flag)

# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
