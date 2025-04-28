#!/usr/bin/env python3
"""
classify_trash_cpu.py  – v2  (top‑K output)
-------------------------------------------
CPU‑only ImageNet inference for a Raspberry Pi smart‑trash‑can.

Usage examples
--------------
python classify_trash_cpu.py -i photo.jpg               # top‑5 predictions
python classify_trash_cpu.py -i photo.jpg --top-k 10    # top‑10 predictions
"""

# ---------- std‑lib ----------
import argparse, pickle, sys
from pathlib import Path
from datetime import datetime

# ---------- third‑party ----------
import torch          # CPU wheel only
from torchvision import transforms
from torchvision.models.inception import inception_v3
from PIL import Image

from pathlib import Path   # ← already imported earlier; shown for clarity

# ---------- paths / constants ----------
# Keep raw strings
LABEL_PATH_STR   = "/home/ianre/Smart_Trash_Can/Smartsort/trashdetectionapi/config/imagenet.p"
DEFAULT_IMG_STR  = "/home/ianre/Smart_Trash_Can/images/test.png"   # unchanged

# Convert to Path objects when you actually need them
LABEL_PATH  = Path(LABEL_PATH_STR)
DEFAULT_IMG = Path(DEFAULT_IMG_STR)


RECYCLABLE = {
    'can','water bottle','pop bottle','soda bottle','wine bottle',
    'envelope','mailbag','postbag','menu','comic book','crossword puzzle',
    'cardigan','bath towel','wooden spoon','chain','cloak','bow',
    'ruler','suit','sunglasses','pajama','pyjama',"pj's",'jammies'
}

# ---------- transforms ----------
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ---------- model ----------
def load_model() -> torch.nn.Module:
    # Must keep aux_logits=True to match pretrained weights
    model = inception_v3(pretrained=True, aux_logits=True)
    model.eval().to("cpu")
    return model

# ---------- inference ----------
@torch.inference_mode()
def classify(img_path: str | Path,
             model: torch.nn.Module,
             k: int) -> tuple[int, list[tuple[str, float]]]:

    img_path = Path(img_path)                  # accept str *or* Path
    img      = Image.open(img_path).convert("RGB")
    batch    = preprocess(img).unsqueeze(0)    # (1, 3, 299, 299)

    out      = model(batch)
    logits   = out.logits if hasattr(out, "logits") else out
    if isinstance(logits, tuple):              # safety: some models return a tuple
        logits = logits[0]

    logits = logits.squeeze(0)                 # → (1000,)  remove batch dim
    probs  = torch.softmax(logits, dim=0)      # softmax over the only remaining dim

    top_p, top_i = probs.topk(k)

    with open(LABEL_PATH, "rb") as fp:
        labels = pickle.load(fp)

    top_k = [(labels[idx], float(p)) for idx, p in zip(top_i.tolist(), top_p.tolist())]

    recycle_flag = int(any(tok in top_k[0][0] for tok in RECYCLABLE))
    return recycle_flag, top_k

# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i","--image", type=Path, default=DEFAULT_IMG,
                   help=f"Image file to classify (default {DEFAULT_IMG})")
    p.add_argument("--top-k", type=int, default=5, metavar="N",
                   help="How many top predictions to display (default 5)")
    args = p.parse_args()

    if not args.image.exists():
        sys.exit(f"[error] file not found: {args.image}")

    status, top_k = classify(args.image, MODEL, max(1, args.top_k))

    # ---- pretty‑print results ----
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{ts}] Top‑{len(top_k)} predictions:")
    for rank, (label, p) in enumerate(top_k, 1):
        print(f"  #{rank:<2}  {label:<25}  {p*100:5.1f}%")

    outcome = "RECYCLE" if status else "TRASH"
    print(f"\nDecision based on #1: {outcome}")
    sys.exit(status)          # 1 = recycle, 0 = trash

# ---------- bootstrap ----------
if __name__ == "__main__":
    MODEL = load_model()
    main()
