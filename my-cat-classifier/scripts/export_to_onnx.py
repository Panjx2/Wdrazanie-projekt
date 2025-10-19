
import json, sys, re
from pathlib import Path
import torch
from torchvision import models

# ---- Paths (relative to this script) ----
ROOT = Path(__file__).resolve().parent.parent   # -> my-cat-classifier/
ASSETS = ROOT / "assets"
MODELS = ASSETS / "models"
SCRIPTS = ROOT / "scripts"

IN_PTH = MODELS / "mobilenetv2_finetuned.pth"
OUT_ONNX = MODELS / "mobilenetv2_finetuned.onnx"
LABELS_JSON = ASSETS / "labels.json"

# ---- Sanity checks ----
if not LABELS_JSON.exists():
    sys.exit(f"[ERROR] labels.json not found at {LABELS_JSON}. "
             "Generate it from your training dataset order first.")
if not IN_PTH.exists():
    sys.exit(f"[ERROR] model checkpoint (.pth) not found at {IN_PTH}")

OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

# ---- Load labels ----
labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
num_classes = len(labels)
print(f"[INFO] num_classes = {num_classes}")

# ---- Load checkpoint ----
obj = torch.load(IN_PTH, map_location="cpu")
sd = obj.state_dict() if isinstance(obj, torch.nn.Module) else obj
if "state_dict" in sd: sd = sd["state_dict"]
if "model_state_dict" in sd: sd = sd["model_state_dict"]

def strip_prefix(k: str) -> str:
    for p in ("module.", "model.", "net."):
        if k.startswith(p):
            return k[len(p):]
    return k
sd = {strip_prefix(k): v for k, v in sd.items()}

# ---- Build model and load weights ----
base = models.mobilenet_v2(weights=None)
base.classifier[1] = torch.nn.Linear(base.last_channel, num_classes)

missing, unexpected = base.load_state_dict(sd, strict=False)
print("[INFO] missing keys:", missing)
print("[INFO] unexpected keys:", unexpected)

if any(k in ("classifier.1.weight", "classifier.1.bias") for k in missing):
    sys.exit("[ERROR] Trained classifier head missing from checkpoint "
             "(classifier.1.*). Export aborted.")

# ---- Export to ONNX ----
base.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    base, dummy, str(OUT_ONNX),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=18, do_constant_folding=True, verbose=False
)

print(f"[OK] Exported to: {OUT_ONNX.resolve()}")
