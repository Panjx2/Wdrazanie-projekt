
"""Export trained cat classifier checkpoints to ONNX format.

This script discovers the preferred checkpoint (favoring MobileNetV3 finetunes),
rebuilds the appropriate TorchVision model head based on `labels.json`, loads
weights with common wrapper prefixes stripped, and produces a dynamic-batch
ONNX graph under ``assets/models``. Run it with an optional checkpoint path to
override auto-detection:

    python scripts/export_to_onnx.py path/to/checkpoint.pth
"""

import json, sys, re
from pathlib import Path
import torch
from torchvision import models

# ---- Paths (relative to this script) ----
ROOT = Path(__file__).resolve().parent.parent   # -> my-cat-classifier/
ASSETS = ROOT / "assets"
MODELS = ASSETS / "models"
SCRIPTS = ROOT / "scripts"

DEFAULT_CKPT_CANDIDATES = (
    MODELS / "mobilenetv3_finetuned.pth",  # bundled inside the app
)
LABELS_JSON = ASSETS / "labels.json"


def resolve_checkpoint(arg: str | None) -> Path:
    if arg:
        raw = Path(arg)
        return raw if raw.is_absolute() else (ROOT / raw).resolve()
    for candidate in DEFAULT_CKPT_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit("[ERROR] No checkpoint found. Pass a path or place mobilenetv3_finetuned.pth inside assets/models.")


IN_PTH = resolve_checkpoint(sys.argv[1] if len(sys.argv) > 1 else None)
OUT_ONNX = (MODELS / f"{IN_PTH.stem}.onnx").resolve()

print(f"[INFO] Using checkpoint: {IN_PTH}")

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
print("[INFO] Selected architecture: mobilenet_v3_large")
base = models.mobilenet_v3_large(weights=None)

head_name = None
for name, module in base.named_modules():
    if module is base.classifier[-1]:
        head_name = name
        break

if not isinstance(base.classifier[-1], torch.nn.Linear):
    sys.exit("[ERROR] Unexpected classifier layout. Update export_to_onnx.py to find the final Linear layer.")

base.classifier[-1] = torch.nn.Linear(base.classifier[-1].in_features, num_classes)

missing, unexpected = base.load_state_dict(sd, strict=False)
print("[INFO] missing keys:", missing)
print("[INFO] unexpected keys:", unexpected)

if head_name:
    head_missing = [k for k in missing if k.startswith(f"{head_name}.")]
    if head_missing:
        sys.exit("[ERROR] Trained classifier head missing from checkpoint. Export aborted.")

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
