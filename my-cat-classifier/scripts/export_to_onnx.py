import json
from pathlib import Path
import torch

# --- Ścieżki ---
IN_PTH = Path("assets/models/mobilenetv2_finetuned.pth")
OUT_ONNX = Path("assets/models/mobilenetv2_finetuned.onnx")
LABELS_JSON = Path("assets/labels.json")
OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

# --- Liczba klas z labels.json ---
num_classes = None
if LABELS_JSON.exists():
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, list) and len(labels) > 0:
        num_classes = len(labels)
        print(f"[INFO] Wykryto {num_classes} klas z assets/labels.json")

# --- Wczytaj checkpoint (.pth) ---
obj = torch.load(IN_PTH, map_location="cpu")
from torchvision import models
base = models.mobilenet_v2(weights=None)

# Sprawdź, czy to pełny model czy tylko state_dict
if isinstance(obj, torch.nn.Module):
    print("[INFO] Załadowano pełny model, wyciągam state_dict()")
    sd = obj.state_dict()
else:
    print("[INFO] Wykryto state_dict w .pth")
    sd = obj
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

# Usuń prefiks "module." i pomiń classifier (będzie dopasowany do liczby klas)
from collections import OrderedDict
filtered = OrderedDict()
for k, v in sd.items():
    nk = k.replace("module.", "") if k.startswith("module.") else k
    if nk.startswith("classifier."):
        continue
    filtered[nk] = v

missing, unexpected = base.load_state_dict(filtered, strict=False)
if missing:
    print("[WARN] Brakujące klucze:", missing)
if unexpected:
    print("[WARN] Nieoczekiwane klucze:", unexpected)

# Podmień classifier na odpowiednią liczbę klas
if num_classes is None:
    num_classes = 12
base.classifier[1] = torch.nn.Linear(base.last_channel, num_classes)
print(f"[INFO] Ustawiono classifier na {num_classes} klas")

# --- Eksport do ONNX ---
from torch.onnx import utils as onnx_utils

base.eval()
dummy = torch.randn(1, 3, 224, 224)
print("[INFO] Eksport klasycznym backendem torch.onnx.utils.export ...")

onnx_utils.export(
    base,
    dummy,
    str(OUT_ONNX),
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True,
    verbose=False
)

print(f"[OK] Zapisano ONNX: {OUT_ONNX.resolve()}")
