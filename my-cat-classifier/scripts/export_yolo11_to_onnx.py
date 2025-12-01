"""Export YOLO11 checkpoint to ONNX with built-in NMS for the mobile app.

Usage:
    python scripts/export_yolo11_to_onnx.py [path/to/yolo11.pt] [--imgsz 640]

The script relies on the `ultralytics` package. It writes the ONNX file
into ``assets/models/`` next to the supplied checkpoint so Metro can bundle
it for React Native.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"
DEFAULT_PT = MODELS / "yolo11s.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11 to ONNX for the RN app")
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_PT), help="Path to yolo11.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size (square)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset")
    parser.add_argument("--half", action="store_true", help="Export half precision")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch (recommended for varying input counts)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise SystemExit(f"[ERROR] Checkpoint not found: {ckpt}")

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import guard only
        raise SystemExit("[ERROR] Install ultralytics first: pip install ultralytics") from exc

    print(f"[INFO] Loading checkpoint: {ckpt}")
    model = YOLO(str(ckpt))

    print("[INFO] Exporting to ONNX (with NMS head)")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=args.dynamic,
        half=args.half,
        nms=True,
    )

    onnx_path = Path(export_path).resolve()
    target = MODELS / f"{ckpt.stem}.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)

    if onnx_path != target:
        shutil.move(str(onnx_path), target)

    labels_path = ROOT / "assets" / "labels_yolo11.json"
    try:
        names = model.model.names  # type: ignore[attr-defined]
        if isinstance(names, dict):
            labels = [name for _, name in sorted(names.items())]
        elif isinstance(names, (list, tuple)):
            labels = list(names)
        else:  # pragma: no cover - defensive fallback
            labels = []
        labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        print(f"[OK] Labels saved to: {labels_path}")
    except Exception as exc:  # pragma: no cover - metadata helper
        print(f"[WARN] Could not save labels: {exc}")

    print(f"[OK] ONNX saved to: {target}")
    print("Place the resulting .onnx (and .onnx.data if generated) under assets/models/ for bundling.")


if __name__ == "__main__":
    main()
