"""Export YOLO checkpoints to ONNX for the React Native app.

This script wraps the Ultralytics `YOLO.export` helper so you can
convert trained YOLOv8/YOLO11 weights into ONNX files that the app
loads from `assets/models/`.

Examples:
    python scripts/export_yolo_to_onnx.py yolov8n.pt
    python scripts/export_yolo_to_onnx.py runs/detect/train/weights/best.pt --img 640 --batch 1 --dynamic

The output ONNX is written to `assets/models/` by default so Metro
bundles it automatically. If the exporter emits an external `.onnx.data`
file, place it next to the ONNX before bundling.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "assets" / "models"


def require_ultralytics():
    """Return the Ultralytics YOLO class or exit with guidance."""
    if importlib.util.find_spec("ultralytics") is None:
        sys.exit(
            "[ERROR] Ultralytics is not installed. Install it first: pip install ultralytics"
        )
    from ultralytics import YOLO

    return YOLO


def resolve_rooted(path_str: str) -> Path:
    raw = Path(path_str)
    return raw if raw.is_absolute() else (ROOT / raw).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="Path to YOLO checkpoint (.pt)")
    parser.add_argument(
        "--img",
        type=int,
        default=640,
        help="Square input size expected by the model (e.g., 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size (use >1 only if your model supports it)",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic axes in the ONNX graph"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string understood by Ultralytics (e.g., cpu, cuda:0)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to export with",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run ONNX simplification after export (requires onnxsim)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTDIR),
        help="Where to place the ONNX file (default: assets/models)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base filename (without extension) for the exported ONNX",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO = require_ultralytics()

    weights = resolve_rooted(args.weights)
    if not weights.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {weights}")

    out_dir = resolve_rooted(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_base = args.name or weights.stem
    out_path = out_dir / f"{out_base}.onnx"

    print(f"[INFO] Loading weights from {weights}")
    model = YOLO(str(weights))

    print(
        "[INFO] Exporting to ONNX...\n"
        f"       target path: {out_path}\n"
        f"       imgsz: {args.img}\n"
        f"       batch: {args.batch}\n"
        f"       dynamic: {args.dynamic}\n"
        f"       opset: {args.opset}\n"
        f"       simplify: {args.simplify}\n"
        f"       device: {args.device}"
    )

    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=args.img,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            device=args.device,
            batch=args.batch,
            project=str(out_dir),
            name=out_base,
        )
    )

    if exported_path.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(exported_path.read_bytes())
        exported_path.unlink()

    print(
        "[OK] Export complete. If an .onnx.data file was created, keep it next\n"
        f"to the ONNX in assets/models/ so the app can load both. Final path: {out_path}"
    )


if __name__ == "__main__":
    main()
