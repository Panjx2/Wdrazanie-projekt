from ultralytics import YOLO
import os

HOME = os.getcwd()

BEST_MODEL_PATH = f"{HOME}/runs/detect/train3/weights/best.pt"

# 5. ==== EXPORT DO ONNX ====================================

model = YOLO(BEST_MODEL_PATH)
model.export(format="onnx")

import shutil

shutil.copy(BEST_MODEL_PATH, f"{HOME}/model/best.pt")
shutil.copy(f"{HOME}/runs/detect/train3/weights/best.onnx", f"{HOME}/model/best.onnx")