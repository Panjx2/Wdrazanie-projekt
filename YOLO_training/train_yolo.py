from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import supervision as sv
from ultralytics.nn.tasks import DetectionModel
import torch

# Add DetectionModel to PyTorch's safe globals allowlist
torch.serialization.add_safe_globals([DetectionModel])


HOME = os.getcwd()
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/output_data/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/output_data/train/images"
DATA_YAML_PATH = f"{HOME}/output_data/data.yaml"

# 1. Wczytanie datasetu YOLO w celu wizualizacji
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH
)

# 2. Podgląd kilku obrazów
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

images = []
names = []

for i, (image_path, img, ann) in enumerate(dataset):
    if i == 16:
        break

    annotated = img.copy()
    annotated = mask_annotator.annotate(annotated, ann)
    annotated = box_annotator.annotate(annotated, ann)
    annotated = label_annotator.annotate(annotated, ann)

    images.append(annotated)
    names.append(Path(image_path).name)

sv.plot_images_grid(
    images=images,
    titles=names,
    grid_size=(4, 4),
    size=(16, 10)
)

# 3. ==== TRENING YOLOv8 ====

model = YOLO("yolov8n.pt")       # baza YOLO
model.train(
    data=DATA_YAML_PATH,
    epochs=50,
    imgsz=640
)
