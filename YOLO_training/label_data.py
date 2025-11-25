import supervision as sv
import os
from pathlib import Path
import cv2
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from ultralytics import YOLO


HOME = os.getcwd()
IMAGE_DIR_PATH = f"{HOME}/data/"

def list_all_files(directory, extensions):
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().split(".")[-1] in extensions:
                paths.append(os.path.join(root, file))
    return paths

IMAGE_DIR_PATH = f"{HOME}/data/"

image_paths = list_all_files(IMAGE_DIR_PATH, ["png", "jpg", "jpeg"])
print("image count:", len(image_paths))


SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)


image_paths = [Path(p) for p in image_paths]



titles = [
    image_path.stem
    for image_path
    in image_paths[:SAMPLE_SIZE]]
images = [
    cv2.imread(str(image_path))
    for image_path
    in image_paths[:SAMPLE_SIZE]]

sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)


ontology=CaptionOntology({
    "cat": "cat",
})

DATASET_DIR_PATH = f"{HOME}/output_data"


base_model = GroundedSAM(ontology=ontology)

# przechodzi po wszystkich folderach w IMAGE_DIR_PATH (rekurencyjnie)
for root, dirs, files in os.walk(IMAGE_DIR_PATH):

    # sprawdzamy czy są pliki PNG
    has_images = any(f.lower().endswith(".jpg") for f in files)

    if has_images:
        print(f"Labeluję folder: {root}")
        base_model.label(
            input_folder=root,
            extension=".jpg",
            output_folder=DATASET_DIR_PATH
        )



