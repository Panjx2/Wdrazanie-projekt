"""
Automatyczna segmentacja kotów z tła używając YOLO + SAM.
YOLO automatycznie znajduje wszystkie koty, SAM tworzy precyzyjne maski.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, Optional

# Konfiguracja
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_MASKS_DIR = OUTPUT_DIR / "masks"
OUTPUT_SEGMENTED_DIR = OUTPUT_DIR / "segm0,ented"
OUTPUT_CROPPED_DIR = OUTPUT_DIR / "cropped"

# Tworzenie katalogów wyjściowych
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_MASKS_DIR.mkdir(exist_ok=True)
OUTPUT_SEGMENTED_DIR.mkdir(exist_ok=True)
OUTPUT_CROPPED_DIR.mkdir(exist_ok=True)

# Ścieżki do modeli
YOLO_MODEL_NAME = "yolov8m.pt"  # Możesz zmienić na yolov8n.pt (szybszy) lub yolov8l.pt (dokładniejszy)
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# Klasa "cat" w COCO dataset (używana przez YOLO)
CAT_CLASS_ID = 15

# Próg pewności dla detekcji YOLO (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.25


def download_sam_model():
    """Pobiera model SAM jeśli nie jest dostępny."""
    import urllib.request
    
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print("Pobieranie SAM checkpoint (może to chwilę potrwać, ~2.4GB)...")
        try:
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                SAM_CHECKPOINT_PATH
            )
            print("✓ SAM checkpoint pobrany")
        except Exception as e:
            print(f"✗ Błąd pobierania SAM: {e}")
            print("Możesz pobrać ręcznie z: https://github.com/facebookresearch/segment-anything")


def load_models():
    """Ładuje modele YOLO i SAM."""
    print("Ładowanie modeli...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Używane urządzenie: {device}")
    
    # Załaduj YOLO (automatycznie pobierze model jeśli nie istnieje)
    print(f"Ładowanie YOLO ({YOLO_MODEL_NAME})...")
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print("✓ YOLO załadowany")
    
    # Pobierz SAM jeśli nie istnieje
    download_sam_model()
    
    # Załaduj SAM
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Nie znaleziono checkpointu SAM: {SAM_CHECKPOINT_PATH}\n"
            "Pobierz ręcznie z: https://github.com/facebookresearch/segment-anything"
        )
    
    print("Ładowanie SAM...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("✓ SAM załadowany")
    
    print("✓ Wszystkie modele załadowane!")
    return yolo_model, sam_predictor


def segment_cat(image_path: str, yolo_model, sam_predictor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segmentuje kota z obrazu używając YOLO + SAM.
    
    Args:
        image_path: Ścieżka do obrazu
        yolo_model: Załadowany model YOLO
        sam_predictor: Załadowany predictor SAM
    
    Returns:
        mask: Maska segmentacji (numpy array) lub None
        segmented_image: Obraz z segmentacją (kot na białym tle) lub None
        cropped_cat: Wycięty kot (bez tła) lub None
    """
    # Załaduj obraz
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Błąd: Nie można załadować obrazu {image_path}")
        return None, None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO - znajdź koty
    results = yolo_model(image_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    cat_boxes = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == CAT_CLASS_ID:  # Klasa "cat" w COCO
                # YOLO zwraca boxy w formacie xyxy
                cat_boxes.append(box.xyxy[0].cpu().numpy())
    
    if len(cat_boxes) == 0:
        return None, None, None
    
    # SAM - segmentacja
    sam_predictor.set_image(image_rgb)
    
    # Dla każdego znalezionego kota, stwórz maskę
    all_masks = []
    for box in cat_boxes:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        all_masks.append(masks[0])  # Najlepsza maska
    
    # Połącz wszystkie maski kotów
    if len(all_masks) > 1:
        combined_mask = np.logical_or.reduce(all_masks)
    else:
        combined_mask = all_masks[0]
    
    # Konwertuj maskę do uint8
    mask = (combined_mask * 255).astype(np.uint8)
    
    # Stwórz obraz z segmentacją (kot na białym tle)
    segmented_image = image_rgb.copy()
    segmented_image[~combined_mask] = [255, 255, 255]  # Białe tło
    
    # Wycięty kot (tylko kot, bez tła - przezroczyste tło jako czarne)
    cropped_cat = image_rgb.copy()
    cropped_cat[~combined_mask] = [0, 0, 0]  # Czarne tło
    
    return mask, segmented_image, cropped_cat


def process_all_images():
    """Przetwarza wszystkie obrazy z katalogu data."""
    # Załaduj modele
    yolo_model, sam_predictor = load_models()
    
    # Znajdź wszystkie obrazy
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    
    for breed_dir in DATA_DIR.iterdir():
        if breed_dir.is_dir():
            for img_file in breed_dir.iterdir():
                if img_file.suffix in image_extensions:
                    image_paths.append(img_file)
    
    print(f"\nZnaleziono {len(image_paths)} obrazów do przetworzenia")
    print(f"Model YOLO: {YOLO_MODEL_NAME}")
    print(f"Próg pewności: {CONFIDENCE_THRESHOLD}")
    print("=" * 50)
    
    processed = 0
    skipped = 0
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Przetwarzanie: {image_path.name}")
        
        try:
            # Segmentacja
            mask, segmented, cropped = segment_cat(
                str(image_path),
                yolo_model,
                sam_predictor
            )
            
            if mask is None:
                skipped += 1
                print(f"  ⚠ Nie znaleziono kotów")
                continue
            
            # Zapisz wyniki
            relative_path = image_path.relative_to(DATA_DIR)
            output_subdir_masks = OUTPUT_MASKS_DIR / relative_path.parent
            output_subdir_segmented = OUTPUT_SEGMENTED_DIR / relative_path.parent
            output_subdir_cropped = OUTPUT_CROPPED_DIR / relative_path.parent
            
            output_subdir_masks.mkdir(parents=True, exist_ok=True)
            output_subdir_segmented.mkdir(parents=True, exist_ok=True)
            output_subdir_cropped.mkdir(parents=True, exist_ok=True)
            
            # Zapisz maskę
            mask_path = output_subdir_masks / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            
            # Zapisz segmentowany obraz (kot na białym tle)
            segmented_path = output_subdir_segmented / f"{image_path.stem}_segmented.jpg"
            cv2.imwrite(str(segmented_path), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
            
            # Zapisz wyciętego kota
            cropped_path = output_subdir_cropped / f"{image_path.stem}_cropped.jpg"
            cv2.imwrite(str(cropped_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
            processed += 1
            print(f"  ✓ Zapisano: {mask_path.name}")
            
        except Exception as e:
            print(f"  ✗ Błąd: {e}")
            skipped += 1
            continue
    
    print("\n" + "=" * 50)
    print(f"Zakończono!")
    print(f"Przetworzono: {processed}")
    print(f"Pominięto: {skipped}")
    print(f"\nWyniki zapisane w:")
    print(f"  - Maski: {OUTPUT_MASKS_DIR}")
    print(f"  - Segmentowane (białe tło): {OUTPUT_SEGMENTED_DIR}")
    print(f"  - Wycięte koty: {OUTPUT_CROPPED_DIR}")


if __name__ == "__main__":
    process_all_images()
