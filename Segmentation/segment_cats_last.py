"""
Segmentacja kota z tła używając last.pt (wbudowany model do rozpoznawania kotów) + SAM.
Przyjmuje pojedynczy obraz jako argument i zapisuje zsegmentowany obraz.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, Optional

# Ścieżka do modelu last.pt (wbudowany do rozpoznawania kotów)
# Plik jest w katalogu Segmentation
YOLO_MODEL_PATH = Path(__file__).parent / "last.pt"
if not YOLO_MODEL_PATH.exists():
    # Alternatywna ścieżka
    YOLO_MODEL_PATH = Path(__file__).parent.parent / "Segmentation" / "last.pt"

SAM_CHECKPOINT_PATH = Path(__file__).parent / "sam_vit_h_4b8939.pth"
if not SAM_CHECKPOINT_PATH.exists():
    # Alternatywna ścieżka
    SAM_CHECKPOINT_PATH = Path(__file__).parent.parent / "Segmentation" / "sam_vit_h_4b8939.pth"

SAM_MODEL_TYPE = "vit_h"

# Klasa "cat" w last.pt (wbudowany model do kotów, więc klasa 0)
CAT_CLASS_ID = 0

# Próg pewności dla detekcji YOLO (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.25

# Katalog wyjściowy (w katalogu Segmentation)
OUTPUT_DIR = Path(__file__).parent
OUTPUT_SEGMENTED_DIR = OUTPUT_DIR / "segmented"
OUTPUT_SEGMENTED_DIR.mkdir(exist_ok=True)


def download_sam_model():
    """Pobiera model SAM jeśli nie jest dostępny."""
    import urllib.request
    
    if not SAM_CHECKPOINT_PATH.exists():
        print("Pobieranie SAM checkpoint (może to chwilę potrwać, ~2.4GB)...")
        try:
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                str(SAM_CHECKPOINT_PATH)
            )
            print("✓ SAM checkpoint pobrany")
        except Exception as e:
            print(f"✗ Błąd pobierania SAM: {e}")
            print("Możesz pobrać ręcznie z: https://github.com/facebookresearch/segment-anything")


def load_models():
    """Ładuje modele YOLO (last.pt) i SAM."""
    print("Ładowanie modeli...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Używane urządzenie: {device}")
    
    # Załaduj last.pt
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu last.pt: {YOLO_MODEL_PATH}\n"
            "Upewnij się, że plik last.pt znajduje się w katalogu Segmentation."
        )
    
    print(f"Ładowanie YOLO (last.pt)...")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    print("✓ YOLO (last.pt) załadowany")
    
    # Pobierz SAM jeśli nie istnieje
    download_sam_model()
    
    # Załaduj SAM
    if not SAM_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono checkpointu SAM: {SAM_CHECKPOINT_PATH}\n"
            "Pobierz ręcznie z: https://github.com/facebookresearch/segment-anything"
        )
    
    print("Ładowanie SAM...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("✓ SAM załadowany")
    
    print("✓ Wszystkie modele załadowane!")
    return yolo_model, sam_predictor


def segment_cat(image_path: str, yolo_model, sam_predictor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segmentuje kota z obrazu używając YOLO (last.pt) + SAM.
    
    Args:
        image_path: Ścieżka do obrazu
        yolo_model: Załadowany model YOLO (last.pt)
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
    
    # YOLO (last.pt) - znajdź koty
    results = yolo_model(image_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    cat_boxes = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == CAT_CLASS_ID:  # Klasa 0 w last.pt (kot)
                # YOLO zwraca boxy w formacie xyxy
                cat_boxes.append(box.xyxy[0].cpu().numpy())
    
    if len(cat_boxes) == 0:
        print("  ⚠ Nie znaleziono kotów")
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
    
    if len(all_masks) > 1:
        combined_mask = np.logical_or.reduce(all_masks)
    else:
        combined_mask = all_masks[0]
    
    mask = (combined_mask * 255).astype(np.uint8)
    
    #  obraz z segmentacją (kot na białym tle)
    segmented_image = image_rgb.copy()
    segmented_image[~combined_mask] = [255, 255, 255]  # Białe tło
    
    # Wycięty kot (tylko kot, bez tła - przezroczyste tło jako czarne)
    cropped_cat = image_rgb.copy()
    cropped_cat[~combined_mask] = [0, 0, 0]  # Czarne tło
    
    return mask, segmented_image, cropped_cat


def process_single_image(image_path: str, output_filename: str = None, output_dir: str = None):
    """
    Przetwarza pojedynczy obraz i zapisuje zsegmentowany wynik.
    
    Args:
        image_path: Ścieżka do obrazu wejściowego
        output_filename: Nazwa pliku wyjściowego (opcjonalnie)
        output_dir: Katalog wyjściowy (opcjonalnie, domyślnie OUTPUT_SEGMENTED_DIR)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Błąd: Obraz nie istnieje: {image_path}")
        return None
    
    # Załaduj modele
    yolo_model, sam_predictor = load_models()
    
    print(f"\nPrzetwarzanie: {image_path.name}")
    
    try:
        # Segmentacja
        mask, segmented, cropped = segment_cat(
            str(image_path),
            yolo_model,
            sam_predictor
        )
        
        if mask is None:
            print("  ⚠ Nie znaleziono kotów")
            return None
        
        # Określ katalog wyjściowy
        if output_dir:
            output_segmented_dir = Path(output_dir)
            output_segmented_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_segmented_dir = OUTPUT_SEGMENTED_DIR
        
        # Określ nazwę pliku wyjściowego
        if output_filename is None:
            output_filename = f"{image_path.stem}_segmented.jpg"
        
        output_path = output_segmented_dir / output_filename
        
        # Zapisz segmentowany obraz (kot na białym tle)
        cv2.imwrite(str(output_path), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
        
        print(f"  ✓ Zapisano: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"  ✗ Błąd: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python segment_cats_last.py <ścieżka_do_obrazu> [nazwa_wyjściowa] [katalog_wyjściowy]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = process_single_image(image_path, output_filename, output_dir)
    if result:
        print(f"\n✓ Segmentacja zakończona pomyślnie!")
        print(f"Wynik: {result}")
    else:
        print("\n✗ Segmentacja nie powiodła się")
        sys.exit(1)

