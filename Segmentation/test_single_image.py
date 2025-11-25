"""
Test segmentacji na pojedynczym obrazie.
Użyj tego skryptu do przetestowania segmentacji przed przetwarzaniem wszystkich obrazów.
"""

import sys
from pathlib import Path
from segment_cats import load_models, segment_cat
import cv2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python test_single_image.py <ścieżka_do_obrazu>")
        print("Przykład: python test_single_image.py data/Abyssinian/Abyssinian_1.jpg")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Błąd: Obraz nie istnieje: {image_path}")
        sys.exit(1)
    
    print(f"Testowanie segmentacji na: {image_path}")
    print("=" * 50)
    
    # Załaduj modele
    yolo_model, sam_predictor = load_models()
    
    # Segmentacja
    print(f"\nPrzetwarzanie...")
    mask, segmented, cropped = segment_cat(
        str(image_path),
        yolo_model,
        sam_predictor
    )
    
    if mask is None:
        print("✗ Nie znaleziono kotów w obrazie")
        sys.exit(1)
    
    # Zapisz wyniki testowe
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "test_mask.png"), mask)
    cv2.imwrite(str(output_dir / "test_segmented.jpg"), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "test_cropped.jpg"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Sukces!")
    print(f"Wyniki zapisane w: {output_dir}/")
    print(f"  - test_mask.png - maska segmentacji")
    print(f"  - test_segmented.jpg - kot na białym tle")
    print(f"  - test_cropped.jpg - wycięty kot")

