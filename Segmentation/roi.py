"""
Funkcje do wycinania ROI (Region of Interest) z zsegmentowanych obrazów kotów.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def create_binary_mask(image: np.ndarray) -> np.ndarray:
    """
    Tworzy maskę binarną z zsegmentowanego obrazu.
    Zakłada, że tło jest białe (255), a kot to wszystko inne.
    
    Args:
        image: Obraz BGR (z OpenCV)
    
    Returns:
        Binary mask (255 dla kota, 0 dla tła)
    """
    # Konwertuj do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold - wszystko co nie jest czysto białe (250+) staje się białe (255)
    # Czysto białe tło staje się czarne (0)
    _, binary_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    return binary_mask


def find_roi_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Znajduje bounding box ROI na podstawie maski binarnej.
    
    Args:
        mask: Binary mask (grayscale)
    
    Returns:
        Tuple (x, y, w, h) lub None jeśli nie znaleziono konturów
    """
    # Znajdź kontury w masce binarnej
    # cv2.RETR_EXTERNAL - tylko zewnętrzne kontury
    # cv2.CHAIN_APPROX_SIMPLE - kompresuje segmenty
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Znajdź największy kontur (powinien być kot)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Oblicz bounding box dla największego konturu
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, w, h)


def extract_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Wycina ROI z obrazu na podstawie bounding box.
    
    Args:
        image: Obraz BGR
        bbox: Tuple (x, y, w, h)
    
    Returns:
        Wycięty obraz
    """
    x, y, w, h = bbox
    
    # Ogranicz współrzędne do rozmiaru obrazu
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # Wycinanie ROI
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def process_roi(segmented_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Główna funkcja do przetwarzania ROI z zsegmentowanego obrazu.
    
    Args:
        segmented_image: Zsegmentowany obraz RGB (kot na białym tle) - z segment_cat
    
    Returns:
        Wycięty ROI (obraz kota w RGB) lub None jeśli nie znaleziono ROI
    """
    try:
        # Konwertuj RGB do BGR tylko dla operacji OpenCV (maskowanie)
        # segment_cat zwraca RGB, więc konwertujmy do BGR dla cv2 operacji
        segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR) if len(segmented_image.shape) == 3 else segmented_image
        
        # 1. Utwórz maskę binarną (używa BGR)
        binary_mask = create_binary_mask(segmented_bgr)
        
        # 2. Znajdź bounding box ROI
        bbox = find_roi_bbox(binary_mask)
        
        if bbox is None:
            print("⚠️ Nie znaleziono ROI w obrazie")
            return None
        
        # 3. Wyciągnij ROI z oryginalnego RGB obrazu (aby zachować oryginalne kolory)
        roi_image = extract_roi(segmented_image, bbox)
        
        if roi_image.size == 0:
            print("⚠️ Wycięty ROI jest pusty")
            return None
        
        print(f"✅ ROI wycięty pomyślnie: bbox={bbox}, rozmiar={roi_image.shape}")
        return roi_image
        
    except Exception as e:
        print(f"❌ Błąd podczas przetwarzania ROI: {e}")
        import traceback
        traceback.print_exc()
        return None


# Funkcja pomocnicza do wizualizacji (opcjonalnie)
def visualize_roi_process(segmented_image: np.ndarray, roi_image: np.ndarray, 
                          bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Tworzy wizualizację procesu ROI (dla debugowania).
    
    Args:
        segmented_image: Oryginalny zsegmentowany obraz
        roi_image: Wycięty ROI
        bbox: Bounding box (opcjonalnie)
    
    Returns:
        Obraz z wizualizacją
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    
    # 1. Oryginalny zsegmentowany obraz
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Zsegmentowany obraz")
    ax1.axis("off")
    
    # 2. Maska binarna
    ax2 = fig.add_subplot(gs[1])
    mask = create_binary_mask(segmented_image)
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("2. Maska binarna")
    ax2.axis("off")
    
    # 3. Obraz z bounding box
    if bbox:
        ax3 = fig.add_subplot(gs[2])
        img_with_bbox = cv2.cvtColor(segmented_image.copy(), cv2.COLOR_BGR2RGB)
        x, y, w, h = bbox
        cv2.rectangle(img_with_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ax3.imshow(img_with_bbox)
        ax3.set_title("3. Z bounding box")
        ax3.axis("off")
    
    # 4. Wycięty ROI
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    ax4.set_title("4. Wycięty ROI")
    ax4.axis("off")
    
    plt.tight_layout()
    return fig
