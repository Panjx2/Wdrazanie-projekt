# Instrukcje: Eksport YOLOv8-seg do ONNX

## Wymagania

- Python 3.8+
- ultralytics (YOLOv8)
- onnx

## Instalacja

```bash
pip install ultralytics onnx
```

## Eksport modelu YOLOv8-seg do ONNX

### Opcja 1: Użyj pre-trenowanego modelu YOLOv8-seg

```python
from ultralytics import YOLO

# Załaduj pre-trenowany model YOLOv8-seg
model = YOLO('yolov8n-seg.pt')  # nano (najmniejszy, najszybszy)
# model = YOLO('yolov8s-seg.pt')  # small
# model = YOLO('yolov8m-seg.pt')  # medium
# model = YOLO('yolov8l-seg.pt')  # large
# model = YOLO('yolov8x-seg.pt')  # xlarge (największy, najdokładniejszy)

# Eksportuj do ONNX
model.export(format='onnx', imgsz=640, simplify=True)
```

### Opcja 2: Użyj własnego wytrenowanego modelu

```python
from ultralytics import YOLO

# Załaduj swój wytrenowany model
model = YOLO('path/to/your/best.pt')

# Eksportuj do ONNX
model.export(format='onnx', imgsz=640, simplify=True)
```

### Opcja 3: Trening własnego modelu na kocie

```python
from ultralytics import YOLO

# Załaduj pre-trenowany model
model = YOLO('yolov8n-seg.pt')

# Fine-tune na swoim datasetcie z kotami
model.train(
    data='path/to/your/dataset.yaml',  # YOLO format dataset
    epochs=100,
    imgsz=640,
    batch=16,
)

# Eksportuj do ONNX
model.export(format='onnx', imgsz=640, simplify=True)
```

## Format datasetu YOLO

Twój dataset powinien być w formacie YOLO:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Plik `dataset.yaml`:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: cat
```

## Umieść model w projekcie

Po eksporcie, skopiuj wygenerowany plik `.onnx` do:

```
assets/models/yolov8n-seg.onnx
```

**UWAGA:** Zmień nazwę pliku w `App.tsx` (linia 48) jeśli używasz innej nazwy modelu.

## Weryfikacja modelu

Model ONNX powinien mieć:
- **Input:** `[1, 3, 640, 640]` (RGB, float32, znormalizowane [0,1])
- **Output 1:** `[1, num_detections, 4+num_classes+32]` - detections (boxes + scores + mask coefficients)
- **Output 2:** `[1, 32, mask_h, mask_w]` - prototypy masek

## Klasa "cat" w COCO

W standardowym datasetcie COCO:
- Klasa "cat" ma ID = **15** (0-indexed)

Jeśli używasz własnego modelu z jedną klasą (tylko kot), ustaw `CAT_CLASS_ID = 0` w `App.tsx`.

## Rozmiar modelu

- **yolov8n-seg**: ~6 MB (najszybszy, mniej dokładny)
- **yolov8s-seg**: ~22 MB
- **yolov8m-seg**: ~52 MB
- **yolov8l-seg**: ~87 MB
- **yolov8x-seg**: ~136 MB (najwolniejszy, najbardziej dokładny)

Dla React Native zalecamy **yolov8n-seg** lub **yolov8s-seg** dla najlepszej wydajności.

## Troubleshooting

### Problem: Model nie ładuje się w React Native

**Rozwiązanie:**
1. Upewnij się, że model jest w formacie ONNX (nie .pt)
2. Sprawdź czy model ma external data file (jeśli >2GB) - wtedy potrzebujesz `.onnx.data`
3. Użyj `simplify=True` podczas eksportu

### Problem: Błędne detekcje

**Rozwiązanie:**
1. Sprawdź `CAT_CLASS_ID` w `App.tsx` - ustaw odpowiednią klasę
2. Dostosuj `CONF_THRESHOLD` (domyślnie 0.25)
3. Dostosuj `IOU_THRESHOLD` dla NMS (domyślnie 0.45)

### Problem: Maska jest nieprecyzyjna

**Rozwiązanie:**
1. Dostosuj `MASK_THRESHOLD` (domyślnie 0.5)
2. Użyj większego modelu (yolov8m-seg lub większy)
3. Upewnij się, że model był trenowany na podobnych obrazach

