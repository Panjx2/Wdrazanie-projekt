# Model YOLOv8-seg ONNX

## ⚠️ Wymagany plik

Aby aplikacja działała, musisz dodać model YOLOv8-seg w formacie ONNX:

**Plik:** `yolov8n-seg.onnx`

## 📥 Jak uzyskać model?

### Opcja 1: Pobierz pre-trenowany model

```python
from ultralytics import YOLO

# Pobierz i wyeksportuj model
model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

Plik zostanie wygenerowany jako `yolov8n-seg.onnx` - skopiuj go tutaj.

### Opcja 2: Użyj własnego modelu

Jeśli masz własny wytrenowany model:

```python
from ultralytics import YOLO

model = YOLO('path/to/your/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

## 📍 Lokalizacja

Umieść plik tutaj:
```
assets/models/yolov8n-seg.onnx
```

## 🔧 Zmiana nazwy modelu

Jeśli używasz innej nazwy modelu, zaktualizuj `App.tsx` (linia 52):

```typescript
require('./assets/models/twoja-nazwa.onnx')
```

## 📚 Więcej informacji

Zobacz [YOLO_EXPORT_INSTRUCTIONS.md](../../YOLO_EXPORT_INSTRUCTIONS.md) dla szczegółowych instrukcji.

