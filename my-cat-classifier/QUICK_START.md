# 🚀 Szybki start - YOLOv8-seg Cat Segmenter

## Problem: "Unable to resolve ./assets/models/yolov8n-seg.onnx"

Ten błąd oznacza, że model YOLOv8-seg nie został jeszcze dodany do projektu.

## ✅ Rozwiązanie (3 kroki)

### Krok 1: Zainstaluj zależności Python

```bash
pip install ultralytics onnx
```

### Krok 2: Pobierz i wyeksportuj model

**Opcja A: Automatycznie (zalecane)**
```bash
npm run download-yolo-model
```

**Opcja B: Ręcznie**
```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

Następnie skopiuj wygenerowany plik `yolov8n-seg.onnx` do:
```
assets/models/yolov8n-seg.onnx
```

### Krok 3: Uruchom aplikację

```bash
npm start
```

## 📝 Alternatywnie: Użyj własnego modelu

Jeśli masz własny wytrenowany model YOLOv8-seg:

1. Wyeksportuj do ONNX:
   ```python
   from ultralytics import YOLO
   model = YOLO('path/to/your/best.pt')
   model.export(format='onnx', imgsz=640, simplify=True)
   ```

2. Skopiuj plik `.onnx` do `assets/models/yolov8n-seg.onnx`

3. Jeśli używasz innej nazwy, zaktualizuj `App.tsx` (linia 56):
   ```typescript
   require('./assets/models/twoja-nazwa.onnx')
   ```

## 🔍 Weryfikacja

Po dodaniu modelu, sprawdź czy plik istnieje:
```bash
ls assets/models/yolov8n-seg.onnx
```

Powinien istnieć plik o rozmiarze ~6-22 MB (zależy od wersji modelu).

## 📚 Więcej informacji

- [YOLO_EXPORT_INSTRUCTIONS.md](./YOLO_EXPORT_INSTRUCTIONS.md) - szczegółowe instrukcje
- [README_YOLO_SEG.md](./README_YOLO_SEG.md) - pełna dokumentacja
- [assets/models/README_MODEL.md](./assets/models/README_MODEL.md) - informacje o modelu

## ❓ Problemy?

### Błąd: "ultralytics not found"
```bash
pip install ultralytics onnx
```

### Błąd: "Model nie ładuje się w aplikacji"
- Sprawdź czy plik istnieje w `assets/models/yolov8n-seg.onnx`
- Sprawdź czy model jest w formacie ONNX (nie .pt)
- Uruchom ponownie: `npm start`

### Błąd: "Nie znaleziono kota"
- Sprawdź `CAT_CLASS_ID` w `App.tsx` (domyślnie 15 dla COCO)
- Zmniejsz `CONF_THRESHOLD` (domyślnie 0.25)

