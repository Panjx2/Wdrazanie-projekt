# 🐱 Cat Segmenter - YOLOv8-seg Implementation

Implementacja segmentacji kota z użyciem **YOLOv8-seg** w React Native. Model zwraca precyzyjną maskę piksel po pikselu, która pozwala na wycięcie kota z przezroczystym tłem.

## ✨ Funkcje

- ✅ **Segmentacja piksel po pikselu** - precyzyjna maska dla każdego piksela
- ✅ **Działa na każdą rasę kota** - uniwersalny model
- ✅ **Działa w każdym świetle i tle** - nie wymaga ręcznego dostrajania
- ✅ **Wycinanie kota** - automatyczne tworzenie przezroczystego tła
- ✅ **Wizualizacja maski** - podgląd segmentacji
- ✅ **ONNX Runtime** - szybka inferencja na urządzeniu mobilnym

## 📁 Struktura projektu

```
my-cat-classifier/
├── App.tsx                          # Główny komponent z UI i logiką YOLO
├── src/
│   └── utils/
│       ├── yoloPreprocess.ts        # Preprocessing YOLO (letterbox resize)
│       ├── yoloPostprocess.ts       # Postprocessing (NMS + mask extraction)
│       └── maskApplication.ts       # Aplikowanie maski (wycinanie kota)
├── assets/
│   └── models/
│       └── yolov8n-seg.onnx        # Model YOLOv8-seg ONNX (dodaj własny)
└── YOLO_EXPORT_INSTRUCTIONS.md      # Instrukcje eksportu modelu
```

## 🚀 Szybki start

### 1. Zainstaluj zależności

```bash
npm install
```

### 2. Dodaj model YOLOv8-seg ONNX

Umieść model w:
```
assets/models/yolov8n-seg.onnx
```

**Jak uzyskać model?** Zobacz [YOLO_EXPORT_INSTRUCTIONS.md](./YOLO_EXPORT_INSTRUCTIONS.md)

### 3. Uruchom aplikację

```bash
npm start
# lub
npm run android
npm run ios
```

## 🔧 Konfiguracja

### Parametry YOLO (w `App.tsx`)

```typescript
const YOLO_INPUT_SIZE = 640;      // Rozmiar wejściowy (standard YOLOv8)
const CONF_THRESHOLD = 0.25;      // Próg pewności detekcji
const IOU_THRESHOLD = 0.45;       // Próg IoU dla NMS
const MASK_THRESHOLD = 0.5;       // Próg dla maski segmentacyjnej
const CAT_CLASS_ID = 15;          // ID klasy "cat" w COCO (zmień jeśli używasz własnego modelu)
```

### Zmiana ścieżki modelu

W `App.tsx`, linia 48:
```typescript
require('./assets/models/yolov8n-seg.onnx') // Zmień na swój model
```

## 📖 Jak to działa?

### 1. Preprocessing
- **Letterbox resize** - zachowuje proporcje obrazu, dodaje padding
- **Normalizacja [0,1]** - konwersja RGB do zakresu [0,1]
- **Resize do 640×640** - standardowy rozmiar wejściowy YOLOv8

### 2. Inference
- Model YOLOv8-seg zwraca:
  - **Detections**: [boxes, scores, mask_coefficients]
  - **Prototypes**: [32, mask_h, mask_w] - prototypy masek

### 3. Postprocessing
- **NMS** (Non-Maximum Suppression) - usuwa duplikaty detekcji
- **Mask generation** - oblicza maskę: `sigmoid(coeffs @ protos)`
- **Mask scaling** - skaluje maskę do rozmiaru bounding boxa
- **Coordinate transformation** - konwertuje współrzędne z powrotem do oryginalnego rozmiaru

### 4. Mask Application
- Aplikuje maskę do oryginalnego obrazu
- Tworzy przezroczyste tło (PNG z alpha channel)
- Zapisuje wynik jako PNG

## 🎯 Przykładowe użycie

```typescript
// W App.tsx - automatycznie wywoływane po wyborze zdjęcia
await segmentImage(imageUri);

// Wewnętrznie:
// 1. Preprocessing
const { tensor, metadata } = yoloPreprocess(imageBase64, 640);

// 2. Inference
const outputs = await session.run({ [inputName]: inputTensor });

// 3. Postprocessing
const result = yoloPostprocess(outputs, metadata, 0.25, 0.45, 0.5);

// 4. Aplikowanie maski
const segmentedUri = await applyMask(imageUri, mask, box, true);
```

## 🐛 Troubleshooting

### Model nie ładuje się

**Problem:** `Cannot load model YOLOv8-seg`

**Rozwiązanie:**
1. Sprawdź czy plik istnieje w `assets/models/yolov8n-seg.onnx`
2. Upewnij się, że model jest w formacie ONNX (nie .pt)
3. Jeśli model >2GB, potrzebujesz pliku `.onnx.data` (external data)

### Brak detekcji kota

**Problem:** `Nie znaleziono kota na zdjęciu`

**Rozwiązanie:**
1. Sprawdź `CAT_CLASS_ID` - ustaw odpowiednią klasę dla swojego modelu
2. Zmniejsz `CONF_THRESHOLD` (np. 0.15) dla słabszych detekcji
3. Upewnij się, że model był trenowany na podobnych obrazach

### Maska jest nieprecyzyjna

**Problem:** Maska nie pasuje do kota

**Rozwiązanie:**
1. Dostosuj `MASK_THRESHOLD` (0.3-0.7)
2. Użyj większego modelu (yolov8m-seg lub większy)
3. Sprawdź czy preprocessing zachowuje proporcje (letterbox)

### Błąd podczas aplikowania maski

**Problem:** `Error applying mask`

**Rozwiązanie:**
1. Sprawdź czy bounding box jest w granicach obrazu
2. Sprawdź czy maska ma poprawny rozmiar
3. Sprawdź logi w konsoli dla szczegółów błędu

## 📚 Dokumentacja

- [YOLO Export Instructions](./YOLO_EXPORT_INSTRUCTIONS.md) - jak wyeksportować model do ONNX
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/) - dokumentacja YOLOv8
- [ONNX Runtime React Native](https://onnxruntime.ai/docs/tutorials/mobile/) - dokumentacja ONNX Runtime

## 🔄 Migracja z poprzedniej wersji

Jeśli używałeś poprzedniej wersji z `detectROI`:

1. **Usuń** `src/utils/roiDetection.ts` (jeśli istnieje)
2. **Zaktualizuj** `App.tsx` - używa teraz YOLOv8-seg
3. **Dodaj model** YOLOv8-seg ONNX do `assets/models/`
4. **Zaktualizuj** `CAT_CLASS_ID` jeśli używasz własnego modelu

## 📝 Notatki

- Model YOLOv8-seg jest **uniwersalny** - działa na każdą rasę kota
- Nie wymaga **ręcznego dostrajania** progów, morfologii ani flood-fill
- **Precyzyjna maska piksel po pikselu** - lepsza niż metody oparte na progach
- Działa w **każdym świetle i tle** - model uczy się z danych treningowych

## 🎉 Gotowe!

Teraz masz działającą implementację YOLOv8-seg dla segmentacji kota w React Native. Model zwraca precyzyjną maskę, którą możesz użyć do wycięcia kota z przezroczystym tłem.

