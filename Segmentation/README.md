# Automatyczna Segmentacja Kotów - YOLO + SAM

Automatyczna segmentacja kotów z tła używając **YOLO** (You Only Look Once) + **SAM (Segment Anything Model)**.

## 🎯 Funkcjonalność

- ✅ **Zero klikania** - automatyczna detekcja kotów przez YOLO
- ✅ **Automatyczna segmentacja** - SAM tworzy precyzyjne maski
- ✅ **Przetwarzanie hurtowe** - wszystkie obrazy z katalogu `data/`
- ✅ **Wielokrotne koty** - automatycznie znajduje i segmentuje wszystkie koty na obrazie
- ✅ **Szybkie i dokładne** - YOLO jest szybszy niż GroundingDINO

## 📋 Wymagania

- Python 3.8+
- CUDA (opcjonalnie, ale zalecane dla szybkości)
- ~3GB miejsca na dysku (dla modeli)

## 🚀 Instalacja

### Prosta instalacja

1. **Zainstaluj zależności:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To wszystko! YOLO automatycznie pobierze model przy pierwszym uruchomieniu.

### Modele

**Modele zostaną pobrane automatycznie** przy pierwszym uruchomieniu:
   - YOLO model (`yolov8m.pt`) - pobierze automatycznie przez ultralytics (~50MB)
   - SAM checkpoint (`sam_vit_h_4b8939.pth`) - ~2.4GB

   Jeśli automatyczne pobieranie SAM nie działa, pobierz ręcznie:
   - SAM: https://github.com/facebookresearch/segment-anything
   - Umieść `sam_vit_h_4b8939.pth` w głównym katalogu projektu

## 💻 Użycie

### Test na pojedynczym obrazie (zalecane na start)

Najpierw przetestuj na jednym obrazie:

```bash
python test_single_image.py data/Abyssinian/1.jpg
```

Wyniki testowe zapiszą się w katalogu `test_output/`.

### Przetwarzanie wszystkich obrazów

```bash
python segment_cats.py
```

Skrypt automatycznie:
1. Załaduje modele (lub pobierze je jeśli nie istnieją)
2. Przetworzy wszystkie obrazy z katalogu `data/`
3. Zapisze wyniki w katalogu `output/`

### Struktura wyników

```
output/
├── masks/              # Maski segmentacji (PNG)
│   ├── Abyssinian/
│   ├── Bengal/
│   └── ...
├── segmented/          # Koty na białym tle (JPG)
│   ├── Abyssinian/
│   ├── Bengal/
│   └── ...
└── cropped/           # Wycięte koty (czarne tło) (JPG)
    ├── Abyssinian/
    ├── Bengal/
    └── ...
```

## ⚙️ Konfiguracja

Możesz zmienić parametry w pliku `segment_cats.py`:

```python
YOLO_MODEL_NAME = "yolov8m.pt"      # Model YOLO (yolov8n.pt=szybszy, yolov8l.pt=dokładniejszy)
CONFIDENCE_THRESHOLD = 0.25         # Próg pewności dla detekcji (0.0-1.0)
```

### Wybór modelu YOLO

- `yolov8n.pt` - Nano (najszybszy, mniej dokładny)
- `yolov8s.pt` - Small (szybki, dobry kompromis)
- `yolov8m.pt` - Medium (domyślny, dobry balans)
- `yolov8l.pt` - Large (dokładniejszy, wolniejszy)
- `yolov8x.pt` - Extra Large (najdokładniejszy, najwolniejszy)

### Dostosowanie progu pewności

- **Zwiększ** `CONFIDENCE_THRESHOLD` (np. 0.5) jeśli znajduje zbyt wiele fałszywych detekcji
- **Zmniejsz** `CONFIDENCE_THRESHOLD` (np. 0.15) jeśli pomija koty

## 🔧 Rozwiązywanie problemów

### Błąd: "Nie znaleziono checkpointu SAM"

Pobierz model SAM ręcznie:
1. Pobierz `sam_vit_h_4b8939.pth` z https://github.com/facebookresearch/segment-anything
2. Umieść w głównym katalogu projektu

Alternatywnie, możesz użyć mniejszego modelu SAM:
- `sam_vit_l_0b3195.pth` - Large (szybszy)
- `sam_vit_b_01ec64.pth` - Base (najszybszy)

Zmień w `segment_cats.py`:
```python
SAM_CHECKPOINT_PATH = "sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"
```

### Błąd: YOLO nie znajduje kotów

1. **Sprawdź próg pewności** - może być za wysoki, spróbuj zmniejszyć `CONFIDENCE_THRESHOLD`
2. **Sprawdź czy obraz zawiera koty** - YOLO rozpoznaje tylko koty (klasa 15 w COCO)
3. **Spróbuj innego modelu YOLO** - `yolov8l.pt` jest dokładniejszy

### Wolne przetwarzanie

- Użyj GPU (CUDA) dla znacznie szybszego przetwarzania
- Użyj mniejszego modelu YOLO (`yolov8n.pt` lub `yolov8s.pt`)
- Użyj mniejszego modelu SAM (`sam_vit_b_01ec64.pth`)

### Błąd importu ultralytics

```bash
pip install --upgrade ultralytics
```

## 📊 Przykładowe wyniki

Dla każdego obrazu otrzymujesz:
1. **Maskę** - binarna maska segmentacji (PNG)
2. **Segmentowany obraz** - kot na białym tle (JPG)
3. **Wycięty kot** - kot z czarnym tłem (JPG)

## 🎨 Jak to działa?

1. **YOLO** wykrywa wszystkie koty na obrazie (używa klasy 15 z datasetu COCO)
2. **SAM** tworzy precyzyjne maski segmentacji dla każdego znalezionego kota
3. Maski są łączone i zapisywane w różnych formatach

## 📝 Licencja

Używa modeli:
- YOLO (AGPL-3.0)
- SAM (Apache 2.0)

## 🤝 Wsparcie

W razie problemów sprawdź:
- Czy wszystkie zależności są zainstalowane (`pip install -r requirements.txt`)
- Czy modele zostały pobrane (YOLO pobierze automatycznie, SAM trzeba pobrać ręcznie)
- Czy obrazy są w katalogu `data/` w odpowiedniej strukturze
- Czy masz wystarczająco miejsca na dysku (~3GB dla modeli)
