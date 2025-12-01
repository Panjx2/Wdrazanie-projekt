# YOLO11 React Native demo

1. Wyeksportuj swój checkpoint YOLO11 do ONNX (z wbudowanym NMS), np.:
   ```bash
   cd my-cat-classifier
   python scripts/export_yolo11_to_onnx.py assets/models/yolo11s.pt --imgsz 640
   ```
   W katalogu `assets/models/` pojawi się `yolo11.onnx` (i ewentualnie `yolo11.onnx.data` dla modeli >2GB).
   Repo zawiera placeholdery `yolo11.onnx` i `yolo11.onnx.data` tylko po to, by Metro się
   budował — zamień je na swój wyeksportowany model (oraz plik `.data`, jeśli powstanie).

2. Zbuduj aplikację Expo:
   ```bash
   cd my-cat-classifier
   npm install
   npx expo prebuild --clean
   npx expo run:android
   ```

Aplikacja skaluje klatki kamery do 640 px szerokości, wykonuje letterbox w locie,
ładuje model ONNX przez onnxruntime-react-native i wyświetla listę wykrytych obiektów.

> Uwaga: aplikacja odrzuca modele YOLO, których liczba klas nie zgadza się z `assets/labels.json` —
> walidacja podczas ładowania przerwie działanie z jasnym komunikatem, aby uniknąć cichych
> rozbieżności etykiet.

## Dlaczego widzisz komunikaty o niespójności klas (proste wyjaśnienie)

- W repozytorium są **dwa różne modele**: YOLO11 do wykrywania kota w kadrze oraz osobny klasyfikator MobileNetV3 rozpoznający 12 ras + „Not cat”. YOLO jest pobierany w oryginalnej wersji COCO (`yolo11s.pt`), która ma **80 klas COCO** (m.in. pies, rower, kanapa).【F:model_v3_tests-Copy1.ipynb†L115-L174】
- Eksport do ONNX zachowuje ten oryginalny układ COCO, dlatego w detekcjach pojawiają się identyfikatory klas np. 15, 16 czy 63, które **nie istnieją w `assets/labels.json`** (tam jest tylko 13 wpisów).【F:my-cat-classifier/assets/labels.json†L1-L15】
- Aplikacja jest celowo zaprogramowana tak, aby przerywać działanie, gdy liczba lub numery klas z modelu **nie pokrywają się dokładnie** z listą w `labels.json`. Dzięki temu nie zobaczysz „szczątkowych” etykiet ani pomieszanych wyników.
- Aby pozbyć się błędu, masz dwie drogi:
  1. **Przetrenować i wyeksportować YOLO z Twoimi 13 etykietami** w identycznej kolejności jak w `labels.json`, a następnie podmienić plik `assets/models/yolo11.onnx` w aplikacji.
  2. **Zostawić YOLO w wersji COCO i nie używać go do klas ras**, lecz wyłącznie do wycinania kota (tak jak w notebooku `model_v3_tests-Copy1.ipynb`). Wtedy wynik klasyfikacji nadal pochodzi z MobileNetV3, a etykiety ras pozostają spójne.

### Jak korzystać z opcji 2 (YOLO COCO tylko do wycinania kota)

- Aplikacja ładuje teraz dwa modele ONNX: `assets/models/yolo11.onnx` (COCO) do wykrycia kota i `assets/models/mobilenetv3_finetuned.onnx` do rozpoznania rasy.
- YOLO filtruje wyłącznie klasę kota COCO (`class_id = 15`), wybiera największy box i wycina kadr; jeśli YOLO nie znajdzie kota, klasyfikator dostaje pełny kadr.
- Kadrowanie jest wykonywane w aplikacji (Expo ImageManipulator), a do klasyfikatora trafia kwadrat 224×224 z normalizacją ImageNet, więc kolejność etykiet zawsze pochodzi z `labels.json`.
- Dzięki temu nie ma konfliktu liczby klas: YOLO pozostaje w układzie 80 COCO, a ostateczna etykieta zawsze pochodzi z MobileNetV3 (13 klas z `labels.json`).

## Utrzymanie kolejności klas i ponowny eksport modelu

- Kolejność etykiet jest definiowana wyłącznie w `my-cat-classifier/assets/labels.json`. Podczas eksportu skrypt `scripts/export_to_onnx.py` buduje głowicę modelu na bazie tej listy, więc zachowuje dokładnie ten sam układ klas co w pliku z etykietami.【F:my-cat-classifier/scripts/export_to_onnx.py†L2-L106】【F:my-cat-classifier/assets/labels.json†L1-L15】
- Aby ponownie wyeksportować model z tą samą kolejnością, upewnij się, że `labels.json` jest poprawny, a następnie uruchom:
  ```bash
  cd my-cat-classifier
  python scripts/export_to_onnx.py path/do/twojego_checkpointu.pth
  ```
  Skrypt domyślnie wyszukuje `assets/models/mobilenetv3_finetuned.pth`, ale możesz wskazać inny checkpoint jako argument. Powstały plik ONNX trafi do `assets/models/` z taką samą kolejnością klas jak w `labels.json`.
- Jeśli chcesz zmienić zestaw etykiet, najpierw zaktualizuj `labels.json` w dokładnej kolejności klas użytej podczas trenowania, a następnie wyeksportuj nowy model tą samą komendą. Nowa kolejność zacznie obowiązywać w modelu i aplikacji dopiero po wygenerowaniu nowego checkpointu i jego eksporcie.
- Notebook używany do tworzenia pierwotnego modelu znajduje się w repozytorium jako `model_creation.ipynb` (w katalogu głównym).【F:model_creation.ipynb†L1-L1】
