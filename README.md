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
