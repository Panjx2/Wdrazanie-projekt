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
