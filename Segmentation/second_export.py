from ultralytics import YOLO

model2 = YOLO("yolov8n.pt")       # baza YOLO
model2.export(format="onnx")