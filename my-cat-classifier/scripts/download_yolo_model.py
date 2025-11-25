#!/usr/bin/env python3
"""
Skrypt do automatycznego pobrania i eksportu modelu YOLOv8-seg do ONNX
"""

import os
import sys
from pathlib import Path

def download_and_export_yolo_model():
    """Pobiera i eksportuje YOLOv8-seg do ONNX"""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Błąd: ultralytics nie jest zainstalowany")
        print("\nZainstaluj:")
        print("  pip install ultralytics onnx")
        sys.exit(1)
    
    # Ścieżka do modelu
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "assets" / "models"
    model_path = model_dir / "yolov8n-seg.onnx"
    
    # Utwórz katalog jeśli nie istnieje
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Pobieranie modelu YOLOv8n-seg...")
    print("   (To może chwilę potrwać przy pierwszym uruchomieniu)")
    
    # Załaduj pre-trenowany model
    model = YOLO('yolov8n-seg.pt')
    
    print("🔄 Eksportowanie do ONNX...")
    
    # Eksportuj do ONNX
    model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        opset=12,  # ONNX opset version
    )
    
    # Znajdź wyeksportowany plik (zwykle w bieżącym katalogu)
    exported_file = Path("yolov8n-seg.onnx")
    if not exported_file.exists():
        # Sprawdź w katalogu roboczym
        exported_file = Path.cwd() / "yolov8n-seg.onnx"
    
    if not exported_file.exists():
        print("❌ Błąd: Nie znaleziono wyeksportowanego pliku")
        print("   Sprawdź czy eksport się powiódł")
        sys.exit(1)
    
    # Skopiuj do assets/models/
    import shutil
    shutil.copy2(exported_file, model_path)
    
    print(f"✅ Model zapisany w: {model_path}")
    print(f"   Rozmiar: {model_path.stat().st_size / (1024*1024):.2f} MB")
    print("\n🎉 Gotowe! Możesz teraz uruchomić aplikację:")
    print("   npm start")

if __name__ == "__main__":
    download_and_export_yolo_model()

