"""
Backend server Flask do komunikacji z React Native aplikacją.
Przyjmuje obrazy, uruchamia segmentację i zwraca zsegmentowane obrazy.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from pathlib import Path
import io
import base64
from PIL import Image
import tempfile
import os
from segment_cats_last import load_models, segment_cat
from roi import process_roi
import json

app = Flask(__name__)
# CORS - pozwala na żądania z React Native (wszystkie źródła w development)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow-headers": ["Content-Type", "Authorization"]
    }
})

# Globalne zmienne do przechowywania załadowanych modeli
yolo_model = None
sam_predictor = None

# Inicjalizacja modeli przy starcie serwera
print("🔄 Ładowanie modeli segmentacji...")
try:
    yolo_model, sam_predictor = load_models()
    if yolo_model is not None and sam_predictor is not None:
        print("✅ Modele załadowane pomyślnie!")
        print(f"   YOLO model: {yolo_model is not None}")
        print(f"   SAM predictor: {sam_predictor is not None}")
    else:
        print("⚠️ Modele zwrócone jako None!")
        yolo_model = None
        sam_predictor = None
except Exception as e:
    import traceback
    print(f"❌ Błąd ładowania modeli: {e}")
    print("⚠️ Serwer uruchomi się, ale segmentacja nie będzie działać")
    traceback.print_exc()
    yolo_model = None
    sam_predictor = None


@app.before_request
def log_request_info():
    """Loguj wszystkie żądania."""
    print(f"📥 {request.method} {request.path} od: {request.remote_addr}")
    if request.method == 'POST':
        print(f"   Content-Type: {request.content_type}")
        if request.is_json:
            data_size = len(str(request.json))
            print(f"   Rozmiar danych JSON: ~{data_size} znaków")


@app.route('/', methods=['GET'])
def index():
    """Endpoint główny - informacje o API."""
    return jsonify({
        'message': 'Backend segmentacji kotów',
        'endpoints': {
            'GET /health': 'Sprawdzenie statusu serwera',
            'POST /segment': 'Segmentacja obrazu (zwraca base64)',
            'POST /segment_file': 'Segmentacja obrazu (zwraca plik)',
            'POST /roi': 'Wycinanie ROI z zsegmentowanego obrazu (zwraca base64)',
            'POST /process_video': 'Przetwarzanie wideo (ekstrakcja klatek i analiza)'
        },
        'models_loaded': yolo_model is not None and sam_predictor is not None
    })


@app.route('/health', methods=['GET'])
def health():
    """Endpoint do sprawdzania czy serwer działa."""
    print(f"💚 Health check od: {request.remote_addr}")
    models_loaded = yolo_model is not None and sam_predictor is not None
    print(f"   Modele załadowane: {models_loaded} (YOLO: {yolo_model is not None}, SAM: {sam_predictor is not None})")
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'yolo_loaded': yolo_model is not None,
        'sam_loaded': sam_predictor is not None
    })


@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Endpoint do segmentacji obrazu.
    Przyjmuje obraz w formacie base64 lub jako plik.
    Zwraca zsegmentowany obraz w formacie base64.
    """
    global yolo_model, sam_predictor
    
    print(f"📥 Otrzymano żądanie segmentacji od: {request.remote_addr}")
    
    if yolo_model is None or sam_predictor is None:
        print("❌ Modele nie są załadowane!")
        return jsonify({
            'error': 'Modele nie są załadowane'
        }), 500
    
    try:
        # Sprawdź czy obraz jest w formacie base64
        if 'image' in request.json:
            # Obraz jako base64 string
            image_data = request.json['image']
            # Usuń prefix "data:image/jpeg;base64," jeśli istnieje
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Dekoduj base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif 'file' in request.files:
            # Obraz jako plik
            file = request.files['file']
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        else:
            return jsonify({
                'error': 'Brak obrazu w żądaniu. Użyj "image" (base64) lub "file" (multipart/form-data)'
            }), 400
        
        if image is None:
            print("❌ Nie można zdekodować obrazu")
            return jsonify({
                'error': 'Nie można zdekodować obrazu'
            }), 400
        
        print(f"📸 Obraz zdekodowany pomyślnie, rozmiar: {image.shape}")
        
        # Zapisz obraz tymczasowo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image)
        
        print(f"💾 Obraz zapisany tymczasowo: {tmp_path}")
        
        try:
            # Wykonaj segmentację
            print(f"🔄 Rozpoczynam segmentację obrazu...")
            mask, segmented, cropped = segment_cat(
                tmp_path,
                yolo_model,
                sam_predictor
            )
            
            if mask is None:
                print("⚠️ Nie znaleziono kota na obrazie")
                return jsonify({
                    'error': 'Nie znaleziono kota na obrazie'
                }), 400
            
            print(f"✅ Segmentacja zakończona pomyślnie")
            
            # Konwertuj zsegmentowany obraz do base64
            # Użyj segmented (kot na białym tle)
            print(f"🖼️ Konwertuję zsegmentowany obraz do base64...")
            segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', segmented_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            print(f"✅ Wysyłanie zsegmentowanego obrazu do klienta")
            return jsonify({
                'success': True,
                'segmented_image': image_base64,
                'message': 'Segmentacja zakończona pomyślnie'
            })
            
        finally:
            # Usuń tymczasowy plik
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'error': f'Błąd podczas segmentacji: {error_msg}'
        }), 500


@app.route('/segment_file', methods=['POST'])
def segment_image_file():
    """
    Endpoint do segmentacji obrazu - zwraca plik zamiast base64.
    Przyjmuje obraz jako plik.
    Zwraca zsegmentowany obraz jako plik JPEG.
    """
    global yolo_model, sam_predictor
    
    if yolo_model is None or sam_predictor is None:
        return jsonify({
            'error': 'Modele nie są załadowane'
        }), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'Brak pliku w żądaniu'
            }), 400
        
        file = request.files['file']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'error': 'Nie można zdekodować obrazu'
            }), 400
        
        # Zapisz obraz tymczasowo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image)
        
        try:
            # Wykonaj segmentację
            mask, segmented, cropped = segment_cat(
                tmp_path,
                yolo_model,
                sam_predictor
            )
            
            if mask is None:
                return jsonify({
                    'error': 'Nie znaleziono kota na obrazie'
                }), 400
            
            # Konwertuj zsegmentowany obraz do JPEG w pamięci
            segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', segmented_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Zwróć jako plik
            return send_file(
                io.BytesIO(buffer),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='segmented.jpg'
            )
            
        finally:
            # Usuń tymczasowy plik
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'error': f'Błąd podczas segmentacji: {error_msg}'
        }), 500


@app.route('/roi', methods=['POST'])
def extract_roi_from_segmented():
    """
    Endpoint do wycinania ROI z zsegmentowanego obrazu.
    Przyjmuje zsegmentowany obraz (kot na białym tle) i zwraca wycięty ROI.
    """
    print(f"📥 Otrzymano żądanie ROI od: {request.remote_addr}")
    
    try:
        # Sprawdź czy obraz jest w formacie base64
        if 'image' in request.json:
            # Obraz jako base64 string
            image_data = request.json['image']
            # Usuń prefix "data:image/jpeg;base64," jeśli istnieje
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Dekoduj base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            segmented_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif 'file' in request.files:
            # Obraz jako plik
            file = request.files['file']
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            segmented_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        else:
            return jsonify({
                'error': 'Brak obrazu w żądaniu. Użyj "image" (base64) lub "file" (multipart/form-data)'
            }), 400
        
        if segmented_image is None:
            print("❌ Nie można zdekodować zsegmentowanego obrazu")
            return jsonify({
                'error': 'Nie można zdekodować obrazu'
            }), 400
        
        print(f"📸 Zsegmentowany obraz zdekodowany, rozmiar: {segmented_image.shape}")
        
        # Przetwórz ROI
        print(f"🔄 Przetwarzanie ROI...")
        roi_image = process_roi(segmented_image)
        
        if roi_image is None:
            return jsonify({
                'error': 'Nie udało się wyciąć ROI z obrazu'
            }), 400
        
        # Konwertuj ROI do base64
        print(f"🖼️ Konwertuję ROI do base64...")
        # ROI jest w RGB, konwertuj do BGR dla cv2.imencode
        roi_bgr = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        roi_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ ROI wycięty i wysłany do klienta")
        return jsonify({
            'success': True,
            'roi_image': roi_base64,
            'message': 'ROI wycięty pomyślnie'
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"❌ Błąd podczas wycinania ROI: {error_msg}")
        traceback.print_exc()
        return jsonify({
            'error': f'Błąd podczas wycinania ROI: {error_msg}'
        }), 500


@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Endpoint do przetwarzania wideo.
    Ekstraktuje klatki, przetwarza każdą klatkę i zwraca wyniki.
    """
    global yolo_model, sam_predictor
    
    print(f"📥 Otrzymano żądanie przetwarzania wideo od: {request.remote_addr}")
    
    if yolo_model is None or sam_predictor is None:
        return jsonify({
            'error': 'Modele nie są załadowane'
        }), 500
    
    try:
        # Sprawdź czy wideo jest w formacie base64 lub jako plik
        if 'video' in request.json:
            # Wideo jako base64 string
            video_data = request.json['video']
            frame_count = int(request.json.get('frame_count', 10))
            
            # Usuń prefix "data:video/mp4;base64," jeśli istnieje
            if ',' in video_data:
                video_data = video_data.split(',')[1]
            
            # Dekoduj base64
            video_bytes = base64.b64decode(video_data)
            
            # Zapisz wideo tymczasowo
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(video_bytes)
                
        elif 'file' in request.files:
            # Wideo jako plik
            file = request.files['file']
            frame_count = int(request.form.get('frame_count', 10))
            
            # Zapisz wideo tymczasowo
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_path = tmp_file.name
                file.save(tmp_path)
        else:
            return jsonify({
                'error': 'Brak wideo w żądaniu. Użyj "video" (base64) lub "file" (multipart/form-data)'
            }), 400
        
        print(f"💾 Wideo zapisane tymczasowo: {tmp_path}")
        
        try:
            # Otwórz wideo
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return jsonify({
                    'error': 'Nie można otworzyć wideo'
                }), 400
            
            # Pobierz informacje o wideo
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📹 Wideo: {total_frames} klatek, {fps:.2f} FPS, {duration:.2f}s")
            
            # Oblicz interwał klatek
            frame_interval = max(1, total_frames // (frame_count + 1))
            
            results = []
            frame_indices = []
            
            # Ekstrahuj i przetwarzaj klatki
            for i in range(1, frame_count + 1):
                frame_idx = i * frame_interval
                if frame_idx >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"⚠️ Nie można odczytać klatki {frame_idx}")
                    continue
                
                print(f"🔄 Przetwarzanie klatki {i}/{frame_count} (frame {frame_idx})...")
                
                # Zapisz klatkę tymczasowo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as frame_file:
                    frame_path = frame_file.name
                    cv2.imwrite(frame_path, frame)
                
                try:
                    # Segmentacja
                    mask, segmented, cropped = segment_cat(
                        frame_path,
                        yolo_model,
                        sam_predictor
                    )
                    
                    if mask is None:
                        print(f"⚠️ Nie znaleziono kota w klatce {i}")
                        continue
                    
                    # ROI
                    roi_image = process_roi(segmented)
                    
                    # Konwertuj wyniki do base64
                    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
                    _, segmented_buffer = cv2.imencode('.jpg', segmented_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    segmented_base64 = base64.b64encode(segmented_buffer).decode('utf-8')
                    
                    roi_base64 = None
                    if roi_image is not None:
                        # ROI jest w RGB, konwertuj do BGR dla cv2.imencode
                        roi_bgr = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)
                        _, roi_buffer = cv2.imencode('.jpg', roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        roi_base64 = base64.b64encode(roi_buffer).decode('utf-8')
                    
                    results.append({
                        'frame_index': frame_idx,
                        'frame_number': i,
                        'time_seconds': frame_idx / fps if fps > 0 else 0,
                        'segmented_image': segmented_base64,
                        'roi_image': roi_base64,
                    })
                    
                    frame_indices.append(frame_idx)
                    
                finally:
                    if os.path.exists(frame_path):
                        os.unlink(frame_path)
            
            cap.release()
            
            print(f"✅ Przetworzono {len(results)} klatek z wideo")
            
            return jsonify({
                'success': True,
                'video_info': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration_seconds': duration,
                },
                'frames': results,
                'message': f'Przetworzono {len(results)} klatek'
            })
            
        finally:
            # Usuń tymczasowy plik wideo
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"❌ Błąd podczas przetwarzania wideo: {error_msg}")
        traceback.print_exc()
        return jsonify({
            'error': f'Błąd podczas przetwarzania wideo: {error_msg}'
        }), 500


if __name__ == '__main__':
    print("\n🚀 Uruchamianie serwera backend...")
    print("📡 Serwer będzie dostępny na: http://localhost:5000")
    print("📝 Endpointy:")
    print("   GET  /health - sprawdzenie statusu")
    print("   POST /segment - segmentacja (zwraca base64)")
    print("   POST /segment_file - segmentacja (zwraca plik)")
    print("   POST /roi - wycinanie ROI z zsegmentowanego obrazu (zwraca base64)")
    print("   POST /process_video - przetwarzanie wideo (ekstrakcja klatek i analiza)")
    print("\n⚠️  Upewnij się, że React Native aplikacja używa właściwego adresu IP!")
    print("   Dla emulatora Android: http://10.0.2.2:5000")
    print("   Dla urządzenia fizycznego: http://<IP_KOMPUTERA>:5000")
    print("\n" + "="*60)
    print("Status modeli:")
    print(f"   YOLO: {'✅ Załadowany' if yolo_model is not None else '❌ Nie załadowany'}")
    print(f"   SAM:  {'✅ Załadowany' if sam_predictor is not None else '❌ Nie załadowany'}")
    print("="*60)
    
    if yolo_model is None or sam_predictor is None:
        print("\n⚠️ UWAGA: Modele nie są załadowane!")
        print("   Segmentacja i ROI nie będą działać.")
        print("\n   Sprawdź:")
        print("   1. Czy plik last.pt znajduje się w katalogu Segmentation/")
        print("   2. Czy plik sam_vit_h_4b8939.pth znajduje się w katalogu Segmentation/")
        print("   3. Czy wszystkie zależności są zainstalowane (pip install -r requirements.txt)")
        print("   4. Sprawdź błędy powyżej - powinny być widoczne szczegóły problemu")
        print("\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

