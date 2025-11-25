# Backend Server dla Segmentacji Kotów

Backend Flask do komunikacji między React Native aplikacją a skryptem Python do segmentacji kotów.

## 🚀 Instalacja

1. **Zainstaluj zależności:**
```bash
cd Segmentation
pip install -r requirements.txt
```

2. **Upewnij się, że masz modele:**
   - `last.pt` - model YOLO do detekcji kotów (w katalogu Segmentation)
   - `sam_vit_h_4b8939.pth` - model SAM do segmentacji (w katalogu Segmentation)

## 📡 Uruchomienie

```bash
cd Segmentation
python backend_server.py
```

Serwer uruchomi się na: `http://localhost:5000`

## 🔧 Konfiguracja w React Native

W pliku `App.tsx` ustaw właściwy URL backendu:

```typescript
const SEGMENTATION_BACKEND_URL = __DEV__ 
  ? 'http://localhost:5000'      // Dla developmentu (iOS simulator)
  : 'http://10.0.2.2:5000';       // Dla emulatora Android
```

**Dla urządzenia fizycznego:**
- Znajdź IP swojego komputera (np. `192.168.1.100`)
- Użyj: `http://192.168.1.100:5000`
- Upewnij się, że telefon i komputer są w tej samej sieci WiFi

## 📝 Endpointy API

### GET `/health`
Sprawdza status serwera i czy modele są załadowane.

**Odpowiedź:**
```json
{
  "status": "ok",
  "models_loaded": true
}
```

### POST `/segment`
Segmentuje obraz i zwraca zsegmentowany obraz w formacie base64.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Odpowiedź:**
```json
{
  "success": true,
  "segmented_image": "base64_string...",
  "message": "Segmentacja zakończona pomyślnie"
}
```

### POST `/segment_file`
Segmentuje obraz i zwraca zsegmentowany obraz jako plik JPEG.

**Request:**
- Multipart form-data z kluczem `file`

**Odpowiedź:**
- Plik JPEG z zsegmentowanym obrazem

## 🐛 Rozwiązywanie problemów

### Błąd: "Modele nie są załadowane"
- Sprawdź czy pliki `last.pt` i `sam_vit_h_4b8939.pth` istnieją w katalogu Segmentation
- Sprawdź logi serwera przy starcie

### Błąd: "Network request failed" w React Native
- Sprawdź czy serwer jest uruchomiony
- Sprawdź czy używasz właściwego URL (10.0.2.2 dla Android, localhost dla iOS)
- Sprawdź firewall - port 5000 musi być otwarty
- Dla urządzenia fizycznego: upewnij się, że telefon i komputer są w tej samej sieci

### Wolne przetwarzanie
- Użyj GPU (CUDA) dla znacznie szybszego przetwarzania
- Modele ładują się przy starcie serwera (może to chwilę potrwać)

## 📱 Testowanie

Możesz przetestować backend używając curl:

```bash
# Sprawdź status
curl http://localhost:5000/health

# Segmentuj obraz (wymaga pliku test.jpg)
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

