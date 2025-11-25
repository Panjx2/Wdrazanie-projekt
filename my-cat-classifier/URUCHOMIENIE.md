# 🚀 Jak uruchomić aplikację z segmentacją

## Krok 1: Uruchom Backend (już zrobione ✅)

Backend działa na:
- **http://localhost:5000**
- **http://192.168.1.12:5000** (IP sieciowe)

## Krok 2: Uruchom aplikację React Native

### Opcja A: Expo Go (najszybsze)

```bash
cd my-cat-classifier
npm start
```

Następnie:
- Naciśnij `a` dla Android
- Naciśnij `i` dla iOS
- Zeskanuj QR kod w aplikacji Expo Go

### Opcja B: Android Emulator
rr
```bash
cd my-cat-classifier
npm run android
```

**Ważne:** W emulatorze Android użyj URL: `http://10.0.2.2:5000` (to jest localhost dla emulatora)

### Opcja C: iOS Simulator (tylko macOS)

```bash
cd my-cat-classifier
npm run ios
```

**Ważne:** W iOS simulator użyj URL: `http://localhost:5000`

### Opcja D: Urządzenie fizyczne

1. Upewnij się, że telefon i komputer są w tej samej sieci WiFi
2. Użyj IP z terminala backendu: `http://192.168.1.12:5000`
3. Uruchom aplikację:
   ```bash
   cd my-cat-classifier
   npm start
   ```
4. Zeskanuj QR kod w aplikacji Expo Go

## Krok 3: Testowanie

1. Otwórz aplikację w telefonie/emulatorze
2. Wybierz zdjęcie kota
3. Aplikacja automatycznie:
   - Wykryje kota (krok 1-2)
   - Wyśle obraz do backendu (krok 3 - segmentacja)
   - Wyświetli zsegmentowany obraz
   - Wykona klasyfikację (krok 4)

## 🔧 Rozwiązywanie problemów

### Błąd: "Network request failed"

**Dla Android Emulator:**
- Upewnij się, że używasz `http://10.0.2.2:5000`
- Sprawdź czy backend działa: `curl http://localhost:5000/health`

**Dla iOS Simulator:**
- Upewnij się, że używasz `http://localhost:5000`

**Dla urządzenia fizycznego:**
- Upewnij się, że telefon i komputer są w tej samej sieci WiFi
- Użyj IP z terminala backendu: `http://192.168.1.12:5000`
- Sprawdź firewall - port 5000 musi być otwarty

### Sprawdzenie połączenia

Możesz przetestować backend w przeglądarce:
```
http://localhost:5000/health
```

Powinieneś zobaczyć:
```json
{
  "status": "ok",
  "models_loaded": true
}
```

## 📱 Szybki start

```bash
# Terminal 1: Backend (już działa ✅)
cd Segmentation
python backend_server.py

# Terminal 2: React Native
cd my-cat-classifier
npm start
# Następnie naciśnij 'a' dla Android lub 'i' dla iOS
```

