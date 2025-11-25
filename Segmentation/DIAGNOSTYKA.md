# 🔍 Diagnostyka połączenia Backend ↔ React Native

## Problem: Aplikacja nie może połączyć się z backendem

### Krok 1: Sprawdź czy backend działa

W przeglądarce otwórz:
- `http://localhost:5000/health` - powinno zwrócić JSON z `status: "ok"`
- `http://localhost:5000/` - powinno pokazać informacje o API

### Krok 2: Sprawdź logi backendu

W terminalu backendu powinieneś widzieć:
```
📥 GET /health od: 127.0.0.1
💚 Health check od: 127.0.0.1
```

Jeśli nie widzisz żądań z aplikacji, oznacza to, że żądania nie docierają do backendu.

### Krok 3: Sprawdź URL w aplikacji

W logach React Native powinieneś widzieć:
```
LOG [CatApp] Sprawdzanie połączenia z backendem: http://10.0.2.2:5000
LOG Platform: android
```

**Dla Android Emulator:**
- URL powinien być: `http://10.0.2.2:5000`
- `10.0.2.2` to specjalny adres Android emulatora, który mapuje na `localhost` hosta

**Dla iOS Simulator:**
- URL powinien być: `http://localhost:5000`

**Dla urządzenia fizycznego:**
- URL powinien być: `http://192.168.1.12:5000` (IP z terminala backendu)
- Upewnij się, że telefon i komputer są w tej samej sieci WiFi

### Krok 4: Test połączenia z emulatora

W terminalu Android emulatora (adb shell):
```bash
adb shell
curl http://10.0.2.2:5000/health
```

Powinno zwrócić JSON z `status: "ok"`.

### Krok 5: Sprawdź firewall

Windows może blokować port 5000. Sprawdź:
1. Otwórz "Zapora systemu Windows z zabezpieczeniami zaawansowanymi"
2. Sprawdź czy port 5000 jest otwarty
3. Lub tymczasowo wyłącz firewall do testów

### Krok 6: Sprawdź czy backend nasłuchuje na wszystkich interfejsach

W `backend_server.py` powinno być:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

`0.0.0.0` oznacza, że backend nasłuchuje na wszystkich interfejsach sieciowych.

### Krok 7: Alternatywne rozwiązanie - użyj IP sieciowe

Jeśli `10.0.2.2` nie działa dla Android emulatora:

1. Znajdź IP swojego komputera:
   ```bash
   ipconfig
   # Szukaj "IPv4 Address" - np. 192.168.1.12
   ```

2. Zmień URL w `App.tsx`:
   ```typescript
   android: __DEV__ ? 'http://192.168.1.12:5000' : 'http://192.168.1.12:5000',
   ```

3. Upewnij się, że telefon/emulator i komputer są w tej samej sieci

### Krok 8: Sprawdź logi aplikacji

W React Native powinieneś widzieć:
```
LOG [CatApp] Sprawdzanie połączenia z backendem: http://10.0.2.2:5000
LOG [CatApp] Backend dostępny, modele załadowane: true
LOG [CatApp] Wysyłanie obrazu do backendu segmentacji...
```

Jeśli widzisz błąd "Network request failed", oznacza to problem z połączeniem sieciowym.

### Krok 9: Test z curl

Z terminala komputera:
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test z IP sieciowego
curl http://192.168.1.12:5000/health
```

Oba powinny zwrócić JSON.

### Krok 10: Sprawdź CORS

Backend ma skonfigurowany CORS, ale jeśli nadal są problemy, sprawdź logi backendu - powinny pokazywać wszystkie żądania.

## Najczęstsze problemy:

1. **Backend nie nasłuchuje na 0.0.0.0** - zmień na `host='0.0.0.0'`
2. **Firewall blokuje port 5000** - otwórz port lub wyłącz firewall
3. **Zły URL w aplikacji** - sprawdź `SEGMENTATION_BACKEND_URL` w `App.tsx`
4. **Emulator nie może połączyć się z hostem** - użyj IP sieciowego zamiast 10.0.2.2
5. **Timeout** - segmentacja może trwać długo, zwiększ timeout w aplikacji

