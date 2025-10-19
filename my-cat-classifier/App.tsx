// App.jsx — robust + logi + auto wykrywanie nazw wej/wyj ONNX
import 'react-native-reanimated'; // ważne: JSI przed wszystkim
import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  Pressable,
  Image,
  ActivityIndicator,
  FlatList,
  Alert,
} from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';

// labels.json przez require (działa bez TS)
const labels = require('./assets/labels.json');

// helper konwersji JPEG base64 → Float32 CHW
import { chwFromBase64JPEG224 } from './src/utils/preprocess';

// normalizacja jak w ImageNet
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

// UI kolory
const BG = '#0b0b0c';
const FG = '#ffffff';
const FG_MUTED = '#cfcfcf';
const ACCENT = '#1f6feb';
const BORDER = '#222';

export default function App() {
  const [status, setStatus] = useState('⏳ Inicjalizacja…');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [previewUri, setPreviewUri] = useState(null);
  const [probTopK, setProbTopK] = useState([]);
  const sessionRef = useRef(null);

  const log = (...a) => console.log('[CatApp]', ...a);
  const warn = (...a) => console.warn('[CatApp]', ...a);
  const err  = (...a) => console.error('[CatApp]', ...a);

  const topK = (probs, k = 3) =>
    probs.map((p, i) => ({ i, p }))
         .sort((a, b) => b.p - a.p)
         .slice(0, Math.min(k, probs.length));

  // ---- Ładowanie modelu (z logami) ----
  const loadModel = useCallback(async () => {
    try {
      setReady(false);
      setStatus('📦 Ładowanie modelu…');
      console.time('Asset.loadAsync');

      const [modelAsset] = await Asset.loadAsync(
        // <-- UPEWNIJ SIĘ, że nazwa pliku zgadza się z dyskiem
        require('./assets/models/mobilenetv2_finetuned.onnx')
      );
      console.timeEnd('Asset.loadAsync');
      log('Model asset uri:', modelAsset.localUri);

      setStatus('🧠 Tworzenie sesji ORT…');
      console.time('ORT.create');
      sessionRef.current = await ort.InferenceSession.create(modelAsset.localUri, {
        executionProviders: ['cpu'], // można spróbować 'xnnpack' na nowszych urządzeniach
      });
      console.timeEnd('ORT.create');

      // krótki podgląd nazw wej/wyj
      try {
        log('Input names (jeśli dostępne):', sessionRef.current.inputNames);
      } catch {}
      try {
        const outNames = Object.keys(await sessionRef.current.outputNames || {});
        log('Output names (jeśli dostępne):', outNames);
      } catch {}

      setStatus('✅ Gotowe');
      setReady(true);
    } catch (e) {
      err('Błąd ładowania modelu:', e?.message || e);
      setStatus('❌ Błąd ładowania modelu (detale w konsoli)');
      Alert.alert('Model error', String(e?.message || e));
    }
  }, []);

  useEffect(() => {
    loadModel();
  }, [loadModel]);

  // ---- Wybór zdjęcia ----
  const pickImage = useCallback(async () => {
    try {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Brak uprawnień', 'Potrzebny dostęp do galerii.');
        return;
      }
      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 1,
        base64: false,
      });
      if (res.canceled || !res.assets?.length) return;

      const uri = res.assets[0].uri;
      log('Wybrano:', uri);

      setStatus('🛠️ Skaluję do 224×224…');
      console.time('Resize+Base64');
      const resized = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 224, height: 224 } }],
        { compress: 1, format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );
      console.timeEnd('Resize+Base64');

      setPreviewUri(resized.uri);
      if (!resized.base64) throw new Error('Brak base64 po przetwarzaniu obrazu');
      await classify(resized.base64);
    } catch (e) {
      err('Błąd wyboru/skalowania:', e?.message || e);
      setStatus('❌ Błąd obrazu (detale w konsoli)');
      Alert.alert('Image error', String(e?.message || e));
    }
  }, []);

  // ---- Klasyfikacja ----
  const classify = useCallback(async (jpegBase64) => {
    const session = sessionRef.current;
    if (!session) {
      warn('Sesja ORT niegotowa');
      setStatus('⏳ Model jeszcze się ładuje…');
      return;
    }

    setBusy(true);
    setStatus('🤖 Klasyfikuję…');
    try {
      console.time('Preprocess');
      const chwFloat32 = chwFromBase64JPEG224(jpegBase64, IMAGENET_MEAN, IMAGENET_STD);
      console.timeEnd('Preprocess');

      const inputTensor = new ort.Tensor('float32', chwFloat32, [1, 3, 224, 224]);

      // podgląd nazw wejść i dynamiczne mapowanie feedów
      let inputName = 'input';
      try {
        if (Array.isArray(session.inputNames) && session.inputNames.length > 0) {
          inputName = session.inputNames[0];
        }
      } catch {}
      const feeds = {};
      feeds[inputName] = inputTensor;
      log('Używam input name:', inputName);

      console.time('ORT.run');
      const outputMap = await session.run(feeds);
      console.timeEnd('ORT.run');

      const outKeys = Object.keys(outputMap);
      log('Output keys:', outKeys);
      if (outKeys.length === 0) throw new Error('Brak wyjść z modelu');

      const firstOutName = outputMap.logits ? 'logits' : outKeys[0];
      const outTensor = outputMap[firstOutName];
      if (!outTensor?.data) throw new Error('Puste wyjście modelu');

      const logits = outTensor.data; // Float32Array
      log('logits len:', logits.length, 'first5:', Array.from(logits).slice(0, 5));

      // softmax (stabilny)
      const max = Math.max(...logits);
      const exps = Array.from(logits, v => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(v => v / (sum || 1)); // guard przed dzieleniem przez 0

      if (!Number.isFinite(probs[0])) {
        throw new Error('Wyliczono NaN/Inf w softmax — sprawdź wejście i normalizację');
      }

      // dopasowanie długości labels
      if (labels.length !== probs.length) {
        warn(`labels(${labels.length}) != probs(${probs.length}) — zmapuję po indeksach dostępnych`);
      }

      const top = topK(probs, 3).map(({ i, p }) => ({
        label: labels[i] ?? `cls_${i}`,
        p,
      }));

      setProbTopK(top);
      setStatus('✅ Gotowe');
      log('TOP-3:', top.map(t => `${t.label}: ${(t.p * 100).toFixed(1)}%`).join(', '));
    } catch (e) {
      err('Błąd klasyfikacji:', e?.message || e);
      setStatus('❌ Błąd klasyfikacji (detale w konsoli)');
      Alert.alert('Inference error', String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }, []);

  // ---- UI ----
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: BG }}>
      <View style={{ padding: 16, gap: 12 }}>
        <Text style={{ color: FG, fontSize: 28, fontWeight: '800' }}>🐱 Cat Classifier (ONNX)</Text>
        <Text style={{ color: ready ? '#6ee17a' : FG_MUTED }}>☑ {status}</Text>

        <View style={{ flexDirection: 'row', gap: 12 }}>
          <Pressable
            onPress={pickImage}
            disabled={!ready || busy}
            style={{
              backgroundColor: ready && !busy ? ACCENT : '#3a3a3a',
              padding: 14,
              borderRadius: 16,
              alignItems: 'center',
              flex: 1,
              opacity: ready && !busy ? 1 : 0.7,
            }}
          >
            <Text style={{ color: FG, fontSize: 18, fontWeight: '700' }}>Wybierz zdjęcie</Text>
          </Pressable>

          <Pressable
            onPress={loadModel}
            disabled={busy}
            style={{
              backgroundColor: '#2c2c2c',
              padding: 14,
              borderRadius: 16,
              alignItems: 'center',
              width: 140,
            }}
          >
            <Text style={{ color: FG, fontSize: 16, fontWeight: '600' }}>🔁 Przeładuj model</Text>
          </Pressable>
        </View>

        {previewUri && (
          <Image
            source={{ uri: previewUri }}
            style={{ width: 224, height: 224, borderRadius: 16, alignSelf: 'center', marginTop: 10 }}
          />
        )}

        {busy && (
          <View style={{ marginTop: 16, alignItems: 'center' }}>
            <ActivityIndicator />
            <Text style={{ color: FG_MUTED, marginTop: 8 }}>Klasyfikuję…</Text>
          </View>
        )}

        {probTopK.length > 0 && !busy && (
          <View style={{ marginTop: 16 }}>
            <Text style={{ color: '#ddd', fontSize: 18, marginBottom: 8 }}>Wynik (Top-3):</Text>
            <FlatList
              data={probTopK}
              keyExtractor={(item, idx) => `${item.label}_${idx}`}
              renderItem={({ item }) => (
                <View
                  style={{
                    flexDirection: 'row',
                    justifyContent: 'space-between',
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: BORDER,
                  }}
                >
                  <Text style={{ color: FG, fontSize: 16 }}>{item.label}</Text>
                  <Text style={{ color: FG, fontSize: 16 }}>{(item.p * 100).toFixed(1)}%</Text>
                </View>
              )}
            />
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}
