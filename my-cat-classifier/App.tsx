// App.jsx — dokładność priorytet, ONNX z external data, Resize 224×224 + Normalize
import 'react-native-reanimated';
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
  Platform,
} from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';
import { chwFromBase64JPEG224 } from './src/utils/preprocess';

// labels.json (kolejność MUSI być taka sama jak w treningu)
const labels = require('./assets/labels.json');

// Normalizacja ImageNet
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

// UI kolory
const BG = '#0b0b0c';
const FG = '#ffffff';
const FG_MUTED = '#cfcfcf';
const ACCENT = '#1f6feb';
const BORDER = '#222';

// Ustawienia
const USE_BGR = false;          // ustaw na true tylko jeśli trenowałeś w BGR (OpenCV)
const USE_PNG_LOSSLESS = false; // ustaw na true, by zapisać do PNG (dokładniejszy tensor, większy plik)

// ---- Helper: przygotuj ścieżkę modelu z plikiem external data ----
async function prepareOnnxWithExternalData() {
  // Załaduj oba assety, aby dostać localUri
  const [onnxAsset, dataAsset] = await Asset.loadAsync([
    require('./assets/models/mobilenetv2_finetuned.onnx'),
    require('./assets/models/mobilenetv2_finetuned.onnx.data'),
  ]);

  const dir = FileSystem.cacheDirectory + 'ort-model/';
  try { await FileSystem.makeDirectoryAsync(dir, { intermediates: true }); } catch {}

  const modelDst = dir + 'mobilenetv2_finetuned.onnx';
  const dataDst  = dir + 'mobilenetv2_finetuned.onnx.data';

  // Nadpisz, aby zgadzały się dokładne nazwy i lokalizacja
  await FileSystem.copyAsync({ from: onnxAsset.localUri, to: modelDst });
  await FileSystem.copyAsync({ from: dataAsset.localUri, to: dataDst });

  return modelDst; // ORT znajdzie .data w tym samym folderze
}

export default function App() {
  const [status, setStatus] = useState('⏳ Inicjalizacja…');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [previewUri, setPreviewUri] = useState(null);
  const [probTopK, setProbTopK] = useState([]);
  const sessionRef = useRef(null);

  const log  = (...a) => console.log('[CatApp]', ...a);
  const warn = (...a) => console.warn('[CatApp]', ...a);
  const err  = (...a) => console.error('[CatApp]', ...a);

  const topK = (probs, k = 3) =>
    probs.map((p, i) => ({ i, p }))
         .sort((a, b) => b.p - a.p)
         .slice(0, Math.min(k, probs.length));

  // ---- Ładowanie modelu (.onnx + .onnx.data) ----
  const loadModel = useCallback(async () => {
    try {
      setReady(false);
      setStatus('📦 Ładowanie modelu…');

      const modelPath = await prepareOnnxWithExternalData();
      log('Model local path:', modelPath);

      setStatus('🧠 Tworzenie sesji ORT…');
      sessionRef.current = await ort.InferenceSession.create(modelPath, {
        executionProviders: Platform.OS === 'android' ? ['xnnpack', 'cpu'] : ['cpu'],
      });

      log('Input names:', sessionRef.current.inputNames ?? []);
      log('Output names:', sessionRef.current.outputNames ?? []);
      setStatus('✅ Gotowe');
      setReady(true);
    } catch (e) {
      err('Błąd ładowania modelu:', e?.message || e);
      setStatus('❌ Błąd ładowania modelu');
      Alert.alert('Model error', String(e?.message || e));
    }
  }, []);

  useEffect(() => { loadModel(); }, [loadModel]);

  // ---- Wybór zdjęcia ----
  const pickImage = useCallback(async () => {
    try {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Brak uprawnień', 'Potrzebny dostęp do galerii.');
        return;
      }

      // Nowe API: mediaTypes to pojedyncza wartość enum
      const mediaImages =
        (ImagePicker.MediaType && ImagePicker.MediaType.Images) ||
        ImagePicker.MediaTypeOptions.Images;

      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: mediaImages,
        quality: 1,
        base64: false,
      });
      if (res.canceled || !res.assets?.length) return;

      const uri = res.assets[0].uri;
      log('Wybrano:', uri);

      // Dokładnie jak w notebooku: resize do 224x224 (bez cropa)
      setStatus('🛠️ Resize 224×224…');
      console.time('Resize+Base64');

      const outFormat = USE_PNG_LOSSLESS
        ? ImageManipulator.SaveFormat.PNG
        : ImageManipulator.SaveFormat.JPEG;

      const resized = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 224, height: 224 } }],
        {
          compress: USE_PNG_LOSSLESS ? 1 : 0.95,
          format: outFormat,
          base64: true,
        }
      );
      console.timeEnd('Resize+Base64');

      setPreviewUri(resized.uri);
      if (!resized.base64) throw new Error('Brak base64 po przetwarzaniu');

      await classifySingle(resized.base64);
    } catch (e) {
      err('Błąd obrazu:', e?.message || e);
      setStatus('❌ Błąd obrazu');
      Alert.alert('Image error', String(e?.message || e));
    }
  }, []);

  // ---- Klasyfikacja (pojedynczy forward) ----
  const classifySingle = useCallback(async (jpegBase64) => {
    const session = sessionRef.current;
    if (!session) {
      warn('Sesja ORT niegotowa');
      setStatus('⏳ Model się ładuje…');
      return;
    }

    setBusy(true);
    setStatus('🤖 Klasyfikuję…');
    try {
      // JPEG base64 -> Float32 CHW + normalize (ImageNet, RGB/BGR)
      const chw = chwFromBase64JPEG224(jpegBase64, IMAGENET_MEAN, IMAGENET_STD, USE_BGR);

      // Szybkie sanity: średnie kanałów ~0 po normalizacji
      {
        const size = 224 * 224;
        const mR = chw.slice(0, size).reduce((a,b)=>a+b, 0) / size;
        const mG = chw.slice(size, 2*size).reduce((a,b)=>a+b, 0) / size;
        const mB = chw.slice(2*size).reduce((a,b)=>a+b, 0) / size;
        log('CHW means (≈0):', mR.toFixed(3), mG.toFixed(3), mB.toFixed(3));
      }

      const inputName = session.inputNames?.[0] ?? 'input';
      const tensor = new ort.Tensor('float32', chw, [1, 3, 224, 224]);

      const outputMap = await session.run({ [inputName]: tensor });
      const keys = Object.keys(outputMap);

      // preferuj 'prob'/'softmax' jeśli są, inaczej 'logits' / pierwszy klucz
      const probAliases = ['prob', 'probs', 'probabilities', 'softmax'];
      const logitAliases = ['logits', 'output'];
      const outName =
        probAliases.find(k => keys.includes(k)) ??
        logitAliases.find(k => keys.includes(k)) ??
        keys[0];

      const outT = outputMap[outName];
      if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);
      const data = outT.data;

      let probs;
      if (probAliases.includes(outName)) {
        // już prawdopodobieństwa
        probs = Array.from(data);
      } else {
        // logits -> softmax (stabilny)
        let max = -Infinity;
        for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
        const exps = new Float32Array(data.length);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = Math.exp(data[i] - max);
          exps[i] = v;
          sum += v;
        }
        probs = Array.from(exps, v => v / (sum || 1));
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
      setStatus('❌ Błąd klasyfikacji');
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
