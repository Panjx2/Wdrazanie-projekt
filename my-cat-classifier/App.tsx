import React, { useMemo, useRef, useState } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  Pressable,
  Image,
  ActivityIndicator,
  FlatList,
} from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';

// ✅ Import JSON przez require (działa bez configu TS)
const labels: string[] = require('./assets/labels.json');

// ✅ Import helpera z src/utils
import { chwFromBase64JPEG224 } from './src/utils/preprocess';

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export default function App() {
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [probTopK, setProbTopK] = useState<Array<{ label: string; p: number }>>([]);
  const [busy, setBusy] = useState(false);
  const sessionRef = useRef<ort.InferenceSession | null>(null);

  // Ładowanie modelu przy starcie
  useMemo(() => {
    (async () => {
      // ✅ Zmieniony path na lokalny
      const [modelAsset] = await Asset.loadAsync(
        require('./assets/models/cats_mobilenetv2.onnx')
      );
      sessionRef.current = await ort.InferenceSession.create(modelAsset.localUri!);
      console.log('Model ONNX załadowany');
    })();
  }, []);

  // Wybór zdjęcia z galerii
  const pickImage = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      alert('Potrzebne uprawnienie do galerii!');
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
      base64: false,
    });

    if (res.canceled || !res.assets?.length) return;
    const uri = res.assets[0].uri;

    // Resize i konwersja do base64
    const resized = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: 224, height: 224 } }],
      { compress: 1, format: ImageManipulator.SaveFormat.JPEG, base64: true }
    );

    setPreviewUri(resized.uri);
    await classify(resized.base64!);
  };

  // Klasyfikacja
  const classify = async (jpegBase64: string) => {
    const session = sessionRef.current;
    if (!session) return;

    setBusy(true);
    try {
      const chwFloat32 = chwFromBase64JPEG224(jpegBase64, IMAGENET_MEAN, IMAGENET_STD);
      const input = new ort.Tensor('float32', chwFloat32, [1, 3, 224, 224]);
      const outputMap = await session.run({ input });
      const logits = outputMap.logits.data as Float32Array;

      // Softmax
      const max = Math.max(...logits);
      const exps = logits.map(v => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(v => v / sum);

      // Top-K (K=3)
      const idx = probs
        .map((p, i) => ({ i, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, Math.min(3, probs.length));

      setProbTopK(idx.map(({ i, p }) => ({ label: labels[i] ?? `cls_${i}`, p })));
    } catch (e) {
      console.warn('Classification error', e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#0b0b0c' }}>
      <View style={{ padding: 16, gap: 12 }}>
        <Text style={{ color: 'white', fontSize: 24, fontWeight: '700' }}>
          🐱 Cat Classifier (ONNX)
        </Text>

        <Pressable
          onPress={pickImage}
          style={{
            backgroundColor: '#1f6feb',
            padding: 14,
            borderRadius: 12,
            alignItems: 'center',
          }}
        >
          <Text style={{ color: 'white', fontSize: 16, fontWeight: '600' }}>
            Wybierz zdjęcie
          </Text>
        </Pressable>

        {previewUri && (
          <Image
            source={{ uri: previewUri }}
            style={{
              width: 224,
              height: 224,
              borderRadius: 12,
              alignSelf: 'center',
              marginTop: 8,
            }}
          />
        )}

        {busy && (
          <View style={{ marginTop: 16, alignItems: 'center' }}>
            <ActivityIndicator />
            <Text style={{ color: '#aaa', marginTop: 8 }}>Klasyfikuję…</Text>
          </View>
        )}

        {probTopK.length > 0 && (
          <View style={{ marginTop: 16 }}>
            <Text style={{ color: '#ddd', fontSize: 18, marginBottom: 8 }}>
              Wynik:
            </Text>
            <FlatList
              data={probTopK}
              keyExtractor={(item) => item.label}
              renderItem={({ item }) => (
                <View
                  style={{
                    flexDirection: 'row',
                    justifyContent: 'space-between',
                    paddingVertical: 8,
                    borderBottomWidth: 1,
                    borderBottomColor: '#222',
                  }}
                >
                  <Text style={{ color: 'white', fontSize: 16 }}>{item.label}</Text>
                  <Text style={{ color: 'white', fontSize: 16 }}>
                    {(item.p * 100).toFixed(1)}%
                  </Text>
                </View>
              )}
            />
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}
