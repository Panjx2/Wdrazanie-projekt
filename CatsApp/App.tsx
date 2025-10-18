// App.tsx
import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {
  SafeAreaView, View, Text, Image, TouchableOpacity, FlatList, StyleSheet,
} from 'react-native';
import labels from './assets/labels.json';

// PyTorch Core
import {torch, torchvision, media, MobileModel} from 'react-native-pytorch-core';

type Result = { label: string; prob: number };

export default function App() {
  const [model, setModel] = useState<MobileModel | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      // Ładowanie modelu z bundla / assets
      const m = await torch.jit._loadForMobile('catmobilenetv2.ptl'); // iOS/Android: w bundle
      setModel(m);
      // Warm-up (opcjonalnie)
      const dummy = torch.zeros([1, 3, 224, 224]);
      await m.forward(dummy);
    })().catch(console.error);
  }, []);

  const pickAndClassify = useCallback(async () => {
    if (!model) return;
    try {
      setBusy(true);
      setResults([]);

      // 1) Wybór zdjęcia (lub media.openCamera())
      const img = await media.pickImage(); // zwraca obiekt obrazu z URI

      // 2) Preprocessing = identyczny jak w treningu:
      //    resize 224, toTensor (0..1, CHW), normalize, unsqueeze(0)
      const resized = await torchvision.transforms.resize(img, {width: 224, height: 224});
      const t = await torchvision.transforms.toTensor(resized);              // [3,224,224]
      const tn = await torchvision.transforms.normalize(
        t,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
      );
      const input = tn.unsqueeze(0);                                         // [1,3,224,224]

      // 3) Inference
      const out = (await model.forward(input)).squeeze(0);                   // [12]
      const probs = out.softmax().tolist() as number[];

      // 4) Top-K
      const topK = 3;
      const idx = probs
        .map((p, i) => ({i, p}))
        .sort((a, b) => b.p - a.p)
        .slice(0, topK);

      setImageUri(img.uri);
      setResults(idx.map(({i, p}) => ({label: labels[i], prob: p})));
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  }, [model]);

  const header = useMemo(
    () => (
      <View style={styles.header}>
        <Text style={styles.title}>Cat Breed Classifier</Text>
        <TouchableOpacity onPress={pickAndClassify} style={styles.button} disabled={!model || busy}>
          <Text style={styles.buttonText}>{busy ? 'Analizuję…' : 'Wybierz zdjęcie'}</Text>
        </TouchableOpacity>
      </View>
    ),
    [pickAndClassify, busy, model],
  );

  return (
    <SafeAreaView style={styles.container}>
      {header}
      {!!imageUri && (
        <Image source={{uri: imageUri}} style={styles.preview} resizeMode="cover" />
      )}
      <FlatList
        data={results}
        keyExtractor={(it) => it.label}
        renderItem={({item}) => (
          <View style={styles.row}>
            <Text style={styles.label}>{item.label}</Text>
            <Text style={styles.prob}>{(item.prob * 100).toFixed(1)}%</Text>
          </View>
        )}
        ListEmptyComponent={
          <Text style={styles.hint}>
            Wybierz zdjęcie kota, a aplikacja pokaże Top-3 rasy z prawdopodobieństwami.
          </Text>
        }
        contentContainerStyle={styles.list}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#111'},
  header: {padding: 16, alignItems: 'center'},
  title: {color: 'white', fontSize: 20, fontWeight: '700', marginBottom: 8},
  button: {backgroundColor: '#4f46e5', paddingVertical: 10, paddingHorizontal: 16, borderRadius: 12},
  buttonText: {color: 'white', fontWeight: '700'},
  preview: {width: '100%', aspectRatio: 1.3},
  list: {padding: 16},
  row: {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10, borderBottomColor: '#333', borderBottomWidth: StyleSheet.hairlineWidth},
  label: {color: 'white', fontSize: 16},
  prob: {color: '#9ca3af', fontVariant: ['tabular-nums']},
  hint: {color: '#9ca3af', textAlign: 'center', padding: 24},
});
