// App.jsx — dokładność priorytet, ONNX z external data, Resize 224×224 + Normalize
import 'react-native-reanimated';
import React from 'react';
import { SafeAreaView, View, Text, Pressable, Image, ActivityIndicator, FlatList } from 'react-native';
import { CameraView } from 'expo-camera';

import { useCatClassifier } from './src/hooks/useCatClassifier';
import { useCameraLoop } from './src/hooks/useCameraLoop';
import { COLORS } from './src/config/constants';

const { BG, FG, FG_MUTED, ACCENT, BORDER } = COLORS;

export default function App() {
  const {
    status,
    updateStatus,
    busy,
    ready,
    previewUri,
    setPreviewUri,
    probTopK,
    pickImage,
    reloadModel,
    classifyBase64,
    resizeTo224Base64,
    resetSilentStatus,
    warn,
    err,
  } = useCatClassifier();

  const { cameraActive, toggleCamera, cameraRef, permission, handleCameraReady, handleMountError } =
    useCameraLoop({
      ready,
      updateStatus,
      classifyBase64,
      resizeTo224Base64,
      resetSilentStatus,
      clearPreview: () => setPreviewUri(null),
      warn,
      err,
    });

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: BG }}>
      <View style={{ padding: 16, gap: 12 }}>
        <Text style={{ color: FG, fontSize: 28, fontWeight: '800' }}>🐱 Cat Classifier (ONNX)</Text>
        <Text style={{ color: ready ? '#6ee17a' : FG_MUTED }}>☑ {status}</Text>

        <View style={{ flexDirection: 'row', gap: 12 }}>
          <Pressable
            onPress={() => {
              void pickImage();
            }}
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
            onPress={() => {
              void toggleCamera();
            }}
            disabled={!ready || busy}
            style={{
              backgroundColor: cameraActive ? '#b91c1c' : '#2c2c2c',
              padding: 14,
              borderRadius: 16,
              alignItems: 'center',
              width: 150,
              opacity: ready && !busy ? 1 : 0.7,
            }}
          >
            <Text style={{ color: FG, fontSize: 16, fontWeight: '600' }}>
              {cameraActive ? '🔚 Wyłącz kamerę' : '📷 Podgląd kamery'}
            </Text>
          </Pressable>

          <Pressable
            onPress={() => {
              void reloadModel();
            }}
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

        {cameraActive && (
          <View
            style={{
              marginTop: 12,
              borderRadius: 16,
              overflow: 'hidden',
              borderWidth: 1,
              borderColor: BORDER,
              height: 320,
            }}
          >
            {permission?.granted ? (
              <CameraView
                ref={cameraRef}
                style={{ flex: 1 }}
                facing="back"
                mode="picture"
                animateShutter={false}
                onCameraReady={handleCameraReady}
                onMountError={handleMountError}
              />
            ) : (
              <View
                style={{
                  flex: 1,
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: '#1a1a1a',
                  padding: 16,
                }}
              >
                <Text style={{ color: FG_MUTED, textAlign: 'center' }}>
                  Aby korzystać z podglądu, udziel dostępu do kamery w ustawieniach systemu.
                </Text>
              </View>
            )}
          </View>
        )}

        {previewUri && !cameraActive && (
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
                    borderColor: BORDER,
                  }}
                >
                  <Text style={{ color: FG, fontSize: 16 }}>{item.label}</Text>
                  <Text style={{ color: FG_MUTED }}>{(item.p * 100).toFixed(1)}%</Text>
                </View>
              )}
            />
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}
