// App.jsx — dokładność priorytet, ONNX z external data, Resize 224×224 + Normalize
import 'react-native-reanimated';
import React, { useCallback, useEffect } from 'react';
import { SafeAreaView, View, Text, Pressable, ActivityIndicator, FlatList } from 'react-native';
import { CameraView } from 'expo-camera';

import { useCatClassifier } from './src/hooks/useCatClassifier';
import { useCameraLoop } from './src/hooks/useCameraLoop';
import { COLORS } from './src/config/constants';

const { BG, FG, FG_MUTED, BORDER } = COLORS;

export default function App() {
  const {
    status,
    updateStatus,
    busy,
    ready,
    setPreviewUri,
    probTopK,
    reloadModel,
    classifyBase64,
    resizeTo224Base64,
    resetSilentStatus,
    warn,
    err,
  } = useCatClassifier();

  const { cameraActive, startCamera, stopCamera, cameraRef, permission, handleCameraReady, handleMountError } =
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

  useEffect(() => {
    if (ready) {
      void startCamera();
    }
  }, [ready, startCamera]);

  const handleReloadModel = useCallback(async () => {
    stopCamera();
    await reloadModel();
    void startCamera();
  }, [reloadModel, startCamera, stopCamera]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: BG }}>
      <View style={{ flex: 1, padding: 16, gap: 12 }}>
        <View style={{ gap: 6 }}>
          <Text style={{ color: FG, fontSize: 28, fontWeight: '800' }}>🐱 Cat Classifier (ONNX)</Text>
          <Text style={{ color: ready ? '#6ee17a' : FG_MUTED }}>☑ {status}</Text>
        </View>

        <View style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
          <Pressable
            onPress={() => {
              void handleReloadModel();
            }}
            disabled={busy}
            style={{
              backgroundColor: '#2c2c2c',
              paddingHorizontal: 16,
              paddingVertical: 12,
              borderRadius: 16,
              alignItems: 'center',
              width: 170,
            }}
          >
            <Text style={{ color: FG, fontSize: 16, fontWeight: '600' }}>🔁 Przeładuj model</Text>
            <Text style={{ color: FG_MUTED, fontSize: 12 }}>kamerka pauzuje na chwilę</Text>
          </Pressable>
        </View>

        {!permission?.granted && (
          <View style={{ padding: 12, borderRadius: 12, backgroundColor: '#1a1a1a', marginTop: 4 }}>
            <Text style={{ color: FG_MUTED }}>
              Udziel dostępu do aparatu w ustawieniach systemu, aby wykonywać klasyfikację w tle.
            </Text>
          </View>
        )}

        {cameraActive && permission?.granted && (
          <CameraView
            ref={cameraRef}
            style={{ width: 1, height: 1, opacity: 0 }}
            facing="back"
            mode="picture"
            animateShutter={false}
            onCameraReady={handleCameraReady}
            onMountError={handleMountError}
          />
        )}

        <View style={{ padding: 12, borderRadius: 12, backgroundColor: '#1a1a1a', marginTop: 4 }}>
          <Text style={{ color: FG_MUTED }}>
            Podgląd aparatu został wyłączony. Zdjęcia są wykonywane automatycznie w tle dla klasyfikacji.
          </Text>
        </View>

        {busy && (
          <View style={{ marginHorizontal: 16, marginTop: 6, alignItems: 'center' }}>
            <ActivityIndicator />
            <Text style={{ color: FG_MUTED, marginTop: 8 }}>Klasyfikuję…</Text>
          </View>
        )}

        {probTopK.length > 0 && !busy && (
          <View style={{ marginHorizontal: 16, marginBottom: 12 }}>
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
