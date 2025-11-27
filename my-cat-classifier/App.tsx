// App.jsx — dokładność priorytet, ONNX z external data, Resize 224×224 + Normalize
import 'react-native-reanimated';
import React, { useCallback, useEffect } from 'react';
import { SafeAreaView, View, Text, Pressable, ActivityIndicator, FlatList } from 'react-native';
import { CameraView } from 'expo-camera';

import { useCatClassifier } from './src/hooks/useCatClassifier';
import { useCameraLoop } from './src/hooks/useCameraLoop';
import { COLORS } from './src/config/constants';

const { FG, FG_MUTED } = COLORS;

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

  const {
    cameraActive,
    startCamera,
    pauseCapture,
    resumeCapture,
    cameraRef,
    permission,
    handleCameraReady,
    handleMountError,
  } = useCameraLoop({
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
    void startCamera();
  }, [startCamera]);

  const handleReloadModel = useCallback(async () => {
    pauseCapture();
    await reloadModel();
    if (!cameraActive) {
      await startCamera();
    }
    resumeCapture();
  }, [cameraActive, pauseCapture, reloadModel, resumeCapture, startCamera]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#000' }}>
      <View style={{ flex: 1, backgroundColor: '#000' }}>
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
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16 }}>
            <Text style={{ color: FG_MUTED, fontSize: 16, textAlign: 'center' }}>
              Udziel dostępu do aparatu w ustawieniach systemu, aby wyświetlić podgląd.
            </Text>
          </View>
        )}

        {!cameraActive && permission?.granted && (
          <View
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              right: 0,
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(0,0,0,0.35)',
            }}
          >
            <ActivityIndicator />
            <Text style={{ color: FG_MUTED, marginTop: 8 }}>Trwa uruchamianie podglądu…</Text>
          </View>
        )}

        <View
          pointerEvents="box-none"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            padding: 14,
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <View style={{ backgroundColor: 'rgba(0,0,0,0.45)', padding: 10, borderRadius: 12 }}>
            <Text style={{ color: FG, fontSize: 13, fontWeight: '700' }}>Cat Classifier</Text>
            <Text style={{ color: ready ? '#b5ffb5' : FG_MUTED, marginTop: 2, fontSize: 12 }}>{status}</Text>
          </View>

          {busy && (
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
              <ActivityIndicator size="small" />
              <Text style={{ color: FG, fontSize: 12 }}>Klasyfikuję…</Text>
            </View>
          )}
        </View>

        <View
          pointerEvents="box-none"
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 0,
            padding: 14,
            gap: 10,
          }}
        >
          {probTopK.length > 0 && !busy && (
            <View style={{ backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 12, padding: 12 }}>
              <FlatList
                data={probTopK}
                keyExtractor={(item, idx) => `${item.label}_${idx}`}
                renderItem={({ item }) => (
                  <View
                    style={{
                      flexDirection: 'row',
                      justifyContent: 'space-between',
                      paddingVertical: 6,
                    }}
                  >
                    <Text style={{ color: FG, fontSize: 15 }}>{item.label}</Text>
                    <Text style={{ color: FG_MUTED }}>{(item.p * 100).toFixed(1)}%</Text>
                  </View>
                )}
              />
            </View>
          )}

          <Pressable
            onPress={() => {
              void handleReloadModel();
            }}
            disabled={busy}
            style={{
              backgroundColor: 'rgba(0,0,0,0.65)',
              paddingVertical: 12,
              borderRadius: 12,
              alignItems: 'center',
              borderWidth: 1,
              borderColor: 'rgba(255,255,255,0.08)',
              opacity: busy ? 0.75 : 1,
            }}
          >
            <Text style={{ color: FG, fontSize: 16, fontWeight: '600' }}>🔁 Przeładuj model</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}
