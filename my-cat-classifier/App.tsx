// App.jsx — dokładność priorytet, ONNX z external data, Resize 224×224 + Normalize
import 'react-native-reanimated';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView, View, Text, Pressable, ActivityIndicator, FlatList } from 'react-native';
import { CameraView } from 'expo-camera';
import {
  GestureHandlerRootView,
  PinchGestureHandler,
  PanGestureHandler,
  State,
  type PinchGestureHandlerGestureEvent,
  type PinchGestureHandlerStateChangeEvent,
  type PanGestureHandlerGestureEvent,
  type PanGestureHandlerStateChangeEvent,
} from 'react-native-gesture-handler';

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

  const [zoom, setZoom] = useState(0);
  const pinchStartZoomRef = useRef(0);
  const sliderStartZoomRef = useRef(0);
  const sliderWidthRef = useRef(0);

  const {
    cameraActive,
    cameraReady,
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

  const clampZoom = useCallback((value: number) => Math.min(1, Math.max(0, value)), []);

  const handlePinchGesture = useCallback(
    (event: PinchGestureHandlerGestureEvent) => {
      const scale = event.nativeEvent.scale;
      if (Number.isFinite(scale)) {
        const nextZoom = clampZoom(pinchStartZoomRef.current * scale);
        setZoom(nextZoom);
      }
    },
    [clampZoom]
  );

  const handlePinchStateChange = useCallback(
    (event: PinchGestureHandlerStateChangeEvent) => {
      if (event.nativeEvent.state === State.BEGAN) {
        pinchStartZoomRef.current = zoom || 0;
      }
      if (event.nativeEvent.state === State.END || event.nativeEvent.state === State.CANCELLED) {
        pinchStartZoomRef.current = zoom || 0;
      }
    },
    [zoom]
  );

  const handleSliderGesture = useCallback(
    (event: PanGestureHandlerGestureEvent) => {
      if (sliderWidthRef.current <= 0) return;
      const delta = event.nativeEvent.translationX / sliderWidthRef.current;
      const nextZoom = clampZoom(sliderStartZoomRef.current + delta);
      setZoom(nextZoom);
    },
    [clampZoom]
  );

  const handleSliderStateChange = useCallback(
    (event: PanGestureHandlerStateChangeEvent) => {
      if (event.nativeEvent.state === State.BEGAN) {
        sliderStartZoomRef.current = zoom || 0;
      }
      if (event.nativeEvent.state === State.END || event.nativeEvent.state === State.CANCELLED) {
        sliderStartZoomRef.current = zoom || 0;
      }
    },
    [zoom]
  );

  const cameraComponent = useMemo(
    () => (
      <PinchGestureHandler
        onGestureEvent={handlePinchGesture}
        onHandlerStateChange={handlePinchStateChange}
        shouldCancelWhenOutside={false}
      >
        <CameraView
          ref={cameraRef}
          style={{ flex: 1 }}
          facing="back"
          mode="picture"
          animateShutter={false}
          zoom={zoom}
          onCameraReady={handleCameraReady}
          onMountError={handleMountError}
        />
      </PinchGestureHandler>
    ),
    [cameraRef, handleCameraReady, handleMountError, handlePinchGesture, handlePinchStateChange, zoom]
  );

  const handleReloadModel = useCallback(async () => {
    pauseCapture();
    await reloadModel();
    if (!cameraActive) {
      await startCamera();
    }
    resumeCapture();
  }, [cameraActive, pauseCapture, reloadModel, resumeCapture, startCamera]);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaView style={{ flex: 1, backgroundColor: '#000' }}>
        <View style={{ flex: 1, backgroundColor: '#000' }}>
          {permission?.granted ? (
            cameraComponent
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

        {/* Top overlay with status */}
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
            gap: 10,
          }}
        >
          <View style={{ backgroundColor: 'rgba(0,0,0,0.45)', padding: 10, borderRadius: 12 }}>
            <Text style={{ color: FG, fontSize: 13, fontWeight: '700' }}>Cat Classifier</Text>
            <Text style={{ color: ready ? '#b5ffb5' : FG_MUTED, marginTop: 2, fontSize: 12 }}>
              {status}
            </Text>
            {cameraReady ? null : (
              <Text style={{ color: FG_MUTED, marginTop: 4, fontSize: 11 }}>
                Podgląd kamery jest aktywny.
              </Text>
            )}
          </View>

          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            {busy && (
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                <ActivityIndicator size="small" />
                <Text style={{ color: FG, fontSize: 12 }}>Klasyfikuję…</Text>
              </View>
            )}
            <Pressable
              onPress={() => {
                void handleReloadModel();
              }}
              disabled={busy}
              style={{
                backgroundColor: 'rgba(0,0,0,0.65)',
                paddingVertical: 10,
                paddingHorizontal: 12,
                borderRadius: 12,
                alignItems: 'center',
                borderWidth: 1,
                borderColor: 'rgba(255,255,255,0.08)',
                opacity: busy ? 0.75 : 1,
              }}
            >
              <Text style={{ color: FG, fontSize: 14, fontWeight: '600' }}>🔁 Przeładuj model</Text>
              <Text style={{ color: FG_MUTED, fontSize: 11 }}>kamerka pauzuje na chwilę</Text>
            </Pressable>
          </View>
        </View>

        {/* Bottom overlay: results + reload button */}
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
          <View style={{ backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 12, padding: 12, gap: 10 }}>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
              <Text style={{ color: FG, fontSize: 14, fontWeight: '600' }}>Zoom</Text>
              <Text style={{ color: FG_MUTED, fontSize: 12 }}>{Math.round(zoom * 100)}%</Text>
            </View>
            <PanGestureHandler
              onGestureEvent={handleSliderGesture}
              onHandlerStateChange={handleSliderStateChange}
              minDist={0}
            >
              <View
                onLayout={(event) => {
                  sliderWidthRef.current = event.nativeEvent.layout.width;
                }}
                style={{
                  height: 28,
                  borderRadius: 14,
                  backgroundColor: 'rgba(255,255,255,0.08)',
                  overflow: 'hidden',
                  justifyContent: 'center',
                }}
              >
                <View
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${zoom * 100}%`,
                    backgroundColor: 'rgba(255,255,255,0.25)',
                  }}
                />
                <View
                  style={{
                    position: 'absolute',
                    left: `${zoom * 100}%`,
                    marginLeft: -8,
                    width: 16,
                    height: 16,
                    borderRadius: 8,
                    backgroundColor: '#fff',
                    borderWidth: 1,
                    borderColor: 'rgba(0,0,0,0.35)',
                  }}
                />
              </View>
            </PanGestureHandler>
          </View>

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
                      paddingVertical: 4,
                    }}
                  >
                    <Text style={{ color: FG, fontSize: 13, fontWeight: '600' }}>{item.label}</Text>
                    <Text style={{ color: FG_MUTED, fontSize: 12 }}>{(item.p * 100).toFixed(1)}%</Text>
                  </View>
                )}
              />
            </View>
          )}
        </View>
        </View>
      </SafeAreaView>
    </GestureHandlerRootView>
  );
}
