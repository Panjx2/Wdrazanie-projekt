import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react';
import { Alert } from 'react-native';
import { CameraView, useCameraPermissions, type CameraPermissionResponse } from 'expo-camera';

import { CAMERA_CAPTURE_INTERVAL_MS, CAMERA_QUALITY } from '../config/constants';

interface UseCameraLoopParams {
  ready: boolean;
  updateStatus: (value: string) => void;
  classifyBase64: (
    jpegBase64: string,
    options?: { silent?: boolean }
  ) => Promise<Array<{ label: string; p: number }> | null>;
  resizeTo224Base64: (uri: string, context: 'gallery' | 'camera') => Promise<{ base64: string }>;
  resetSilentStatus: () => void;
  clearPreview?: () => void;
  warn: (...args: unknown[]) => void;
  err: (...args: unknown[]) => void;
}

interface UseCameraLoopResult {
  cameraActive: boolean;
  cameraReady: boolean;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  toggleCamera: () => Promise<void>;
  cameraRef: MutableRefObject<CameraView | null>;
  permission: CameraPermissionResponse | undefined;
  handleCameraReady: () => void;
  handleMountError: (event: { nativeEvent?: { message?: string } }) => void;
}

export function useCameraLoop({
  ready,
  updateStatus,
  classifyBase64,
  resizeTo224Base64,
  resetSilentStatus,
  clearPreview,
  warn,
  err,
}: UseCameraLoopParams): UseCameraLoopResult {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const cameraRef = useRef<CameraView | null>(null);
  const captureTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const takingPictureRef = useRef(false);

  const stopCameraCapture = useCallback(() => {
    if (captureTimeoutRef.current) {
      clearTimeout(captureTimeoutRef.current);
      captureTimeoutRef.current = null;
    }
    takingPictureRef.current = false;
    resetSilentStatus();
    setCameraReady(false);
  }, [resetSilentStatus]);

  const stopCamera = useCallback(() => {
    if (!cameraActive) return;
    stopCameraCapture();
    setCameraActive(false);
    if (ready) {
      updateStatus('✅ Gotowe');
    }
  }, [cameraActive, ready, stopCameraCapture, updateStatus]);

  const toggleCamera = useCallback(async () => {
    if (cameraActive) {
      stopCamera();
    } else {
      await startCamera();
    }
  }, [cameraActive, startCamera, stopCamera]);

  const captureFrame = useCallback(async () => {
    if (!ready || !cameraActive || !cameraReady) return;
    const camera = cameraRef.current;
    if (!camera || takingPictureRef.current) return;
    takingPictureRef.current = true;
    try {
      const photo = await camera.takePictureAsync({
        base64: false,
        quality: CAMERA_QUALITY,
        skipProcessing: true,
      });
      if (photo?.uri) {
        const resized = await resizeTo224Base64(photo.uri, 'camera');
        await classifyBase64(resized.base64, { silent: true });
      }
    } catch (e) {
      warn('Kamera: błąd przechwytywania klatki', (e as any)?.message || e);
    } finally {
      takingPictureRef.current = false;
    }
  }, [cameraActive, cameraReady, classifyBase64, ready, resizeTo224Base64, warn]);

  const startCamera = useCallback(async () => {
    if (cameraActive) return;

    if (!ready) {
      updateStatus('⏳ Model się ładuje…');
      return;
    }

    try {
      const permState = permission ?? (await requestPermission());
      if (!permState?.granted) {
        updateStatus('❌ Brak dostępu do kamery');
        Alert.alert('Camera access', 'Zezwól na dostęp do kamery, aby wykonywać klasyfikację.');
        return;
      }
      clearPreview?.();
      setCameraReady(false);
      setCameraActive(true);
      updateStatus('📸 Uruchamianie kamery…');
    } catch (e) {
      err('Błąd kamery:', (e as any)?.message || e);
      updateStatus('❌ Błąd kamery');
      Alert.alert('Camera error', String((e as any)?.message || e));
    }
  }, [cameraActive, clearPreview, err, permission, ready, requestPermission, updateStatus]);

  const handleCameraReady = useCallback(() => {
    setCameraReady(true);
    updateStatus('📸 Kamera gotowa');
  }, [updateStatus]);

  const handleMountError = useCallback(
    (event: { nativeEvent?: { message?: string } }) => {
      const message = event?.nativeEvent?.message || 'Nie udało się uruchomić kamery';
      err('Camera mount error:', message);
      updateStatus('❌ Błąd kamery');
      Alert.alert('Camera error', message);
      setCameraActive(false);
    },
    [err, updateStatus]
  );

  useEffect(() => () => {
    stopCamera();
  }, [stopCamera]);

  useEffect(() => {
    if (!cameraActive || !cameraReady || !ready) {
      stopCameraCapture();
      if (cameraActive && ready) {
        updateStatus('📸 Oczekiwanie na kamerę…');
      }
      return;
    }

    updateStatus('📸 Kamera aktywna');

    let cancelled = false;

    const loop = async () => {
      if (cancelled) return;
      const startedAt = Date.now();
      await captureFrame();
      if (cancelled) return;
      const elapsed = Date.now() - startedAt;
      const delay = Math.max(0, CAMERA_CAPTURE_INTERVAL_MS - elapsed);
      captureTimeoutRef.current = setTimeout(loop, delay);
    };

    void loop();

    return () => {
      cancelled = true;
      stopCameraCapture();
    };
  }, [cameraActive, cameraReady, captureFrame, ready, stopCameraCapture, updateStatus]);

  return useMemo(
    () => ({
      cameraActive,
      cameraReady,
      startCamera,
      stopCamera,
      toggleCamera,
      cameraRef,
      permission,
      handleCameraReady,
      handleMountError,
    }),
    [
      cameraActive,
      cameraReady,
      startCamera,
      stopCamera,
      toggleCamera,
      cameraRef,
      permission,
      handleCameraReady,
      handleMountError,
    ]
  );
}
