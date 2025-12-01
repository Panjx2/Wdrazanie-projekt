import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react';
import { Alert } from 'react-native';
import { CameraView, useCameraPermissions, type CameraPermissionResponse } from 'expo-camera';

import { CAMERA_CAPTURE_INTERVAL_MS, CAMERA_QUALITY } from '../config/constants';
import type { ResizeResult } from './useCatClassifier';

interface UseCameraLoopParams {
  ready: boolean;
  updateStatus: (value: string) => void;
  classifyBase64: (
    resized: ResizeResult,
    options?: { silent?: boolean }
  ) => Promise<Array<{ label: string; score: number }> | null>;
  resizeToModelInput: (uri: string, context: 'gallery' | 'camera') => Promise<ResizeResult>;
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
  pauseCapture: () => void;
  resumeCapture: () => void;
  cameraRef: MutableRefObject<CameraView | null>;
  permission: CameraPermissionResponse | undefined;
  handleCameraReady: () => void;
  handleMountError: (event: { nativeEvent?: { message?: string } }) => void;
}

export function useCameraLoop({
  ready,
  updateStatus,
  classifyBase64,
  resizeToModelInput,
  resetSilentStatus,
  clearPreview,
  warn,
  err,
}: UseCameraLoopParams): UseCameraLoopResult {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [capturePaused, setCapturePaused] = useState(false);
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

  const pauseCapture = useCallback(() => {
    setCapturePaused(true);
    stopCameraCapture();
  }, [stopCameraCapture]);

  const resumeCapture = useCallback(() => {
    setCapturePaused(false);
  }, []);

  const captureFrame = useCallback(async () => {
    if (!ready || !cameraActive || !cameraReady || capturePaused) return;
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
        const resized = await resizeToModelInput(photo.uri, 'camera');
        await classifyBase64(resized, { silent: true });
      }
    } catch (e) {
      warn('Kamera: błąd przechwytywania klatki', (e as any)?.message || e);
    } finally {
      takingPictureRef.current = false;
    }
  }, [cameraActive, cameraReady, capturePaused, classifyBase64, ready, resizeToModelInput, warn]);

  const startCamera = useCallback(async () => {
    if (cameraActive) return;

    try {
      const permState = permission ?? (await requestPermission());
      if (!permState?.granted) {
        updateStatus('❌ Brak dostępu do kamery');
        Alert.alert('Camera access', 'Zezwól na dostęp do kamery, aby wykonywać klasyfikację.');
        return;
      }
      clearPreview?.();
      setCameraReady(false);
      setCapturePaused(false);
      setCameraActive(true);
      updateStatus(ready ? '📸 Uruchamianie kamery…' : '⏳ Model się ładuje, kamera startuje…');
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
    if (!cameraActive || !cameraReady || !ready || capturePaused) {
      stopCameraCapture();
      if (cameraActive && ready && !capturePaused) {
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
  }, [cameraActive, cameraReady, captureFrame, ready, capturePaused, stopCameraCapture, updateStatus]);

  return useMemo(
    () => ({
      cameraActive,
      cameraReady,
      startCamera,
      stopCamera,
      pauseCapture,
      resumeCapture,
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
      pauseCapture,
      resumeCapture,
      cameraRef,
      permission,
      handleCameraReady,
      handleMountError,
    ]
  );
}
