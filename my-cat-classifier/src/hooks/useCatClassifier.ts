import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

import { yoloLetterboxChwFromBase64, type LetterboxMeta } from '../utils/preprocess';
import {
  CAMERA_STATUS_INTERVAL_MS,
  USE_BGR,
  USE_PNG_LOSSLESS,
  YOLO_CONF_THRESHOLD,
  YOLO_INPUT_SIZE,
  YOLO_IOU_THRESHOLD,
  YOLO_MAX_DETECTIONS,
} from '../config/constants';

const labels = require('../../assets/labels.json');

type Detection = {
  label: string;
  score: number;
  box: { x1: number; y1: number; x2: number; y2: number }; // współrzędne znormalizowane do 0..1 względem zdjęcia wejściowego
};

type ResizeResult = ImageManipulator.ImageResult & { base64: string };

type ClassifyOptions = {
  silent?: boolean;
};

type CatClassifierHook = {
  status: string;
  updateStatus: (value: string) => void;
  busy: boolean;
  ready: boolean;
  previewUri: string | null;
  setPreviewUri: (uri: string | null) => void;
  detections: Detection[];
  pickImage: () => Promise<void>;
  reloadModel: () => Promise<void>;
  classifyBase64: (jpegBase64: string, options?: ClassifyOptions) => Promise<Detection[] | null>;
  resizeToModelInput: (uri: string, context: 'gallery' | 'camera') => Promise<ResizeResult>;
  resetSilentStatus: () => void;
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  err: (...args: unknown[]) => void;
};

const MODEL_BASENAME = 'yolo11';
const ONNX_ASSET = require('../../assets/models/yolo11.onnx');
const ONNX_DATA_ASSET = require('../../assets/models/yolo11.onnx.data');

async function prepareOnnxWithExternalData() {
  const [onnxAsset, dataAsset] = await Asset.loadAsync([
    ONNX_ASSET,
    ONNX_DATA_ASSET,
  ]);

  const dir = FileSystem.cacheDirectory + 'ort-model/';
  try {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  } catch (e) {
    // directory already exists
  }

  const modelDst = `${dir}${MODEL_BASENAME}.onnx`;
  const dataDst = `${dir}${MODEL_BASENAME}.onnx.data`;

  await FileSystem.copyAsync({ from: onnxAsset.localUri!, to: modelDst });
  await FileSystem.copyAsync({ from: dataAsset.localUri!, to: dataDst });

  return modelDst;
}

function boxIou(a: Detection['box'], b: Detection['box']) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union === 0 ? 0 : inter / union;
}

function applyNms(list: Detection[], iou = YOLO_IOU_THRESHOLD, max = YOLO_MAX_DETECTIONS) {
  const sorted = [...list].sort((a, b) => b.score - a.score);
  const picked: Detection[] = [];

  while (sorted.length && picked.length < max) {
    const current = sorted.shift()!;
    picked.push(current);
    const remaining: Detection[] = [];
    for (const det of sorted) {
      if (boxIou(current.box, det.box) < iou) {
        remaining.push(det);
      }
    }
    sorted.splice(0, sorted.length, ...remaining);
  }

  return picked;
}

export function useCatClassifier(): CatClassifierHook {
  const [status, setStatus] = useState('⏳ Inicjalizacja…');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const lastSilentStatusRef = useRef<{ label: string; timestamp: number }>({ label: '', timestamp: 0 });

  const log = useCallback((...args: unknown[]) => console.log('[CatApp]', ...args), []);
  const warn = useCallback((...args: unknown[]) => console.warn('[CatApp]', ...args), []);
  const err = useCallback((...args: unknown[]) => console.error('[CatApp]', ...args), []);

  const decodeYoloOutput = useCallback(
    (data: Float32Array | number[], dims: ReadonlyArray<number> | undefined, meta: LetterboxMeta) => {
      const outDims = Array.from(dims ?? []);
      const boxes: Detection[] = [];

      if (outDims.length === 3 && outDims[2] >= 6) {
        // [batch, num_boxes, 6+]
        const [batch, numBoxes, stride] = outDims;
        if (batch !== 1) warn('Niespodziewany batch > 1 w wyjściu YOLO');
        for (let i = 0; i < numBoxes; i += 1) {
          const base = i * stride;
          const score = Number(data[base + 4]);
          if (score < YOLO_CONF_THRESHOLD) continue;
          const cls = Math.round(Number(data[base + 5] ?? 0));

          const x1 = Number(data[base]);
          const y1 = Number(data[base + 1]);
          const x2 = Number(data[base + 2]);
          const y2 = Number(data[base + 3]);

          const invRatio = 1 / meta.ratio;
          const nx1 = Math.max(0, Math.min(1, (x1 - meta.pad.x) * invRatio / meta.origSize.width));
          const ny1 = Math.max(0, Math.min(1, (y1 - meta.pad.y) * invRatio / meta.origSize.height));
          const nx2 = Math.max(0, Math.min(1, (x2 - meta.pad.x) * invRatio / meta.origSize.width));
          const ny2 = Math.max(0, Math.min(1, (y2 - meta.pad.y) * invRatio / meta.origSize.height));

          boxes.push({
            label: labels[cls] ?? `cls_${cls}`,
            score,
            box: { x1: nx1, y1: ny1, x2: nx2, y2: ny2 },
          });
        }
      } else if (outDims.length === 2 && outDims[1] >= 6) {
        // [num_boxes, 6+] fallback
        const [numBoxes, stride] = outDims;
        for (let i = 0; i < numBoxes; i += 1) {
          const base = i * stride;
          const score = Number(data[base + 4]);
          if (score < YOLO_CONF_THRESHOLD) continue;
          const cls = Math.round(Number(data[base + 5] ?? 0));

          const x1 = Number(data[base]);
          const y1 = Number(data[base + 1]);
          const x2 = Number(data[base + 2]);
          const y2 = Number(data[base + 3]);

          const invRatio = 1 / meta.ratio;
          const nx1 = Math.max(0, Math.min(1, (x1 - meta.pad.x) * invRatio / meta.origSize.width));
          const ny1 = Math.max(0, Math.min(1, (y1 - meta.pad.y) * invRatio / meta.origSize.height));
          const nx2 = Math.max(0, Math.min(1, (x2 - meta.pad.x) * invRatio / meta.origSize.width));
          const ny2 = Math.max(0, Math.min(1, (y2 - meta.pad.y) * invRatio / meta.origSize.height));

          boxes.push({
            label: labels[cls] ?? `cls_${cls}`,
            score,
            box: { x1: nx1, y1: ny1, x2: nx2, y2: ny2 },
          });
        }
      } else {
        throw new Error(`Nieznany kształt wyjścia YOLO: ${outDims.join('x') || 'brak dims'}`);
      }

      return applyNms(boxes);
    },
    [warn]
  );

  const classifyBase64 = useCallback(
    async (jpegBase64: string, { silent = false }: ClassifyOptions = {}) => {
      const session = sessionRef.current;
      if (!session) {
        warn('Sesja ORT niegotowa');
        setStatus('⏳ Model się ładuje…');
        return null;
      }

      if (!silent) {
        setBusy(true);
        setStatus('🤖 Detekcja…');
      }

      try {
        const { chw, meta } = yoloLetterboxChwFromBase64(jpegBase64, YOLO_INPUT_SIZE, USE_BGR);

        const inputName = session.inputNames?.[0] ?? 'images';
        const tensor = new ort.Tensor('float32', chw, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);

        const outputMap = await session.run({ [inputName]: tensor });
        const keys = Object.keys(outputMap);
        const outName = session.outputNames?.[0] ?? keys[0];
        const outT = outputMap[outName];
        if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);

        const parsed = decodeYoloOutput(outT.data as Float32Array, outT.dims, meta);
        const limited = parsed.slice(0, YOLO_MAX_DETECTIONS);

        setDetections(limited);
        if (!silent) {
          setStatus('✅ Gotowe');
        } else if (limited.length > 0) {
          const best = limited[0];
          const now = Date.now();
          const shouldUpdate =
            best.label !== lastSilentStatusRef.current.label ||
            now - lastSilentStatusRef.current.timestamp > CAMERA_STATUS_INTERVAL_MS;
          if (shouldUpdate) {
            setStatus(`📸 Kamera: ${best.label} ${(best.score * 100).toFixed(1)}%`);
            lastSilentStatusRef.current = { label: best.label, timestamp: now };
          }
        }
        log('Detections:', limited.map(t => `${t.label}: ${(t.score * 100).toFixed(1)}%`).join(', '));
        return limited;
      } catch (e: any) {
        err('Błąd detekcji:', e?.message || e);
        if (!silent) {
          setStatus('❌ Błąd detekcji');
          Alert.alert('Inference error', String(e?.message || e));
        } else {
          setStatus('⚠️ Kamera: błąd detekcji');
        }
      } finally {
        if (!silent) {
          setBusy(false);
        }
      }
      return null;
    },
    [decodeYoloOutput, err, log, warn]
  );

  const resizeToModelInput = useCallback<CatClassifierHook['resizeToModelInput']>(
    async (uri, context) => {
      const outFormat = USE_PNG_LOSSLESS
        ? ImageManipulator.SaveFormat.PNG
        : ImageManipulator.SaveFormat.JPEG;
      try {
        const resized = await ImageManipulator.manipulateAsync(uri, [{ resize: { width: YOLO_INPUT_SIZE } }], {
          compress: USE_PNG_LOSSLESS ? 1 : 0.95,
          format: outFormat,
          base64: true,
        });
        if (!resized.base64) {
          throw new Error('Brak base64 po przetwarzaniu');
        }
        return resized as ResizeResult;
      } catch (e: any) {
        const message = e?.message || e;
        warn(`${context} resize error:`, message);
        throw e;
      }
    },
    [warn]
  );

  const pickImage = useCallback(async () => {
    try {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Brak uprawnień', 'Potrzebny dostęp do galerii.');
        return;
      }

      const mediaTypeEnum = (ImagePicker as any)?.MediaType;
      const mediaImages =
        (mediaTypeEnum?.Images as ImagePicker.MediaType | undefined) ??
        (mediaTypeEnum?.IMAGES as ImagePicker.MediaType | undefined) ??
        ('images' as ImagePicker.MediaType);

      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: mediaImages,
        quality: 1,
        base64: false,
      });
      if (res.canceled || !res.assets?.length) return;

      const uri = res.assets[0].uri;
      log('Wybrano:', uri);

      setStatus(`🛠️ Resize ${YOLO_INPUT_SIZE}×${YOLO_INPUT_SIZE}…`);
      console.time('Resize+Base64');

      const resized = await resizeToModelInput(uri, 'gallery');
      console.timeEnd('Resize+Base64');

      setPreviewUri(resized.uri);
      await classifyBase64(resized.base64);
    } catch (e: any) {
      err('Błąd obrazu:', e?.message || e);
      setStatus('❌ Błąd obrazu');
      Alert.alert('Image error', String(e?.message || e));
    }
  }, [classifyBase64, err, log, resizeToModelInput]);

  const resetSilentStatus = useCallback(() => {
    lastSilentStatusRef.current = { label: '', timestamp: 0 };
  }, []);

  const loadModel = useCallback(async () => {
    try {
      setReady(false);
      setStatus('📦 Ładowanie modelu…');

      const modelPath = await prepareOnnxWithExternalData();
      log('Model local path:', modelPath);

      setStatus('🧠 Tworzenie sesji ORT…');
      const executionProviders = Platform.select({
        android: ['xnnpack', 'cpu'],
        ios: ['coreml', 'cpu'],
        default: ['cpu'],
      });
      sessionRef.current = await ort.InferenceSession.create(modelPath, {
        executionProviders,
      });

      log('Input names:', sessionRef.current.inputNames ?? []);
      log('Output names:', sessionRef.current.outputNames ?? []);
      setStatus('✅ Gotowe');
      setReady(true);
    } catch (e: any) {
      err('Błąd ładowania modelu:', e?.message || e);
      setStatus('❌ Błąd ładowania modelu');
      Alert.alert('Model error', String(e?.message || e));
    }
  }, [err, log]);

  useEffect(() => {
    void loadModel();
  }, [loadModel]);

  const reloadModel = useCallback(async () => {
    await loadModel();
  }, [loadModel]);

  const updateStatus = useCallback((value: string) => {
    setStatus(value);
  }, []);

  return useMemo(
    () => ({
      status,
      updateStatus,
      busy,
      ready,
      previewUri,
      setPreviewUri,
      detections,
      pickImage,
      reloadModel,
      classifyBase64,
      resizeToModelInput,
      resetSilentStatus,
      log,
      warn,
      err,
    }),
    [
      status,
      updateStatus,
      busy,
      ready,
      previewUri,
      detections,
      pickImage,
      reloadModel,
      classifyBase64,
      resizeToModelInput,
      resetSilentStatus,
      log,
      warn,
      err,
    ]
  );
}
