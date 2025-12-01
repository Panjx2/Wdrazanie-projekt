import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

import {
  chwFromBase64JPEG224,
  yoloLetterboxChwFromBase64,
  type LetterboxMeta,
} from '../utils/preprocess';
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
const EXPECTED_LABELS = labels.length;
const NOT_CAT_INDEX = labels.indexOf('Not cat');

const CLASSIFIER_INPUT_SIZE = 224;
const COCO_CAT_CLASS_ID = 15;

type Detection = {
  label: string;
  score: number;
  box: { x1: number; y1: number; x2: number; y2: number }; // współrzędne znormalizowane do 0..1 względem zdjęcia wejściowego
};

export type ResizeResult = ImageManipulator.ImageResult & { base64: string };

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
  classifyBase64: (resized: ResizeResult, options?: ClassifyOptions) => Promise<Detection[] | null>;
  resizeToModelInput: (uri: string, context: 'gallery' | 'camera') => Promise<ResizeResult>;
  resetSilentStatus: () => void;
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  err: (...args: unknown[]) => void;
};

type RawYoloShape = {
  format: 'raw';
  numBoxes: number;
  stride: number;
  classCount: number;
};

type NmsYoloShape = {
  format: 'nms';
  numBoxes: number;
  stride: number;
};

type YoloShape = RawYoloShape | NmsYoloShape;

const MODEL_BASENAME = 'yolo11';
const ONNX_ASSET = require('../../assets/models/yolo11.onnx');
const ONNX_DATA_ASSET = require('../../assets/models/yolo11.onnx.data');

const CLASSIFIER_MODEL_BASENAME = 'mobilenetv3_finetuned';
const CLASSIFIER_ONNX_ASSET = require('../../assets/models/mobilenetv3_finetuned.onnx');
const CLASSIFIER_ONNX_DATA_ASSET = require('../../assets/models/mobilenetv3_finetuned.onnx.data');

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
  if (dataAsset?.localUri) {
    const dataInfo = await FileSystem.getInfoAsync(dataAsset.localUri);
    if (dataInfo.exists && dataInfo.size && dataInfo.size > 0) {
      await FileSystem.copyAsync({ from: dataAsset.localUri, to: dataDst });
    } else {
      console.warn('[CatApp] Pomijam plik .onnx.data (brak lub pusty)');
    }
  }

  return modelDst;
}

async function prepareClassifierOnnx() {
  const [onnxAsset, dataAsset] = await Asset.loadAsync([
    CLASSIFIER_ONNX_ASSET,
    CLASSIFIER_ONNX_DATA_ASSET,
  ]);

  const dir = FileSystem.cacheDirectory + 'ort-classifier/';
  try {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  } catch (e) {
    // directory already exists
  }

  const modelDst = `${dir}${CLASSIFIER_MODEL_BASENAME}.onnx`;
  const dataDst = `${dir}${CLASSIFIER_MODEL_BASENAME}.onnx.data`;

  await FileSystem.copyAsync({ from: onnxAsset.localUri!, to: modelDst });
  if (dataAsset?.localUri) {
    const dataInfo = await FileSystem.getInfoAsync(dataAsset.localUri);
    if (dataInfo.exists && dataInfo.size && dataInfo.size > 0) {
      await FileSystem.copyAsync({ from: dataAsset.localUri, to: dataDst });
    } else {
      console.warn('[CatApp] Pomijam plik klasyfikatora .onnx.data (brak lub pusty)');
    }
  }

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

function parseYoloShape(dims: ReadonlyArray<number> | undefined): YoloShape {
  const outDims = Array.from(dims ?? []);
  if (outDims.length === 3 && outDims[2] >= 6) {
    // [batch, num_boxes, 6+]
    const [batch, numBoxes, stride] = outDims;
    if (batch !== 1) console.warn('[CatApp] Niespodziewany batch > 1 w wyjściu YOLO');
    if (stride === 6) return { format: 'nms', numBoxes, stride };
    return { format: 'raw', numBoxes, stride, classCount: stride - 5 };
  }
  if (outDims.length === 2 && outDims[1] >= 6) {
    // [num_boxes, 6+] fallback
    const [numBoxes, stride] = outDims;
    if (stride === 6) return { format: 'nms', numBoxes, stride };
    return { format: 'raw', numBoxes, stride, classCount: stride - 5 };
  }

  throw new Error(`Nieznany kształt wyjścia YOLO: ${outDims.join('x') || 'brak dims'}`);
}

export function useCatClassifier(): CatClassifierHook {
  const [status, setStatus] = useState('⏳ Inicjalizacja…');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classifierSessionRef = useRef<ort.InferenceSession | null>(null);
  const lastSilentStatusRef = useRef<{ label: string; timestamp: number }>({ label: '', timestamp: 0 });

  const log = useCallback((...args: unknown[]) => console.log('[CatApp]', ...args), []);
  const warn = useCallback((...args: unknown[]) => console.warn('[CatApp]', ...args), []);
  const err = useCallback((...args: unknown[]) => console.error('[CatApp]', ...args), []);

  const validateModelShape = useCallback(async () => {
    const session = sessionRef.current;
    if (!session) return;

    const inputName = session.inputNames?.[0] ?? 'images';
    const tensor = new ort.Tensor('float32', new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE), [
      1,
      3,
      YOLO_INPUT_SIZE,
      YOLO_INPUT_SIZE,
    ]);

    const outputMap = await session.run({ [inputName]: tensor });
    const keys = Object.keys(outputMap);
    const outName = session.outputNames?.[0] ?? keys[0];
    const outT = outputMap[outName];
    if (!outT?.dims) throw new Error(`Brak wymiarów wyjścia modelu "${outName}" podczas walidacji.`);

    const shape = parseYoloShape(outT.dims);
    const shapeMsg =
      shape.format === 'nms'
        ? `wyjście NMS (xyxy, score, class), stride ${shape.stride}, boxów ${shape.numBoxes}`
        : `${shape.classCount} klas, stride ${shape.stride}, boxów ${shape.numBoxes}`;

    log(`Walidacja modelu: ${shapeMsg}`);
    log('Aktywne etykiety YOLO: COCO (używany tylko do detekcji kota)');
  }, [log, warn]);

  const validateClassifierShape = useCallback(async () => {
    const session = classifierSessionRef.current;
    if (!session) return;

    const inputName = session.inputNames?.[0] ?? 'input';
    const tensor = new ort.Tensor('float32', new Float32Array(3 * CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE), [
      1,
      3,
      CLASSIFIER_INPUT_SIZE,
      CLASSIFIER_INPUT_SIZE,
    ]);

    const outputMap = await session.run({ [inputName]: tensor });
    const keys = Object.keys(outputMap);
    const outName = session.outputNames?.[0] ?? keys[0];
    const outT = outputMap[outName];
    const logits = outT?.data as Float32Array | undefined;
    if (!logits?.length) throw new Error('Klasyfikator zwrócił puste dane podczas walidacji.');
    if (logits.length % EXPECTED_LABELS !== 0) {
      throw new Error(
        `Klasyfikator zwraca ${logits.length} wartości, ale labels.json definiuje ${EXPECTED_LABELS} klas. ` +
          'Upewnij się, że eksport MobileNetV3 odpowiada temu zestawowi etykiet.'
      );
    }

    log(`Walidacja klasyfikatora: ${EXPECTED_LABELS} klas, output "${outName}" (${logits.length} wartości)`);
  }, [log, warn]);

  const decodeYoloOutput = useCallback(
    (data: Float32Array | number[], dims: ReadonlyArray<number> | undefined, meta: LetterboxMeta) => {
      const shape = parseYoloShape(dims);

      const boxes: Detection[] = [];

      for (let i = 0; i < shape.numBoxes; i += 1) {
        const base = i * shape.stride;

        if (shape.format === 'nms') {
          const score = Number(data[base + 4]);
          if (score < YOLO_CONF_THRESHOLD) continue;

          const cls = Math.round(Number(data[base + 5]));
          if (cls !== COCO_CAT_CLASS_ID) continue;

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
            label: 'cat',
            score,
            box: { x1: nx1, y1: ny1, x2: nx2, y2: ny2 },
          });
        } else {
          const objScore = Number(data[base + 4]);
          if (objScore < YOLO_CONF_THRESHOLD) continue;

          if (shape.classCount <= COCO_CAT_CLASS_ID) continue;

          const clsScore = Number(data[base + 5 + COCO_CAT_CLASS_ID]);
          const score = objScore * clsScore;
          if (score < YOLO_CONF_THRESHOLD) continue;

          const cx = Number(data[base]);
          const cy = Number(data[base + 1]);
          const w = Number(data[base + 2]);
          const h = Number(data[base + 3]);

          const x1 = cx - w / 2;
          const y1 = cy - h / 2;
          const x2 = cx + w / 2;
          const y2 = cy + h / 2;

          const invRatio = 1 / meta.ratio;
          const nx1 = Math.max(0, Math.min(1, (x1 - meta.pad.x) * invRatio / meta.origSize.width));
          const ny1 = Math.max(0, Math.min(1, (y1 - meta.pad.y) * invRatio / meta.origSize.height));
          const nx2 = Math.max(0, Math.min(1, (x2 - meta.pad.x) * invRatio / meta.origSize.width));
          const ny2 = Math.max(0, Math.min(1, (y2 - meta.pad.y) * invRatio / meta.origSize.height));

          boxes.push({
            label: 'cat',
            score,
            box: { x1: nx1, y1: ny1, x2: nx2, y2: ny2 },
          });
        }
      }

      return applyNms(boxes);
    },
    []
  );

  const pickLargestBox = useCallback((list: Detection[]) => {
    let best: Detection | null = null;
    let bestArea = -Infinity;

    for (const det of list) {
      const area = Math.max(0, det.box.x2 - det.box.x1) * Math.max(0, det.box.y2 - det.box.y1);
      if (area > bestArea) {
        bestArea = area;
        best = det;
      }
    }

    return best;
  }, []);

  const cropToClassifierInput = useCallback(
    async (resized: ResizeResult, cropBox: Detection['box'] | null) => {
      const width = resized.width ?? YOLO_INPUT_SIZE;
      const height = resized.height ?? YOLO_INPUT_SIZE;

      const box = cropBox ?? { x1: 0, y1: 0, x2: 1, y2: 1 };
      const originX = Math.max(0, Math.floor(box.x1 * width));
      const originY = Math.max(0, Math.floor(box.y1 * height));
      const cropW = Math.max(1, Math.min(width - originX, Math.round((box.x2 - box.x1) * width)));
      const cropH = Math.max(1, Math.min(height - originY, Math.round((box.y2 - box.y1) * height)));

      const outFormat = USE_PNG_LOSSLESS
        ? ImageManipulator.SaveFormat.PNG
        : ImageManipulator.SaveFormat.JPEG;

      const cropped = await ImageManipulator.manipulateAsync(
        resized.uri,
        [
          { crop: { originX, originY, width: cropW, height: cropH } },
          { resize: { width: CLASSIFIER_INPUT_SIZE, height: CLASSIFIER_INPUT_SIZE } },
        ],
        { base64: true, format: outFormat, compress: USE_PNG_LOSSLESS ? 1 : 0.95 }
      );

      if (!cropped.base64) {
        throw new Error('Brak base64 po krojeniu do klasyfikatora');
      }

      return cropped.base64;
    },
    []
  );

  const runClassifier = useCallback(
    async (cropBase64: string, { excludeNotCat = false } = {}) => {
      const session = classifierSessionRef.current;
      if (!session) throw new Error('Sesja klasyfikatora niegotowa');

      const chw = chwFromBase64JPEG224(cropBase64);
      const inputName = session.inputNames?.[0] ?? 'input';
      const tensor = new ort.Tensor('float32', chw, [1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE]);

      const outputMap = await session.run({ [inputName]: tensor });
      const keys = Object.keys(outputMap);
      const outName = session.outputNames?.[0] ?? keys[0];
      const logitsT = outputMap[outName];
      const logits = logitsT?.data as Float32Array | undefined;
      if (!logits?.length) throw new Error('Puste wyjście klasyfikatora');

      const candidateIndices = logits.map((_, idx) => idx).filter(idx => {
        if (!excludeNotCat) return true;
        if (NOT_CAT_INDEX === -1) return true;
        return idx !== NOT_CAT_INDEX;
      });

      const filteredLogits = candidateIndices.map(idx => logits[idx]);

      const max = Math.max(...filteredLogits);
      const exps = filteredLogits.map(v => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(v => v / sum);

      let bestIdx = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < probs.length; i += 1) {
        if (probs[i] > bestScore) {
          bestScore = probs[i];
          bestIdx = candidateIndices[i];
        }
      }

      const label = labels[bestIdx] ?? `cls_${bestIdx}`;
      const score = Number.isFinite(bestScore) ? bestScore : 0;

      return { label, score };
    },
    []
  );

  const classifyBase64 = useCallback(
    async (resized: ResizeResult, { silent = false }: ClassifyOptions = {}) => {
      const session = sessionRef.current;
      const classifierSession = classifierSessionRef.current;
      if (!session || !classifierSession) {
        warn('Sesje ORT niegotowe');
        setStatus('⏳ Model się ładuje…');
        return null;
      }

      if (!silent) {
        setBusy(true);
        setStatus('🤖 Detekcja…');
      }

      try {
        const { chw, meta } = yoloLetterboxChwFromBase64(resized.base64, YOLO_INPUT_SIZE, USE_BGR);

        const inputName = session.inputNames?.[0] ?? 'images';
        const tensor = new ort.Tensor('float32', chw, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);

        const outputMap = await session.run({ [inputName]: tensor });
        const keys = Object.keys(outputMap);
        const outName = session.outputNames?.[0] ?? keys[0];
        const outT = outputMap[outName];
        if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);

        const parsed = decodeYoloOutput(outT.data as Float32Array, outT.dims, meta);
        const limited = parsed.slice(0, YOLO_MAX_DETECTIONS);
        const cropSource = pickLargestBox(limited);

        const cropBase64 = await cropToClassifierInput(resized, cropSource?.box ?? null);
        const classification = await runClassifier(cropBase64, { excludeNotCat: Boolean(cropSource) });

        const output: Detection[] = [
          {
            label: classification.label,
            score: classification.score,
            box: cropSource?.box ?? { x1: 0, y1: 0, x2: 1, y2: 1 },
          },
        ];

        setDetections(output);
        if (!silent) {
          setStatus('✅ Gotowe');
        } else if (output.length > 0) {
          const best = output[0];
          const now = Date.now();
          const shouldUpdate =
            best.label !== lastSilentStatusRef.current.label ||
            now - lastSilentStatusRef.current.timestamp > CAMERA_STATUS_INTERVAL_MS;
          if (shouldUpdate) {
            setStatus(`📸 Kamera: ${best.label} ${(best.score * 100).toFixed(1)}%`);
            lastSilentStatusRef.current = { label: best.label, timestamp: now };
          }
        }
        log(
          'Detekcja + klasyfikacja:',
          output.map(t => `${t.label}: ${(t.score * 100).toFixed(1)}%`).join(', '),
          cropSource ? '(z kadrowaniem YOLO COCO cat)' : '(bez detekcji kota, pełny kadr)'
        );
        return output;
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
    [
      cropToClassifierInput,
      decodeYoloOutput,
      err,
      log,
      pickLargestBox,
      runClassifier,
      warn,
    ]
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
      await classifyBase64(resized);
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
      log('Model YOLO local path:', modelPath);

      const classifierPath = await prepareClassifierOnnx();
      log('Model klasyfikatora local path:', classifierPath);

      setStatus('🧠 Tworzenie sesji ORT…');
      const executionProviders = Platform.select({
        android: ['xnnpack', 'cpu'],
        ios: ['coreml', 'cpu'],
        default: ['cpu'],
      });
      sessionRef.current = await ort.InferenceSession.create(modelPath, {
        executionProviders,
      });
      classifierSessionRef.current = await ort.InferenceSession.create(classifierPath, {
        executionProviders,
      });

      await validateModelShape();
      await validateClassifierShape();

      log('YOLO input names:', sessionRef.current.inputNames ?? []);
      log('YOLO output names:', sessionRef.current.outputNames ?? []);
      log('Classifier input names:', classifierSessionRef.current.inputNames ?? []);
      log('Classifier output names:', classifierSessionRef.current.outputNames ?? []);
      setStatus('✅ Gotowe');
      setReady(true);
    } catch (e: any) {
      err('Błąd ładowania modelu:', e?.message || e);
      setStatus('❌ Błąd ładowania modelu');
      Alert.alert('Model error', String(e?.message || e));
    }
  }, [err, log, validateClassifierShape, validateModelShape]);

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
