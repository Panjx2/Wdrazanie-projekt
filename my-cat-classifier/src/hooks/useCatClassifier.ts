import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

import { chwFromBase64JPEG } from '../utils/preprocess';
import { decodeYoloOutput, type YoloDetection } from '../utils/postprocess';
import { CAMERA_STATUS_INTERVAL_MS, USE_PNG_LOSSLESS } from '../config/constants';
import { MODEL, MODEL_KIND } from '../config/modelConfig';

const labels = MODEL.labels;

const PROB_ALIASES = ['prob', 'probs', 'probabilities', 'softmax'];
const LOGIT_ALIASES = ['logits', 'output'];

const MODEL_BASENAME = MODEL.basename;
const ONNX_ASSET = MODEL.onnxAsset;
const ONNX_DATA_ASSET = MODEL.onnxDataAsset;

async function prepareOnnxWithExternalData(
  logFn: (...args: unknown[]) => void = console.log,
  warnFn: (...args: unknown[]) => void = console.warn
) {
  if (!ONNX_ASSET) {
    throw new Error(
      'Brak pliku modelu. Dodaj swój plik ONNX do assets/models i uzupełnij config w src/config/modelConfig.ts'
    );
  }

  const dir = FileSystem.cacheDirectory + 'ort-model/';
  try {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  } catch (e) {
    // directory already exists
  }

  const assetsToLoad = [ONNX_ASSET, ONNX_DATA_ASSET].filter(Boolean) as number[];
  logFn('[Model] Ładowanie zasobów ONNX…', assetsToLoad);
  let loaded: Asset[] = [];
  try {
    loaded = await Asset.loadAsync(assetsToLoad);
  } catch (e: any) {
    warnFn('[Model] Asset.loadAsync error, spróbuję pobrać ręcznie:', e?.message || e);
    const fallbackAssets = assetsToLoad.map(moduleId => Asset.fromModule(moduleId));
    for (const asset of fallbackAssets) {
      const uri = asset.localUri ?? asset.uri;
      if (!uri) {
        warnFn('[Model] Brak URI assetu — przerwane pobieranie awaryjne');
        throw e;
      }
      const dst = `${dir}${asset.name}.${asset.type ?? 'bin'}`;
      warnFn('[Model] Pobieram asset przez FileSystem.downloadAsync', { uri, dst });
      await FileSystem.downloadAsync(uri, dst);
      loaded.push({ ...asset, localUri: dst } as Asset);
    }
  }

  loaded.forEach((asset, idx) => {
    logFn('[Model] Załadowano asset', idx, {
      name: asset.name,
      uri: asset.uri,
      localUri: asset.localUri,
      type: asset.type,
    });
  });
  const [onnxAsset, dataAsset] = loaded;

  const modelDst = `${dir}${MODEL_BASENAME}.onnx`;
  await FileSystem.copyAsync({ from: onnxAsset.localUri!, to: modelDst });
  const modelInfo = await FileSystem.getInfoAsync(modelDst, { size: true });
  logFn('[Model] Skopiowano .onnx do cache', modelDst, modelInfo);

  if (dataAsset?.localUri) {
    const dataDst = `${dir}${MODEL_BASENAME}.onnx.data`;
    await FileSystem.copyAsync({ from: dataAsset.localUri, to: dataDst });
    const dataInfo = await FileSystem.getInfoAsync(dataDst, { size: true });
    logFn('[Model] Skopiowano .onnx.data do cache', dataDst, dataInfo);
  }

  return modelDst;
}

const topK = (probs: number[], k = 3) =>
  probs
    .map((p, i) => ({ i, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, Math.min(k, probs.length));

type ResizeResult = ImageManipulator.ImageResult & { base64: string };

type ClassifyOptions = {
  silent?: boolean;
};

type ClassifyResult =
  | { kind: 'classification'; topK: Array<{ label: string; p: number }> }
  | { kind: 'yolo'; detections: YoloDetection[] };

type CatClassifierHook = {
  status: string;
  updateStatus: (value: string) => void;
  busy: boolean;
  ready: boolean;
  previewUri: string | null;
  setPreviewUri: (uri: string | null) => void;
  probTopK: Array<{ label: string; p: number }>;
  detections: YoloDetection[];
  pickImage: () => Promise<void>;
  reloadModel: () => Promise<void>;
  classifyBase64: (
    jpegBase64: string,
    options?: ClassifyOptions
  ) => Promise<ClassifyResult | null>;
  resizeToModelBase64: (uri: string, context: 'gallery' | 'camera') => Promise<ResizeResult>;
  resetSilentStatus: () => void;
  log: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  err: (...args: unknown[]) => void;
};

export function useCatClassifier(): CatClassifierHook {
  const [status, setStatus] = useState('⏳ Inicjalizacja…');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [probTopK, setProbTopK] = useState<Array<{ label: string; p: number }>>([]);
  const [detections, setDetections] = useState<YoloDetection[]>([]);

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const lastSilentStatusRef = useRef<{ label: string; timestamp: number }>({
    label: '',
    timestamp: 0,
  });

  const log = useCallback((...args: unknown[]) => console.log('[CatApp]', ...args), []);
  const warn = useCallback((...args: unknown[]) => console.warn('[CatApp]', ...args), []);
  const err = useCallback((...args: unknown[]) => console.error('[CatApp]', ...args), []);

  const lastSummaryLogRef = useRef<{ message: string; timestamp: number }>({
    message: '',
    timestamp: 0,
  });

  const logResultSummary = useCallback(
    (message: string) => {
      const now = Date.now();
      if (
        message !== lastSummaryLogRef.current.message ||
        now - lastSummaryLogRef.current.timestamp > 1500
      ) {
        log('[Wynik]', message);
        lastSummaryLogRef.current = { message, timestamp: now };
      }
    },
    [log]
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
        setStatus('🤖 Klasyfikuję…');
      }

      try {
        const chw = chwFromBase64JPEG(
          jpegBase64,
          MODEL.inputWidth,
          MODEL.inputHeight,
          MODEL.mean,
          MODEL.std,
          MODEL.useBgr
        );

        const inputName = session.inputNames?.[0] ?? 'input';
        const tensor = new ort.Tensor('float32', chw, [
          1,
          3,
          MODEL.inputHeight,
          MODEL.inputWidth,
        ]);

        const outputMap = await session.run({ [inputName]: tensor });
        const keys = Object.keys(outputMap);

        const outName =
          PROB_ALIASES.find(key => keys.includes(key)) ??
          LOGIT_ALIASES.find(key => keys.includes(key)) ??
          keys[0];

        const outT = outputMap[outName];
        if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);
        const data = outT.data as Float32Array;

        if (MODEL.kind === 'classification') {
          let probs: number[];
          if (PROB_ALIASES.includes(outName)) {
            probs = Array.from(data);
          } else {
            let max = -Infinity;
            for (let i = 0; i < data.length; i += 1) if (data[i] > max) max = data[i];
            const exps = new Float32Array(data.length);
            let sum = 0;
            for (let i = 0; i < data.length; i += 1) {
              const value = Math.exp(data[i] - max);
              exps[i] = value;
              sum += value;
            }
            probs = Array.from(exps, value => value / (sum || 1));
          }

          const rawTop = topK(probs, 3).map(({ i, p }) => ({
            label: labels[i] ?? `cls_${i}`,
            p,
          }));

          const top = rawTop.map((item, idx) =>
            idx === 0 && item.p < 0.3 ? { ...item, label: 'Unknown' } : item
          );

          setDetections([]);
          setProbTopK(top);
          if (!silent) {
            setStatus('✅ Gotowe');
          } else if (top.length > 0) {
            const best = top[0];
            const now = Date.now();
            const shouldUpdate =
              best.label !== lastSilentStatusRef.current.label ||
              now - lastSilentStatusRef.current.timestamp > CAMERA_STATUS_INTERVAL_MS;
            if (shouldUpdate) {
              setStatus(`📸 Kamera: ${best.label} ${(best.p * 100).toFixed(1)}%`);
              lastSilentStatusRef.current = { label: best.label, timestamp: now };
            }
          }
          const summary = top.length
            ? `Classification • ${top
                .map(t => `${t.label} ${(t.p * 100).toFixed(1)}%`)
                .join(', ')}`
            : 'Classification • no prediction';
          logResultSummary(summary);
          return { kind: 'classification', topK: top } satisfies ClassifyResult;
        }

        const detectionsDecoded = decodeYoloOutput(data, {
          numClasses: labels.length,
          inputWidth: MODEL.inputWidth,
          inputHeight: MODEL.inputHeight,
          confThreshold: MODEL.confThreshold,
          iouThreshold: MODEL.iouThreshold,
          maxDetections: MODEL.maxDetections,
          labels,
          layout: MODEL.outputLayout,
        });

        setProbTopK([]);
        setDetections(detectionsDecoded);

        const yoloSummary = detectionsDecoded.length
          ? `YOLO • ${detectionsDecoded.length} boxes (best ${detectionsDecoded[0].label} ${(detectionsDecoded[0].score * 100).toFixed(
              1
            )}%)`
          : 'YOLO • brak detekcji';
        logResultSummary(yoloSummary);

        if (!silent) {
          setStatus('✅ Gotowe');
        } else {
          const best = detectionsDecoded[0];
          const now = Date.now();
          const label = best
            ? `${best.label} ${(best.score * 100).toFixed(1)}%`
            : 'brak detekcji';
          const shouldUpdate =
            label !== lastSilentStatusRef.current.label ||
            now - lastSilentStatusRef.current.timestamp > CAMERA_STATUS_INTERVAL_MS;
          if (shouldUpdate) {
            setStatus(`📸 Kamera: ${label}`);
            lastSilentStatusRef.current = { label, timestamp: now };
          }
        }

        logResultSummary(
          detectionsDecoded.length
            ? `YOLO detale • ${detectionsDecoded
                .slice(0, 3)
                .map(
                  det =>
                    `${det.label} ${(det.score * 100).toFixed(1)}% @ [${det.box.x1.toFixed(
                      2
                    )},${det.box.y1.toFixed(2)}]-[${det.box.x2.toFixed(2)},${det.box.y2.toFixed(2)}]`
                )
                .join(' | ')}`
            : 'YOLO detale • brak detekcji'
        );

        return { kind: 'yolo', detections: detectionsDecoded } satisfies ClassifyResult;
      } catch (e: any) {
        err('Błąd klasyfikacji:', e?.message || e);
        if (!silent) {
          setStatus('❌ Błąd klasyfikacji');
          Alert.alert('Inference error', String(e?.message || e));
        } else {
          setStatus('⚠️ Kamera: błąd klasyfikacji');
        }
        logResultSummary(`Błąd klasyfikacji: ${e?.message || e}`);
      } finally {
        if (!silent) {
          setBusy(false);
        }
      }
      return null;
    },
    [err, logResultSummary, warn]
  );

  const resizeToModelBase64 = useCallback<CatClassifierHook['resizeToModelBase64']>(
    async (uri, context) => {
      const outFormat = USE_PNG_LOSSLESS
        ? ImageManipulator.SaveFormat.PNG
        : ImageManipulator.SaveFormat.JPEG;
      try {
        const resized = await ImageManipulator.manipulateAsync(
          uri,
          [{ resize: { width: MODEL.inputWidth, height: MODEL.inputHeight } }],
          {
            compress: USE_PNG_LOSSLESS ? 1 : 0.95,
            format: outFormat,
            base64: true,
          }
        );
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

      setStatus(`🛠️ Resize ${MODEL.inputWidth}×${MODEL.inputHeight}…`);
      console.time('Resize+Base64');

      const resized = await resizeToModelBase64(uri, 'gallery');
      console.timeEnd('Resize+Base64');

      setPreviewUri(resized.uri);
      await classifyBase64(resized.base64);
    } catch (e: any) {
      err('Błąd obrazu:', e?.message || e);
      setStatus('❌ Błąd obrazu');
      Alert.alert('Image error', String(e?.message || e));
    }
  }, [classifyBase64, err, log, resizeToModelBase64]);

  const resetSilentStatus = useCallback(() => {
    lastSilentStatusRef.current = { label: '', timestamp: 0 };
  }, []);

  const loadModel = useCallback(async () => {
    try {
      setReady(false);
      setStatus('📦 Ładowanie modelu…');

      log('[Model] Rozpoczynam ładowanie modelu…');
      log('[Model] Konfiguracja', {
        kind: MODEL.kind,
        basename: MODEL.basename,
        input: `${MODEL.inputWidth}x${MODEL.inputHeight}`,
        confThreshold: (MODEL as any).confThreshold,
        iouThreshold: (MODEL as any).iouThreshold,
        maxDetections: (MODEL as any).maxDetections,
        outputLayout: (MODEL as any).outputLayout,
      });
      const modelPath = await prepareOnnxWithExternalData(log, warn);
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
      setStatus('✅ Model gotowy');
      setReady(true);
    } catch (e: any) {
      err('Błąd modelu:', e?.message || e);
      if (e?.stack) {
        warn('Stack trace:', e.stack);
      }
      setStatus('❌ Błąd modelu');
      Alert.alert('Model error', String(e?.message || e));
    }
    }, [err, log, warn]);

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
      probTopK,
      detections,
      pickImage,
      reloadModel,
      classifyBase64,
      resizeToModelBase64,
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
      probTopK,
      detections,
      pickImage,
      reloadModel,
      classifyBase64,
      resizeToModelBase64,
      resetSilentStatus,
      log,
      warn,
      err,
    ]
  );
}
