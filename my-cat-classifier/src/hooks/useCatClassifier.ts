import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Platform } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

import { chwFromBase64JPEG224 } from '../utils/preprocess';
import {
  CAMERA_STATUS_INTERVAL_MS,
  IMAGENET_MEAN,
  IMAGENET_STD,
  USE_BGR,
  USE_PNG_LOSSLESS,
} from '../config/constants';

const labels = require('../../assets/labels.json');

const PROB_ALIASES = ['prob', 'probs', 'probabilities', 'softmax'];
const LOGIT_ALIASES = ['logits', 'output'];

async function prepareOnnxWithExternalData() {
  const [onnxAsset, dataAsset] = await Asset.loadAsync([
    require('../../assets/models/mobilenetv2_finetuned.onnx'),
    require('../../assets/models/mobilenetv2_finetuned.onnx.data'),
  ]);

  const dir = FileSystem.cacheDirectory + 'ort-model/';
  try {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  } catch (e) {
    // directory already exists
  }

  const modelDst = dir + 'mobilenetv2_finetuned.onnx';
  const dataDst = dir + 'mobilenetv2_finetuned.onnx.data';

  await FileSystem.copyAsync({ from: onnxAsset.localUri!, to: modelDst });
  await FileSystem.copyAsync({ from: dataAsset.localUri!, to: dataDst });

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

type CatClassifierHook = {
  status: string;
  updateStatus: (value: string) => void;
  busy: boolean;
  ready: boolean;
  previewUri: string | null;
  setPreviewUri: (uri: string | null) => void;
  probTopK: Array<{ label: string; p: number }>;
  pickImage: () => Promise<void>;
  reloadModel: () => Promise<void>;
  classifyBase64: (jpegBase64: string, options?: ClassifyOptions) => Promise<
    Array<{ label: string; p: number }> | null
  >;
  resizeTo224Base64: (uri: string, context: 'gallery' | 'camera') => Promise<ResizeResult>;
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

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const lastSilentStatusRef = useRef<{ label: string; timestamp: number }>({ label: '', timestamp: 0 });

  const log = useCallback((...args: unknown[]) => console.log('[CatApp]', ...args), []);
  const warn = useCallback((...args: unknown[]) => console.warn('[CatApp]', ...args), []);
  const err = useCallback((...args: unknown[]) => console.error('[CatApp]', ...args), []);

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
        const chw = chwFromBase64JPEG224(jpegBase64, IMAGENET_MEAN, IMAGENET_STD, USE_BGR);

        {
          const size = 224 * 224;
          const mR = chw.slice(0, size).reduce((a, b) => a + b, 0) / size;
          const mG = chw.slice(size, 2 * size).reduce((a, b) => a + b, 0) / size;
          const mB = chw.slice(2 * size).reduce((a, b) => a + b, 0) / size;
          log('CHW means (≈0):', mR.toFixed(3), mG.toFixed(3), mB.toFixed(3));
        }

        const inputName = session.inputNames?.[0] ?? 'input';
        const tensor = new ort.Tensor('float32', chw, [1, 3, 224, 224]);

        const outputMap = await session.run({ [inputName]: tensor });
        const keys = Object.keys(outputMap);

        const outName =
          PROB_ALIASES.find(key => keys.includes(key)) ??
          LOGIT_ALIASES.find(key => keys.includes(key)) ??
          keys[0];

        const outT = outputMap[outName];
        if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);
        const data = outT.data as Float32Array;

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

        const top = topK(probs, 3).map(({ i, p }) => ({
          label: labels[i] ?? `cls_${i}`,
          p,
        }));

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
        log('TOP-3:', top.map(t => `${t.label}: ${(t.p * 100).toFixed(1)}%`).join(', '));
        return top;
      } catch (e: any) {
        err('Błąd klasyfikacji:', e?.message || e);
        if (!silent) {
          setStatus('❌ Błąd klasyfikacji');
          Alert.alert('Inference error', String(e?.message || e));
        } else {
          setStatus('⚠️ Kamera: błąd klasyfikacji');
        }
      } finally {
        if (!silent) {
          setBusy(false);
        }
      }
      return null;
    },
    [err, log, warn]
  );

  const resizeTo224Base64 = useCallback<CatClassifierHook['resizeTo224Base64']>(
    async (uri, context) => {
      const outFormat = USE_PNG_LOSSLESS
        ? ImageManipulator.SaveFormat.PNG
        : ImageManipulator.SaveFormat.JPEG;
      try {
        const resized = await ImageManipulator.manipulateAsync(uri, [{ resize: { width: 224, height: 224 } }], {
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

      setStatus('🛠️ Resize 224×224…');
      console.time('Resize+Base64');

      const resized = await resizeTo224Base64(uri, 'gallery');
      console.timeEnd('Resize+Base64');

      setPreviewUri(resized.uri);
      await classifyBase64(resized.base64);
    } catch (e: any) {
      err('Błąd obrazu:', e?.message || e);
      setStatus('❌ Błąd obrazu');
      Alert.alert('Image error', String(e?.message || e));
    }
  }, [classifyBase64, err, log, resizeTo224Base64]);

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
      probTopK,
      pickImage,
      reloadModel,
      classifyBase64,
      resizeTo224Base64,
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
      pickImage,
      reloadModel,
      classifyBase64,
      resizeTo224Base64,
      resetSilentStatus,
      log,
      warn,
      err,
    ]
  );
}
