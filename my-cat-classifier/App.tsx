// App.tsx — Detekcja kota (best.onnx) → Wyświetlenie z ramką → Klasyfikacja (MobileNet)
import 'react-native-reanimated';
import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  Pressable,
  Image,
  ActivityIndicator,
  FlatList,
  Alert,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
} from 'react-native';
import * as ort from 'onnxruntime-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

// Importy utils - upewnij się że ścieżki są poprawne
import { yoloPreprocess } from './src/utils/yoloPreprocess';
import { chwFromBase64JPEG224 } from './src/utils/preprocess';
import base64 from 'base64-js';
import jpeg from 'jpeg-js';

// labels.json (kolejność MUSI być taka sama jak w treningu)
const labels = require('./assets/labels.json');

// Normalizacja ImageNet
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

// UI kolory
const BG = '#0b0b0c';
const FG = '#ffffff';
const FG_MUTED = '#cfcfcf';
const ACCENT = '#1f6feb';
const BORDER = '#222';

// Ustawienia YOLO
const YOLO_INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const CAT_CLASS_ID = 0;

// Ustawienia
const USE_BGR = false;

// Konfiguracja backendu segmentacji
// Backend działa na: http://192.168.1.12:5000 (IP sieciowe) lub http://localhost:5000
// Dla emulatora Android użyj: 'http://10.0.2.2:5000'
// Dla urządzenia fizycznego użyj: 'http://192.168.1.12:5000' (IP z terminala)
// Dla iOS simulator użyj: 'http://localhost:5000'
const SEGMENTATION_BACKEND_URL = Platform.select({
  ios: __DEV__ ? 'http://localhost:5000' : 'http://192.168.1.12:5000',
  android: 'http://192.168.1.12:5000', // Używamy IP sieciowe dla Android (działa dla emulatora i urządzenia)
  default: 'http://localhost:5000',
});

// ---- Helper: przygotuj ścieżkę modelu best.onnx ----
async function prepareBestOnnxModel() {
  try {
    console.log('🔄 Preparing best.onnx model...');
    const onnxAsset = Asset.fromModule(require('./assets/models/best.onnx'));
    await onnxAsset.downloadAsync();

    if (!onnxAsset.localUri) {
      throw new Error('❌ Nie można załadować best.onnx!');
    }

    const dir = FileSystem.cacheDirectory + 'ort-model/';
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true }).catch(() => {});

    const modelDst = dir + 'best.onnx';
    await FileSystem.copyAsync({ from: onnxAsset.localUri, to: modelDst });

    console.log('✅ best.onnx prepared successfully');
    return modelDst;
  } catch (error) {
    console.error('❌ Error preparing best.onnx:', error);
    throw error;
  }
}

// ---- Helper: przygotuj ścieżkę modelu z plikiem external data ----
async function prepareOnnxWithExternalData() {
  try {
    console.log('🔄 Preparing MobileNet model with external data...');
    const [onnxAsset, dataAsset] = await Asset.loadAsync([
      require('./assets/models/mobilenetv2_finetuned.onnx'),
      require('./assets/models/mobilenetv2_finetuned.onnx.data'),
    ]);

    const dir = FileSystem.cacheDirectory + 'ort-model/';
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true }).catch(() => {});

    const modelDst = dir + 'mobilenetv2_finetuned.onnx';
    const dataDst  = dir + 'mobilenetv2_finetuned.onnx.data';

    await FileSystem.copyAsync({ from: onnxAsset.localUri, to: modelDst });
    await FileSystem.copyAsync({ from: dataAsset.localUri, to: dataDst });

    console.log('✅ MobileNet model prepared successfully');
    return modelDst;
  } catch (error) {
    console.error('❌ Error preparing MobileNet model:', error);
    throw error;
  }
}

// ---- Helper: narysuj bounding box na obrazie ----
async function drawBoundingBox(imageUri, box, color = '#00ff00', lineWidth = 3) {
  try {
    let [x1, y1, x2, y2] = box.map(v => Math.round(v));
   
    // Wczytaj obraz jako base64
    const imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    const bytes = base64.toByteArray(imageBase64);
    const decoded = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true });
    const { width, height, data } = decoded;

    // Ogranicz box do rozmiaru obrazu
    x1 = Math.max(0, Math.min(width - 1, x1));
    y1 = Math.max(0, Math.min(height - 1, y1));
    x2 = Math.max(0, Math.min(width - 1, x2));
    y2 = Math.max(0, Math.min(height - 1, y2));

    if (x2 <= x1 || y2 <= y1) {
      console.warn(`Nieprawidłowy box: [${x1}, ${y1}, ${x2}, ${y2}], obraz: ${width}x${height}`);
      return imageUri;
    }

    console.log(`Rysowanie ramki: box=[${x1}, ${y1}, ${x2}, ${y2}], obraz=${width}x${height}`);

    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    // Rysuj wszystkie 4 krawędzie
    for (let i = 0; i < lineWidth; i++) {
      // Górna krawędź
      for (let x = x1; x <= x2; x++) {
        const y = y1 + i;
        if (y >= 0 && y < height && x >= 0 && x < width) {
          const idx = (y * width + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
        }
      }
     
      // Dolna krawędź
      for (let x = x1; x <= x2; x++) {
        const y = y2 - i;
        if (y >= 0 && y < height && x >= 0 && x < width) {
          const idx = (y * width + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
        }
      }
     
      // Lewa krawędź
      for (let y = y1; y <= y2; y++) {
        const x = x1 + i;
        if (y >= 0 && y < height && x >= 0 && x < width) {
          const idx = (y * width + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
        }
      }
     
      // Prawa krawędź
      for (let y = y1; y <= y2; y++) {
        const x = x2 - i;
        if (y >= 0 && y < height && x >= 0 && x < width) {
          const idx = (y * width + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
        }
      }
    }

    // Zapisz jako JPEG
    const jpegResult = jpeg.encode(
      {
        data: data,
        width: width,
        height: height,
      },
      95
    );

    const jpegData = new Uint8Array(jpegResult.data);
    const jpegBase64 = base64.fromByteArray(jpegData);
    const dataUri = `data:image/jpeg;base64,${jpegBase64}`;

    const result = await ImageManipulator.manipulateAsync(
      dataUri,
      [],
      {
        compress: 0.95,
        format: ImageManipulator.SaveFormat.JPEG,
      }
    );

    return result.uri;
  } catch (error) {
    console.error('Błąd podczas rysowania ramki:', error);
    return imageUri;
  }
}

// Style
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BG,
  },
  scrollContent: {
    padding: 16,
    gap: 12,
    paddingBottom: 32,
  },
  title: {
    color: FG,
    fontSize: 28,
    fontWeight: '800',
  },
  status: {
    color: FG_MUTED,
    fontSize: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  primaryButton: {
    padding: 14,
    borderRadius: 16,
    alignItems: 'center',
    flex: 1,
  },
  secondaryButton: {
    padding: 14,
    borderRadius: 16,
    alignItems: 'center',
    width: 140,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: '700',
  },
  loadingContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  resultsContainer: {
    marginTop: 16,
  },
  resultsTitle: {
    color: FG,
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 12,
  },
  imageContainer: {
    marginBottom: 16,
  },
  imageLabel: {
    color: FG_MUTED,
    fontSize: 14,
    marginBottom: 6,
    fontWeight: '600',
  },
  image: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: BORDER,
    backgroundColor: '#1a1a1a',
  },
  classificationContainer: {
    padding: 16,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#6ee17a',
  },
  classificationTitle: {
    color: '#6ee17a',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  classificationItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  classificationText: {
    color: FG,
    fontSize: 16,
  },
});

export default function App() {
  const [status, setStatus] = useState('⏳ Inicjalizacja…');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [originalImageUri, setOriginalImageUri] = useState(null);
  const [videoUri, setVideoUri] = useState(null);
  const [isVideo, setIsVideo] = useState(false);
  const [videoFrames, setVideoFrames] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [detectedImageUri, setDetectedImageUri] = useState(null);
  const [segmentedImageUri, setSegmentedImageUri] = useState(null);
  const [roiImageUri, setRoiImageUri] = useState(null);
  const [probTopK, setProbTopK] = useState([]);
  const [detectionBox, setDetectionBox] = useState(null);
  const detectionSessionRef = useRef(null);
  const classificationSessionRef = useRef(null);

  const log = (...a) => console.log('[CatApp]', ...a);
  const warn = (...a) => console.warn('[CatApp]', ...a);
  const err = (...a) => console.error('[CatApp]', ...a);

  const topK = (probs, k = 3) =>
    probs.map((p, i) => ({ i, p }))
         .sort((a, b) => b.p - a.p)
         .slice(0, Math.min(k, probs.length));

  // ---- Ładowanie modeli ----
  const loadModels = useCallback(async () => {
    try {
      console.log('🚀 Starting model loading...');
      setReady(false);
      setStatus('📦 Ładowanie modeli…');

      // Sprawdź czy ONNX Runtime jest dostępny
      console.log('ONNX Runtime version:', ort.version);

      // 1. Załaduj best.onnx (detekcja kota)
      setStatus('📦 Ładowanie best.onnx (detekcja)…');
      const detectionModelPath = await prepareBestOnnxModel();
      log('Detection model path:', detectionModelPath);
     
      detectionSessionRef.current = await ort.InferenceSession.create(detectionModelPath, {
        executionProviders: Platform.OS === 'android' ? ['xnnpack', 'cpu'] : ['cpu'],
        graphOptimizationLevel: 'all',
      });
      log('Detection model loaded');

      // 2. Załaduj mobilenetv2_finetuned.onnx (klasyfikacja)
      setStatus('📦 Ładowanie mobilenetv2_finetuned.onnx (klasyfikacja)…');
      const classificationModelPath = await prepareOnnxWithExternalData();
      log('Classification model path:', classificationModelPath);
     
      classificationSessionRef.current = await ort.InferenceSession.create(classificationModelPath, {
        executionProviders: Platform.OS === 'android' ? ['xnnpack', 'cpu'] : ['cpu'],
        graphOptimizationLevel: 'all',
      });
      log('Classification model loaded');

      setStatus('✅ Gotowe - wybierz zdjęcie lub wideo kota');
      setReady(true);
      console.log('✅ All models loaded successfully');
    } catch (e) {
      err('Błąd ładowania modeli:', e);
      setStatus('❌ Błąd ładowania modeli');
      Alert.alert('Model Error', `Nie udało się załadować modeli: ${e?.message || e}`);
    }
  }, []);

  useEffect(() => {
    console.log('🔧 App mounted, loading models...');
    loadModels();
  }, [loadModels]);

  // ---- Wybór zdjęcia lub wideo ----
  const pickImage = useCallback(async () => {
    try {
      console.log('📸 Opening media picker...');
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Brak uprawnień', 'Potrzebny dostęp do galerii.');
        return;
      }

      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All, // Obrazy i wideo
        quality: 1,
        base64: false,
        allowsEditing: false,
      });
     
      if (res.canceled || !res.assets?.length) {
        console.log('Media selection cancelled');
        return;
      }

      const asset = res.assets[0];
      const uri = asset.uri;
      const isVideoFile = asset.type === 'video';
      
      log('Wybrano:', isVideoFile ? 'wideo' : 'obraz', uri);
     
      // Resetuj stan
      setOriginalImageUri(isVideoFile ? null : uri);
      setVideoUri(isVideoFile ? uri : null);
      setIsVideo(isVideoFile);
      setVideoFrames([]);
      setCurrentFrameIndex(0);
      setDetectedImageUri(null);
      setSegmentedImageUri(null);
      setRoiImageUri(null);
      setProbTopK([]);
      setDetectionBox(null);

      if (isVideoFile) {
        // Przetwarzaj wideo
        await processVideo(uri);
      } else {
        // Przetwarzaj obraz
        await detectAndClassify(uri);
      }
    } catch (e) {
      err('Błąd wyboru media:', e);
      setStatus('❌ Błąd media');
      Alert.alert('Media Error', `Błąd podczas wyboru: ${e?.message || e}`);
    }
  }, []);

  // ---- Klasyfikacja ROI ----
  const classifyROI = useCallback(async (roiImageUri: string) => {
    const classificationSession = classificationSessionRef.current;
    
    if (!classificationSession) {
      warn('Model klasyfikacji niegotowy');
      return;
    }
    
    try {
      log('Klasyfikacja ROI...');
      
      // Wczytaj ROI i przeskaluj do 224x224
      const roiResized = await ImageManipulator.manipulateAsync(
        roiImageUri,
        [{ resize: { width: 224, height: 224 } }],
        {
          compress: 0.95,
          format: ImageManipulator.SaveFormat.JPEG,
          base64: true,
        }
      );
      
      if (!roiResized.base64) {
        throw new Error('Brak base64 z ROI');
      }
      
      // Klasyfikacja
      const chw = chwFromBase64JPEG224(roiResized.base64, IMAGENET_MEAN, IMAGENET_STD, USE_BGR);
      const inputName = classificationSession.inputNames?.[0] ?? 'input';
      const classificationTensor = new ort.Tensor('float32', chw, [1, 3, 224, 224]);
      
      const outputMap = await classificationSession.run({ [inputName]: classificationTensor });
      const keys = Object.keys(outputMap);
      
      const probAliases = ['prob', 'probs', 'probabilities', 'softmax'];
      const logitAliases = ['logits', 'output'];
      const outName =
        probAliases.find(k => keys.includes(k)) ??
        logitAliases.find(k => keys.includes(k)) ??
        keys[0];
      
      const outT = outputMap[outName];
      if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);
      const data = outT.data;
      
      let probs;
      if (probAliases.includes(outName)) {
        probs = Array.from(data);
      } else {
        let max = -Infinity;
        for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
        const exps = new Float32Array(data.length);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = Math.exp(data[i] - max);
          exps[i] = v;
          sum += v;
        }
        probs = Array.from(exps, v => v / (sum || 1));
      }
      
      const top = topK(probs, 3).map(({ i, p }) => ({
        label: labels[i] ?? `cls_${i}`,
        p,
      }));
      
      setProbTopK(top);
      log('TOP-3:', top.map(t => `${t.label}: ${(t.p * 100).toFixed(1)}%`).join(', '));
    } catch (e) {
      err('Błąd klasyfikacji ROI:', e);
    }
  }, []);

  // ---- Aktualizacja wyświetlanych obrazów dla wybranej klatki wideo ----
  const updateFrameDisplay = useCallback(async (frameIndex: number) => {
    if (!videoFrames || videoFrames.length === 0 || frameIndex < 0 || frameIndex >= videoFrames.length) {
      return;
    }

    const frame = videoFrames[frameIndex];
    
    try {
      // Zapisz zsegmentowany obraz
      if (frame.segmented_image) {
        const segmentedDir = FileSystem.cacheDirectory + 'Segmentation/segmented/';
        await FileSystem.makeDirectoryAsync(segmentedDir, { intermediates: true }).catch(() => {});
        
        const timestamp = Date.now();
        const segmentedImagePath = segmentedDir + `video_segmented_${timestamp}_${frameIndex}.jpg`;
        
        const base64Data = frame.segmented_image.includes(',') 
          ? frame.segmented_image.split(',')[1] 
          : frame.segmented_image;
        
        await FileSystem.writeAsStringAsync(segmentedImagePath, base64Data, {
          encoding: FileSystem.EncodingType.Base64,
        });
        
        setSegmentedImageUri(segmentedImagePath);
      }
      
      // Zapisz ROI i wykonaj klasyfikację
      if (frame.roi_image) {
        const roiDir = FileSystem.cacheDirectory + 'Segmentation/roi/';
        await FileSystem.makeDirectoryAsync(roiDir, { intermediates: true }).catch(() => {});
        
        const timestamp = Date.now();
        const roiImagePath = roiDir + `video_roi_${timestamp}_${frameIndex}.jpg`;
        
        const roiBase64Data = frame.roi_image.includes(',') 
          ? frame.roi_image.split(',')[1] 
          : frame.roi_image;
        
        await FileSystem.writeAsStringAsync(roiImagePath, roiBase64Data, {
          encoding: FileSystem.EncodingType.Base64,
        });
        
        setRoiImageUri(roiImagePath);
        
        // Wykonaj klasyfikację ROI
        await classifyROI(roiImagePath);
      }
    } catch (e) {
      err('Błąd aktualizacji wyświetlania klatki:', e);
    }
  }, [videoFrames, classifyROI]);

  // ---- Przetwarzanie wideo ----
  const processVideo = useCallback(async (videoUri: string) => {
    try {
      setBusy(true);
      setStatus('🎬 Przetwarzanie wideo…');
      
      log('Wysyłanie wideo do backendu do przetworzenia...');
      
      // Wczytaj wideo jako base64
      log('Wczytywanie wideo...');
      const videoBase64 = await FileSystem.readAsStringAsync(videoUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      log(`Wideo wczytane, rozmiar: ${videoBase64.length} znaków base64`);
      
      // Wyślij do backendu jako JSON z base64
      const response = await Promise.race([
        fetch(`${SEGMENTATION_BACKEND_URL}/process_video`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            video: `data:video/mp4;base64,${videoBase64}`,
            frame_count: 10, // 10 klatek
          }),
        }),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout - przetwarzanie wideo trwa zbyt długo')), 300000) // 5 minut
        )
      ]).catch((e) => {
        log('Błąd podczas wysyłania wideo:', e);
        throw e;
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }
      
      const result = await response.json();
      
      if (!result.success || !result.frames || result.frames.length === 0) {
        throw new Error(result.error || 'Brak przetworzonych klatek w odpowiedzi');
      }
      
      log(`Otrzymano ${result.frames.length} przetworzonych klatek`);
      setVideoFrames(result.frames);
      setCurrentFrameIndex(0);
      
      // Wyświetl pierwszą klatkę
      if (result.frames.length > 0) {
        await updateFrameDisplay(0);
      }
      
      setStatus('✅ Wideo przetworzone');
      
    } catch (e) {
      err('Błąd przetwarzania wideo:', e);
      setStatus('❌ Błąd przetwarzania wideo');
      Alert.alert('Video Error', `Błąd podczas przetwarzania wideo: ${e?.message || e}`);
    } finally {
      setBusy(false);
    }
  }, [updateFrameDisplay]);

  // ---- Detekcja kota i klasyfikacja ----
  const detectAndClassify = useCallback(async (imageUri) => {
    const detectionSession = detectionSessionRef.current;
    const classificationSession = classificationSessionRef.current;

    if (!detectionSession || !classificationSession) {
      warn('Modele niegotowe');
      setStatus('⏳ Modele się ładują…');
      return;
    }

    setBusy(true);
    setStatus('🔍 Detekcja kota…');

    try {
      console.log('🐱 Starting cat detection...');

      // ========== KROK 1: DETEKCJA KOTA (best.onnx) ==========
      const imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Preprocessing dla detekcji
      const { tensor: yoloTensor, metadata } = yoloPreprocess(imageBase64, YOLO_INPUT_SIZE);
      console.log('Preprocessing completed');

      // Inference detekcji
      const detectionInputName = detectionSession.inputNames[0];
      const detectionInputTensor = new ort.Tensor('float32', yoloTensor, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
     
      console.log('Running detection inference...');
      const detectionOutputs = await detectionSession.run({ [detectionInputName]: detectionInputTensor });
      const detectionOutputNames = detectionSession.outputNames;
     
      log('Detection output names:', detectionOutputNames);
      log('Detection output shapes:', Object.values(detectionOutputs).map((o: any) => o.dims));

      // POSTPROCESSING W STYLU PYTHONA
      let box = null;
      let score = 0;

      const detectionOutput = detectionOutputs[detectionOutputNames[0]];
      const outputData = detectionOutput.data as Float32Array | number[];
      const outputDims = detectionOutput.dims;

      console.log('Starting postprocessing...');

      // Przekształć output do formatu (num_detections, 5)
      let detections: number[][] = [];
     
      if (outputDims.length === 3 && outputDims[0] === 1) {
        // Format: [1, 5, num_detections] - transponuj do [num_detections, 5]
        const numDetections = outputDims[2];
        for (let i = 0; i < numDetections; i++) {
          const detection = [
            Number(outputData[i]),                    // x_center
            Number(outputData[numDetections + i]),   // y_center  
            Number(outputData[2 * numDetections + i]), // width
            Number(outputData[3 * numDetections + i]), // height
            Number(outputData[4 * numDetections + i])  // confidence
          ];
          detections.push(detection);
        }
      } else if (outputDims.length === 2 && outputDims[0] === 5) {
        // Format: [5, num_detections] - transponuj do [num_detections, 5]
        const numDetections = outputDims[1];
        for (let i = 0; i < numDetections; i++) {
          const detection = [
            Number(outputData[i]),                    // x_center
            Number(outputData[numDetections + i]),   // y_center
            Number(outputData[2 * numDetections + i]), // width
            Number(outputData[3 * numDetections + i]), // height
            Number(outputData[4 * numDetections + i])  // confidence
          ];
          detections.push(detection);
        }
      } else {
        // Fallback - spróbuj zinterpretować jako [num_detections, 5+]
        const numDetections = outputDims[0];
        const detSize = outputDims[1];
        for (let i = 0; i < numDetections; i++) {
          const offset = i * detSize;
          const x_center = Number(outputData[offset]);
          const y_center = Number(outputData[offset + 1]);
          const width = Number(outputData[offset + 2]);
          const height = Number(outputData[offset + 3]);
          const confidence = detSize > 4 ? Number(outputData[offset + 4]) : 1.0;
         
          detections.push([x_center, y_center, width, height, confidence]);
        }
      }

      log(`Przetworzono ${detections.length} detekcji`);

      // Filtruj detekcje według confidence threshold
      const validDetections = detections.filter(det => det[4] >= CONF_THRESHOLD);
     
      if (validDetections.length === 0) {
        Alert.alert('Brak kota', 'Nie znaleziono kota na zdjęciu. Spróbuj innego obrazu.');
        setStatus('❌ Nie znaleziono kota');
        setBusy(false);
        return;
      }

      // Znajdź detekcję z najwyższym confidence
      let bestDetection = validDetections[0];
      for (let i = 1; i < validDetections.length; i++) {
        if (validDetections[i][4] > bestDetection[4]) {
          bestDetection = validDetections[i];
        }
      }

      const [x_center, y_center, width, height, confidence] = bestDetection;
      score = confidence;

      console.log('Best detection found, converting coordinates...');

      // KONWERSJA WSPÓŁRZĘDNYCH DO ORYGINALNEGO ROZMIARU
      const scale = Math.min(YOLO_INPUT_SIZE / metadata.originalWidth, YOLO_INPUT_SIZE / metadata.originalHeight);
      const newWidth = Math.round(metadata.originalWidth * scale);
      const newHeight = Math.round(metadata.originalHeight * scale);

      // Konwertuj z formatu YOLO [x_center, y_center, width, height] do [x1, y1, x2, y2] w przestrzeni 640x640
      let x1_640 = x_center - width / 2;
      let y1_640 = y_center - height / 2;
      let x2_640 = x_center + width / 2;
      let y2_640 = y_center + height / 2;

      // Usuń padding
      x1_640 = x1_640 - metadata.padLeft;
      y1_640 = y1_640 - metadata.padTop;
      x2_640 = x2_640 - metadata.padLeft;
      y2_640 = y2_640 - metadata.padTop;

      // Ogranicz do zakresu obrazu bez paddingu
      x1_640 = Math.max(0, Math.min(newWidth, x1_640));
      y1_640 = Math.max(0, Math.min(newHeight, y1_640));
      x2_640 = Math.max(0, Math.min(newWidth, x2_640));
      y2_640 = Math.max(0, Math.min(newHeight, y2_640));

      // Przeskaluj do oryginalnego rozmiaru
      const x1_orig = x1_640 / scale;
      const y1_orig = y1_640 / scale;
      const x2_orig = x2_640 / scale;
      const y2_orig = y2_640 / scale;

      // Utwórz finalny box i ogranicz do rozmiaru obrazu
      box = [
        Math.max(0, Math.min(metadata.originalWidth, x1_orig)),
        Math.max(0, Math.min(metadata.originalHeight, y1_orig)),
        Math.max(0, Math.min(metadata.originalWidth, x2_orig)),
        Math.max(0, Math.min(metadata.originalHeight, y2_orig)),
      ];

      console.log('Final box coordinates:', box);

      setDetectionBox(box);

      // ========== KROK 2: NARYSUJ RAMKĘ NA OBRAZIE ==========
      setStatus('📐 Rysowanie ramki…');
      const boxedImageUri = await drawBoundingBox(imageUri, box, '#00ff00', 3);
      setDetectedImageUri(boxedImageUri);

      // ========== KROK 3: SEGMENTACJA (wysyłanie do backendu) ==========
      setStatus('🎨 Segmentacja kota…');
      
      let roiImagePathLocal = null; // Zmienna lokalna do przechowania ścieżki ROI (dostępna dla klasyfikacji)
      
      try {
        // Najpierw sprawdź czy backend jest dostępny
        log('Sprawdzanie połączenia z backendem:', SEGMENTATION_BACKEND_URL);
        log('Platform:', Platform.OS);
        
        const healthCheck = await Promise.race([
          fetch(`${SEGMENTATION_BACKEND_URL}/health`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
            // Timeout po 10 sekundach
          }),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Timeout - backend nie odpowiada')), 10000)
          )
        ]).catch((e) => {
          log('Błąd połączenia z backendem:', e);
          log('URL próbowany:', SEGMENTATION_BACKEND_URL);
          throw new Error(`Nie można połączyć się z backendem: ${SEGMENTATION_BACKEND_URL}. Błąd: ${e.message}`);
        });
        
        if (!healthCheck.ok) {
          throw new Error(`Backend zwrócił błąd: ${healthCheck.status}`);
        }
        
        const healthData = await healthCheck.json();
        log('Backend dostępny, modele załadowane:', healthData.models_loaded);
        log('Szczegóły modeli:', {
          yolo: healthData.yolo_loaded,
          sam: healthData.sam_loaded
        });
        
        if (!healthData.models_loaded) {
          const errorMsg = 
            `Modele w backendzie nie są załadowane.\n\n` +
            `YOLO: ${healthData.yolo_loaded ? '✅' : '❌'}\n` +
            `SAM: ${healthData.sam_loaded ? '✅' : '❌'}\n\n` +
            `Sprawdź konsolę backendu - powinny być widoczne błędy ładowania modeli.\n\n` +
            `Upewnij się, że:\n` +
            `1. Plik last.pt znajduje się w katalogu Segmentation/\n` +
            `2. Plik sam_vit_h_4b8939.pth znajduje się w katalogu Segmentation/\n` +
            `3. Backend został uruchomiony: python Segmentation/backend_server.py`;
          throw new Error(errorMsg);
        }
        
        // Wczytaj obraz jako base64
        const imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        
        // Wyślij obraz do backendu
        log('Wysyłanie obrazu do backendu segmentacji...');
        log('URL:', `${SEGMENTATION_BACKEND_URL}/segment`);
        log('Rozmiar obrazu (base64):', imageBase64.length, 'znaków');
        
        const response = await Promise.race([
          fetch(`${SEGMENTATION_BACKEND_URL}/segment`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image: `data:image/jpeg;base64,${imageBase64}`,
            }),
          }),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Timeout - segmentacja trwa zbyt długo')), 60000)
          )
        ]).catch((e) => {
          log('Błąd podczas wysyłania obrazu:', e);
          throw e;
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
          throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success || !result.segmented_image) {
          throw new Error(result.error || 'Brak zsegmentowanego obrazu w odpowiedzi');
        }
        
        // Zapisz zsegmentowany obraz do pliku
        const segmentedBase64 = result.segmented_image;
        const segmentedDir = FileSystem.cacheDirectory + 'Segmentation/segmented/';
        await FileSystem.makeDirectoryAsync(segmentedDir, { intermediates: true }).catch(() => {});
        
        const timestamp = Date.now();
        const segmentedImagePath = segmentedDir + `segmented_${timestamp}.jpg`;
        
        // Konwertuj base64 na plik
        const base64Data = segmentedBase64.includes(',') 
          ? segmentedBase64.split(',')[1] 
          : segmentedBase64;
        
        await FileSystem.writeAsStringAsync(segmentedImagePath, base64Data, {
          encoding: FileSystem.EncodingType.Base64,
        });
        
        log('Zsegmentowany obraz zapisany:', segmentedImagePath);
        setSegmentedImageUri(segmentedImagePath);
        
        // ========== KROK 4: WYCINANIE ROI Z ZSEGMENTOWANEGO OBRAZU ==========
        setStatus('✂️ Wycinanie ROI…');
        
        try {
          log('Wysyłanie zsegmentowanego obrazu do ROI...');
          const roiResponse = await Promise.race([
            fetch(`${SEGMENTATION_BACKEND_URL}/roi`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                image: `data:image/jpeg;base64,${base64Data}`,
              }),
            }),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('Timeout - ROI trwa zbyt długo')), 30000)
            )
          ]).catch((e) => {
            log('Błąd podczas wysyłania do ROI:', e);
            throw e;
          });
          
          if (!roiResponse.ok) {
            const errorData = await roiResponse.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(errorData.error || `HTTP ${roiResponse.status}`);
          }
          
          const roiResult = await roiResponse.json();
          
          if (!roiResult.success || !roiResult.roi_image) {
            throw new Error(roiResult.error || 'Brak ROI w odpowiedzi');
          }
          
          // Zapisz ROI obraz do pliku
          const roiBase64 = roiResult.roi_image;
          const roiDir = FileSystem.cacheDirectory + 'Segmentation/roi/';
          await FileSystem.makeDirectoryAsync(roiDir, { intermediates: true }).catch(() => {});
          
          const roiTimestamp = Date.now();
          const roiImagePath = roiDir + `roi_${roiTimestamp}.jpg`;
          
          // Konwertuj base64 na plik
          const roiBase64Data = roiBase64.includes(',') 
            ? roiBase64.split(',')[1] 
            : roiBase64;
          
          await FileSystem.writeAsStringAsync(roiImagePath, roiBase64Data, {
            encoding: FileSystem.EncodingType.Base64,
          });
          
          log('ROI obraz zapisany:', roiImagePath);
          setRoiImageUri(roiImagePath);
          roiImagePathLocal = roiImagePath; // Zapisz lokalnie dla klasyfikacji
          
        } catch (e) {
          err('Błąd podczas wycinania ROI:', e);
          warn('ROI nie powiodło się, kontynuowanie bez ROI...');
          // Kontynuuj bez ROI
        }
        
      } catch (e) {
        err('Błąd podczas segmentacji:', e);
        const errorMessage = e?.message || 'Nieznany błąd';
        
        // Sprawdź czy backend jest dostępny
        if (errorMessage.includes('Network request failed') || 
            errorMessage.includes('fetch') || 
            errorMessage.includes('Nie można połączyć się')) {
          Alert.alert(
            'Błąd połączenia',
            `Nie można połączyć się z backendem segmentacji.\n\n` +
            `URL: ${SEGMENTATION_BACKEND_URL}\n\n` +
            `Upewnij się, że:\n` +
            `1. Backend jest uruchomiony: python Segmentation/backend_server.py\n` +
            `2. Używasz właściwego URL dla swojej platformy\n` +
            `3. Telefon i komputer są w tej samej sieci WiFi\n\n` +
            `Dla Android emulator: http://10.0.2.2:5000\n` +
            `Dla iOS simulator: http://localhost:5000\n` +
            `Dla urządzenia: http://192.168.1.12:5000`,
            [{ text: 'OK' }]
          );
        } else {
          Alert.alert('Błąd segmentacji', `Nie udało się zsegmentować obrazu: ${errorMessage}`);
        }
        warn('Segmentacja nie powiodła się, kontynuowanie bez zsegmentowanego obrazu...');
      }

      // ========== KROK 5: KLASYFIKACJA (MobileNet) - używa ROI jeśli dostępny ==========
      setStatus('🏷️ Klasyfikacja rasy…');
      
      let imageForClassification;
      
      // Użyj ROI jeśli dostępny, w przeciwnym razie użyj wyciętego kota z bounding box
      if (roiImagePathLocal) {
        log('Używanie ROI do klasyfikacji');
        
        // Konwertuj ROI do formatu 224x224 dla klasyfikacji
        const roiResized = await ImageManipulator.manipulateAsync(
          roiImagePathLocal,
          [{ resize: { width: 224, height: 224 } }],
          {
            compress: 0.95,
            format: ImageManipulator.SaveFormat.JPEG,
            base64: true,
          }
        );
        
        if (!roiResized.base64) {
          throw new Error('Brak base64 z ROI');
        }
        
        imageForClassification = roiResized.base64;
      } else {
        log('Używanie wyciętego kota z bounding box do klasyfikacji');
        // Fallback - użyj wyciętego kota z bounding box
        const [x1, y1, x2, y2] = box.map(v => Math.round(v));
        const cropWidth = Math.max(1, x2 - x1);
        const cropHeight = Math.max(1, y2 - y1);

        const cropped = await ImageManipulator.manipulateAsync(
          imageUri,
          [
            {
              crop: {
                originX: Math.max(0, x1),
                originY: Math.max(0, y1),
                width: Math.min(cropWidth, metadata.originalWidth - x1),
                height: Math.min(cropHeight, metadata.originalHeight - y1),
              },
            },
            { resize: { width: 224, height: 224 } },
          ],
          {
            compress: 0.95,
            format: ImageManipulator.SaveFormat.JPEG,
            base64: true,
          }
        );

        if (!cropped.base64) {
          throw new Error('Brak base64 po wycięciu');
        }
        
        imageForClassification = cropped.base64;
      }
      
      const chw = chwFromBase64JPEG224(imageForClassification, IMAGENET_MEAN, IMAGENET_STD, USE_BGR);

      const inputName = classificationSession.inputNames?.[0] ?? 'input';
      const classificationTensor = new ort.Tensor('float32', chw, [1, 3, 224, 224]);

      const outputMap = await classificationSession.run({ [inputName]: classificationTensor });
      const keys = Object.keys(outputMap);

      const probAliases = ['prob', 'probs', 'probabilities', 'softmax'];
      const logitAliases = ['logits', 'output'];
      const outName =
        probAliases.find(k => keys.includes(k)) ??
        logitAliases.find(k => keys.includes(k)) ??
        keys[0];

      const outT = outputMap[outName];
      if (!outT?.data) throw new Error(`Puste wyjście modelu "${outName}"`);
      const data = outT.data;

      let probs;
      if (probAliases.includes(outName)) {
        probs = Array.from(data);
      } else {
        let max = -Infinity;
        for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
        const exps = new Float32Array(data.length);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = Math.exp(data[i] - max);
          exps[i] = v;
          sum += v;
        }
        probs = Array.from(exps, v => v / (sum || 1));
      }

      const top = topK(probs, 3).map(({ i, p }) => ({
        label: labels[i] ?? `cls_${i}`,
        p,
      }));

      setProbTopK(top);
      setStatus('✅ Gotowe');
      console.log('Classification completed successfully');
      log('TOP-3:', top.map(t => `${t.label}: ${(t.p * 100).toFixed(1)}%`).join(', '));
    } catch (e) {
      err('Błąd przetwarzania:', e);
      setStatus('❌ Błąd przetwarzania');
      Alert.alert('Processing Error', `Błąd podczas przetwarzania: ${e?.message || e}`);
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar backgroundColor={BG} barStyle="light-content" />
      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={styles.scrollContent}
      >
        <Text style={styles.title}>🐱 Cat Classifier</Text>
        <Text style={[styles.status, { color: ready ? '#6ee17a' : FG_MUTED }]}>
          ☑ {status}
        </Text>

        <View style={styles.buttonRow}>
          <Pressable
            onPress={pickImage}
            disabled={!ready || busy}
            style={[
              styles.primaryButton,
              {
                backgroundColor: ready && !busy ? ACCENT : '#3a3a3a',
                opacity: ready && !busy ? 1 : 0.7,
              }
            ]}
          >
            <Text style={[styles.buttonText, { color: FG }]}>
              Wybierz zdjęcie/wideo
            </Text>
          </Pressable>

          <Pressable
            onPress={loadModels}
            disabled={busy}
            style={[
              styles.secondaryButton,
              { backgroundColor: '#2c2c2c' }
            ]}
          >
            <Text style={[styles.buttonText, { color: FG, fontSize: 16 }]}>
              🔁 Przeładuj
            </Text>
          </Pressable>
        </View>

        {busy && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={ACCENT} />
            <Text style={{ color: FG_MUTED, marginTop: 8 }}>{status}</Text>
          </View>
        )}

        {/* Wyniki wideo */}
        {isVideo && videoUri && videoFrames.length > 0 && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>
              Wyniki wideo ({videoFrames.length} klatek):
            </Text>

            <View style={{ gap: 16 }}>
              {/* Wybór klatki */}
              <View style={styles.imageContainer}>
                <Text style={styles.imageLabel}>
                  Wybrana klatka: {currentFrameIndex + 1} / {videoFrames.length}
                </Text>
                <View style={{ flexDirection: 'row', gap: 8, marginTop: 8 }}>
                  <Pressable
                    onPress={async () => {
                      const newIndex = Math.max(0, currentFrameIndex - 1);
                      setCurrentFrameIndex(newIndex);
                      await updateFrameDisplay(newIndex);
                    }}
                    disabled={currentFrameIndex === 0}
                    style={[
                      styles.secondaryButton,
                      { 
                        backgroundColor: currentFrameIndex === 0 ? '#3a3a3a' : ACCENT,
                        opacity: currentFrameIndex === 0 ? 0.5 : 1,
                      }
                    ]}
                  >
                    <Text style={[styles.buttonText, { color: FG, fontSize: 14 }]}>
                      ← Poprzednia
                    </Text>
                  </Pressable>
                  <Pressable
                    onPress={async () => {
                      const newIndex = Math.min(videoFrames.length - 1, currentFrameIndex + 1);
                      setCurrentFrameIndex(newIndex);
                      await updateFrameDisplay(newIndex);
                    }}
                    disabled={currentFrameIndex === videoFrames.length - 1}
                    style={[
                      styles.secondaryButton,
                      { 
                        backgroundColor: currentFrameIndex === videoFrames.length - 1 ? '#3a3a3a' : ACCENT,
                        opacity: currentFrameIndex === videoFrames.length - 1 ? 0.5 : 1,
                      }
                    ]}
                  >
                    <Text style={[styles.buttonText, { color: FG, fontSize: 14 }]}>
                      Następna →
                    </Text>
                  </Pressable>
                </View>
                {videoFrames[currentFrameIndex] && (
                  <Text style={{ color: FG_MUTED, fontSize: 12, marginTop: 4 }}>
                    Czas: {videoFrames[currentFrameIndex].time_seconds?.toFixed(2)}s
                  </Text>
                )}
              </View>

              {/* Zsegmentowany obraz */}
              {segmentedImageUri && (
                <View style={styles.imageContainer}>
                  <Text style={[styles.imageLabel, { color: '#6ee17a' }]}>
                    3. Segmentacja (klatka {currentFrameIndex + 1})
                  </Text>
                  <Image
                    source={{ uri: segmentedImageUri }}
                    style={[styles.image, { borderColor: '#6ee17a' }]}
                    resizeMode="contain"
                  />
                </View>
              )}

              {/* ROI */}
              {roiImageUri && (
                <View style={styles.imageContainer}>
                  <Text style={[styles.imageLabel, { color: '#ff6b6b' }]}>
                    4. ROI - Wycięty kot (klatka {currentFrameIndex + 1})
                  </Text>
                  <Image
                    source={{ uri: roiImageUri }}
                    style={[styles.image, { borderColor: '#ff6b6b' }]}
                    resizeMode="contain"
                  />
                </View>
              )}

              {/* Wyniki klasyfikacji dla wideo */}
              {probTopK.length > 0 && (
                <View style={styles.classificationContainer}>
                  <Text style={styles.classificationTitle}>
                    5. Wynik klasyfikacji (MobileNet) - klatka {currentFrameIndex + 1}
                  </Text>
                  <FlatList
                    data={probTopK}
                    scrollEnabled={false}
                    keyExtractor={(item, idx) => `${item.label}_${idx}`}
                    renderItem={({ item }) => (
                      <View style={styles.classificationItem}>
                        <Text style={styles.classificationText}>{item.label}</Text>
                        <Text style={[styles.classificationText, { fontWeight: '700' }]}>
                          {(item.p * 100).toFixed(1)}%
                        </Text>
                      </View>
                    )}
                  />
                </View>
              )}
            </View>
          </View>
        )}

        {originalImageUri && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>
              Wyniki:
            </Text>

            <View style={{ gap: 16 }}>
              {/* Oryginalny obraz */}
              <View style={styles.imageContainer}>
                <Text style={styles.imageLabel}>
                  1. Oryginalny obraz
                </Text>
                <Image
                  source={{ uri: originalImageUri }}
                  style={styles.image}
                  resizeMode="contain"
                />
              </View>

              {/* Obraz z ramką detekcji */}
              {detectedImageUri && (
                <View style={styles.imageContainer}>
                  <Text style={[styles.imageLabel, { color: ACCENT }]}>
                    2. Detekcja kota (best.onnx)
                  </Text>
                  <Image
                    source={{ uri: detectedImageUri }}
                    style={[styles.image, { borderColor: ACCENT }]}
                    resizeMode="contain"
                  />
                  {detectionBox && (
                    <Text style={{ color: FG_MUTED, fontSize: 12, marginTop: 4 }}>
                      Box: [{detectionBox.map(v => Math.round(v)).join(', ')}]
                    </Text>
                  )}
                </View>
              )}

              {/* Zsegmentowany obraz */}
              {segmentedImageUri && (
                <View style={styles.imageContainer}>
                  <Text style={[styles.imageLabel, { color: '#6ee17a' }]}>
                    3. Segmentacja (last.pt + SAM)
                  </Text>
                  <Image
                    source={{ uri: segmentedImageUri }}
                    style={[styles.image, { borderColor: '#6ee17a' }]}
                    resizeMode="contain"
                  />
                </View>
              )}

              {/* ROI - Wycięty kot */}
              {roiImageUri && (
                <View style={styles.imageContainer}>
                  <Text style={[styles.imageLabel, { color: '#ff6b6b' }]}>
                    4. ROI - Wycięty kot
                  </Text>
                  <Image
                    source={{ uri: roiImageUri }}
                    style={[styles.image, { borderColor: '#ff6b6b' }]}
                    resizeMode="contain"
                  />
                </View>
              )}

              {/* Wyniki klasyfikacji */}
              {probTopK.length > 0 && (
                <View style={styles.classificationContainer}>
                  <Text style={styles.classificationTitle}>
                    5. Wynik klasyfikacji (MobileNet)
                  </Text>
                  <FlatList
                    data={probTopK}
                    scrollEnabled={false}
                    keyExtractor={(item, idx) => `${item.label}_${idx}`}
                    renderItem={({ item }) => (
                      <View style={styles.classificationItem}>
                        <Text style={styles.classificationText}>{item.label}</Text>
                        <Text style={[styles.classificationText, { fontWeight: '700' }]}>
                          {(item.p * 100).toFixed(1)}%
                        </Text>
                      </View>
                    )}
                  />
                </View>
              )}
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}