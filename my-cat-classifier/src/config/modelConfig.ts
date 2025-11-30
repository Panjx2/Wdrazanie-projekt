import { IMAGENET_MEAN, IMAGENET_STD, USE_BGR } from './constants';

type ModelKind = 'classification' | 'yolo';

type BaseModelConfig = {
  kind: ModelKind;
  basename: string;
  inputWidth: number;
  inputHeight: number;
  mean: number[];
  std: number[];
  useBgr: boolean;
  labels: string[];
};

type ClassificationModelConfig = BaseModelConfig & {
  kind: 'classification';
  onnxAsset: number;
  onnxDataAsset?: number;
};

type YoloModelConfig = BaseModelConfig & {
  kind: 'yolo';
  onnxAsset?: number;
  onnxDataAsset?: number;
  confThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  outputLayout: 'rows' | 'channels_first';
};

type ModelConfig = ClassificationModelConfig | YoloModelConfig;

// Change to 'yolo' after adding your YOLO ONNX asset below.
export const MODEL_KIND: ModelKind = 'classification';

const classificationModel: ClassificationModelConfig = {
  kind: 'classification',
  basename: 'mobilenetv3_finetuned',
  inputWidth: 224,
  inputHeight: 224,
  mean: IMAGENET_MEAN,
  std: IMAGENET_STD,
  useBgr: USE_BGR,
  labels: require('../../assets/labels.json'),
  onnxAsset: require('../../assets/models/mobilenetv3_finetuned.onnx'),
  onnxDataAsset: require('../../assets/models/mobilenetv3_finetuned.onnx.data'),
};

const yoloModel: YoloModelConfig = {
  kind: 'yolo',
  basename: 'yolo11s',
  inputWidth: 640,
  inputHeight: 640,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  useBgr: false,
  labels: require('../../assets/labels.json'),
  onnxAsset: require('../../assets/models/yolo11s.onnx'),
  onnxDataAsset: undefined,
  confThreshold: 0.25,
  iouThreshold: 0.45,
  maxDetections: 50,
  outputLayout: 'rows',
};

export const MODEL: ModelConfig =
  MODEL_KIND === 'yolo' ? yoloModel : classificationModel;
