export const COLORS = {
  BG: '#0b0b0c',
  FG: '#ffffff',
  FG_MUTED: '#cfcfcf',
  ACCENT: '#1f6feb',
  BORDER: '#222',
} as const;

export const USE_BGR = false;
export const USE_PNG_LOSSLESS = false; // ustaw na true, by zapisać do PNG (dokładniejszy tensor, większy plik)

export const CAMERA_CAPTURE_INTERVAL_MS = 10;
export const CAMERA_QUALITY = 1;
export const CAMERA_STATUS_INTERVAL_MS = 10; // jak często odświeżać status w trybie cichym

// YOLO11 input + NMS params
export const YOLO_INPUT_SIZE = 640;
export const YOLO_CONF_THRESHOLD = 0.25;
export const YOLO_IOU_THRESHOLD = 0.45;
export const YOLO_MAX_DETECTIONS = 50;
// Leave empty/null to allow all classes present in assets/labels.json
export const YOLO_ALLOWED_CLASS_IDS: number[] | null = null;
export const YOLO_DEFAULT_LABEL = 'Cat';
