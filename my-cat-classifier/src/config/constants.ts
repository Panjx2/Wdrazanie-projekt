export const COLORS = {
  BG: '#0b0b0c',
  FG: '#ffffff',
  FG_MUTED: '#cfcfcf',
  ACCENT: '#1f6feb',
  BORDER: '#222',
} as const;

export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

export const USE_BGR = false;
export const USE_PNG_LOSSLESS = false; // ustaw na true, by zapisać do PNG (dokładniejszy tensor, większy plik)

export const CAMERA_CAPTURE_INTERVAL_MS = 10;
export const CAMERA_QUALITY = 1;
export const CAMERA_STATUS_INTERVAL_MS = 10; // jak często odświeżać status w trybie cichym
