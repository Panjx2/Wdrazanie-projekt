import base64 from 'base64-js';
import jpeg from 'jpeg-js';

export type LetterboxMeta = {
  ratio: number;
  pad: { x: number; y: number };
  origSize: { width: number; height: number };
};

/**
 * Konwertuje base64 JPEG (224x224 RGB) do Float32Array (CHW),
 * z normalizacją mean/std jak dla ImageNet.
 *
 * @param {string} jpegBase64 base64 string (bez prefiksu data:)
 * @param {number[]} mean np. [0.485, 0.456, 0.406]
 * @param {number[]} std np. [0.229, 0.224, 0.225]
 * @param {boolean} useBGR czy zamienić kanały R↔B (dla modeli trenowanych w BGR)
 * @returns {Float32Array} Float32Array (1,3,224,224)
 */
export function chwFromBase64JPEG224(
  jpegBase64,
  mean = [0.485, 0.456, 0.406],
  std = [0.229, 0.224, 0.225],
  useBGR = false
) {
  const bytes = base64.toByteArray(jpegBase64);
  const decoded = jpeg.decode(bytes, { useTArray: true });
  const { width, height, data } = decoded;

  if (width !== 224 || height !== 224)
    throw new Error(`Oczekiwano 224x224, otrzymano ${width}x${height}`);

  const size = width * height;
  const out = new Float32Array(3 * size);

  for (let i = 0; i < size; i++) {
    const R = data[4 * i] / 255;
    const G = data[4 * i + 1] / 255;
    const B = data[4 * i + 2] / 255;

    const r = useBGR ? B : R;
    const g = G;
    const b = useBGR ? R : B;

    out[0 * size + i] = (r - mean[0]) / std[0];
    out[1 * size + i] = (g - mean[1]) / std[1];
    out[2 * size + i] = (b - mean[2]) / std[2];
  }

  return out;
}

/**
 * Letterbox + normalizacja /255 -> Float32Array CHW (1,3,inputSize,inputSize)
 * dla modeli YOLO. Zwraca również metadane potrzebne do przeskalowania
 * detekcji z powrotem do rozmiaru wejściowego (zdjęcia z kamery/galerii).
 */
export function yoloLetterboxChwFromBase64(
  jpegBase64: string,
  inputSize: number,
  useBGR = false
): { chw: Float32Array; meta: LetterboxMeta } {
  const bytes = base64.toByteArray(jpegBase64);
  const decoded = jpeg.decode(bytes, { useTArray: true });
  const { width, height, data } = decoded;

  if (!width || !height) {
    throw new Error('Nie udało się sparsować JPEG');
  }

  const ratio = Math.min(inputSize / width, inputSize / height);
  const newW = Math.round(width * ratio);
  const newH = Math.round(height * ratio);
  const padX = Math.floor((inputSize - newW) / 2);
  const padY = Math.floor((inputSize - newH) / 2);

  const out = new Float32Array(3 * inputSize * inputSize);
  const padValue = 114 / 255; // zgodnie z Ultralytics

  for (let y = 0; y < inputSize; y += 1) {
    const inY = (y - padY) / ratio;
    const yValid = inY >= 0 && inY < height;
    const ySrc = Math.min(height - 1, Math.max(0, Math.floor(inY)));

    for (let x = 0; x < inputSize; x += 1) {
      const i = y * inputSize + x;
      const inX = (x - padX) / ratio;
      const xValid = inX >= 0 && inX < width;
      const xSrc = Math.min(width - 1, Math.max(0, Math.floor(inX)));

      if (xValid && yValid) {
        const idx = (ySrc * width + xSrc) * 4;
        const R = data[idx] / 255;
        const G = data[idx + 1] / 255;
        const B = data[idx + 2] / 255;

        const r = useBGR ? B : R;
        const g = G;
        const b = useBGR ? R : B;

        out[0 * inputSize * inputSize + i] = r;
        out[1 * inputSize * inputSize + i] = g;
        out[2 * inputSize * inputSize + i] = b;
      } else {
        out[0 * inputSize * inputSize + i] = padValue;
        out[1 * inputSize * inputSize + i] = padValue;
        out[2 * inputSize * inputSize + i] = padValue;
      }
    }
  }

  return {
    chw: out,
    meta: {
      ratio,
      pad: { x: padX, y: padY },
      origSize: { width, height },
    },
  };
}
