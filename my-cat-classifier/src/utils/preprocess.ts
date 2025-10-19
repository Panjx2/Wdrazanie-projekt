import base64 from 'base64-js';
import jpeg from 'jpeg-js';

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
