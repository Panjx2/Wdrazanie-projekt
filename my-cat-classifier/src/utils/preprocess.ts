import base64 from 'base64-js';
import jpeg from 'jpeg-js';

/**
 * Konwertuje base64 JPEG (224x224 RGB) do Float32Array (CHW),
 * z normalizacją mean/std jak dla ImageNet.
 *
 * @param jpegBase64 base64 string (bez prefiksu data:)
 * @param mean np. [0.485, 0.456, 0.406]
 * @param std np. [0.229, 0.224, 0.225]
 * @returns Float32Array z wartościami float32 (1,3,224,224)
 */
export function chwFromBase64JPEG224(
  jpegBase64: string,
  mean: number[] = [0.485, 0.456, 0.406],
  std: number[] = [0.229, 0.224, 0.225]
): Float32Array {
  // Dekoduj base64 → bajty JPEG
  const bytes = base64.toByteArray(jpegBase64);

  // Rozkoduj JPEG → RGBA
  const decoded = jpeg.decode(bytes, { useTArray: true }); // { width, height, data: Uint8Array }
  const { width, height, data } = decoded;

  if (width !== 224 || height !== 224) {
    throw new Error(`Oczekiwano 224x224, a otrzymano ${width}x${height}`);
  }

  const size = width * height;
  const out = new Float32Array(3 * size);

  // RGBA → CHW (RGB)
  for (let i = 0; i < size; i++) {
    const r = data[4 * i] / 255;
    const g = data[4 * i + 1] / 255;
    const b = data[4 * i + 2] / 255;

    out[0 * size + i] = (r - mean[0]) / std[0];
    out[1 * size + i] = (g - mean[1]) / std[1];
    out[2 * size + i] = (b - mean[2]) / std[2];
  }

  return out;
}
