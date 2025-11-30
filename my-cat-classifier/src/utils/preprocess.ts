import base64 from 'base64-js';
import jpeg from 'jpeg-js';

export function chwFromBase64JPEG(
  jpegBase64: string,
  width: number,
  height: number,
  mean = [0.485, 0.456, 0.406],
  std = [0.229, 0.224, 0.225],
  useBGR = false
) {
  const bytes = base64.toByteArray(jpegBase64);
  const decoded = jpeg.decode(bytes, { useTArray: true });
  const { width: decodedW, height: decodedH, data } = decoded;

  if (decodedW !== width || decodedH !== height)
    throw new Error(`Oczekiwano ${width}x${height}, otrzymano ${decodedW}x${decodedH}`);

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

export function chwFromBase64JPEG224(
  jpegBase64: string,
  mean = [0.485, 0.456, 0.406],
  std = [0.229, 0.224, 0.225],
  useBGR = false
) {
  return chwFromBase64JPEG(jpegBase64, 224, 224, mean, std, useBGR);
}
