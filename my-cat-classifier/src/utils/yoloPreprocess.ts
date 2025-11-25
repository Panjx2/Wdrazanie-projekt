import base64 from 'base64-js';
import jpeg from 'jpeg-js';

/**
 * Letterbox resize - zachowuje proporcje, dodaje padding
 */
export function letterbox(
  width: number,
  height: number,
  targetSize: number = 640
): { newWidth: number; newHeight: number; padLeft: number; padTop: number; scale: number } {
  const scale = Math.min(targetSize / width, targetSize / height);
  const newWidth = Math.round(width * scale);
  const newHeight = Math.round(height * scale);
  const padLeft = Math.round((targetSize - newWidth) / 2);
  const padTop = Math.round((targetSize - newHeight) / 2);
  return { newWidth, newHeight, padLeft, padTop, scale };
}

/**
 * Konwertuje base64 JPEG do Float32Array (CHW) dla YOLO
 * Wykonuje letterbox resize do 640x640 i normalizację [0,1]
 */
export function yoloPreprocess(
  jpegBase64: string,
  targetSize: number = 640
): {
  tensor: Float32Array;
  metadata: {
    originalWidth: number;
    originalHeight: number;
    padLeft: number;
    padTop: number;
    scale: number;
  };
} {
  const bytes = base64.toByteArray(jpegBase64);
  const decoded = jpeg.decode(bytes, { useTArray: true });
  const { width, height, data } = decoded;

  // Letterbox resize
  const { newWidth, newHeight, padLeft, padTop, scale } = letterbox(width, height, targetSize);

  const size = targetSize * targetSize;
  const out = new Float32Array(3 * size);

  // Wypełnij zerami (padding)
  out.fill(0);

  // Resize i normalizacja [0, 1]
  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const srcX = Math.floor(x / scale);
      const srcY = Math.floor(y / scale);
      
      const clampedX = Math.min(srcX, width - 1);
      const clampedY = Math.min(srcY, height - 1);
      
      const srcIdx = (clampedY * width + clampedX) * 4;
      
      const dstX = x + padLeft;
      const dstY = y + padTop;
      const dstIdx = dstY * targetSize + dstX;

      // Normalizacja [0, 1] - YOLO używa RGB
      out[0 * size + dstIdx] = data[srcIdx] / 255.0;     // R
      out[1 * size + dstIdx] = data[srcIdx + 1] / 255.0; // G
      out[2 * size + dstIdx] = data[srcIdx + 2] / 255.0; // B
    }
  }

  return {
    tensor: out,
    metadata: {
      originalWidth: width,
      originalHeight: height,
      padLeft,
      padTop,
      scale,
    },
  };
}

