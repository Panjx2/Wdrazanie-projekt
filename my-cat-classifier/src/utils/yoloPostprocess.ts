export type YoloDetection = {
  box: { x1: number; y1: number; x2: number; y2: number };
  score: number;
  classId: number;
  label: string;
};

type DecodeOptions = {
  numClasses: number;
  inputWidth: number;
  inputHeight: number;
  confThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  labels: string[];
  layout?: 'rows' | 'channels_first';
};

function iou(a: YoloDetection['box'], b: YoloDetection['box']) {
  const interX1 = Math.max(a.x1, b.x1);
  const interY1 = Math.max(a.y1, b.y1);
  const interX2 = Math.min(a.x2, b.x2);
  const interY2 = Math.min(a.y2, b.y2);
  const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const denom = areaA + areaB - interArea;
  return denom > 0 ? interArea / denom : 0;
}

function nonMaxSuppression(
  detections: YoloDetection[],
  iouThreshold: number,
  maxDetections: number
) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const keep: YoloDetection[] = [];

  for (const det of sorted) {
    if (keep.length >= maxDetections) break;
    const shouldDrop = keep.some(kept => iou(det.box, kept.box) > iouThreshold);
    if (!shouldDrop) {
      keep.push(det);
    }
  }

  return keep;
}

function decodeRowsLayout(
  data: Float32Array,
  numClasses: number,
  labels: string[],
  confThreshold: number
) {
  const stride = numClasses + 5;
  const rows = Math.floor(data.length / stride);
  const detections: YoloDetection[] = [];

  for (let row = 0; row < rows; row += 1) {
    const offset = row * stride;
    const x = data[offset];
    const y = data[offset + 1];
    const w = data[offset + 2];
    const h = data[offset + 3];
    const obj = data[offset + 4];

    if (obj <= 0) continue;

    let bestClass = 0;
    let bestProb = 0;
    for (let cls = 0; cls < numClasses; cls += 1) {
      const p = data[offset + 5 + cls];
      if (p > bestProb) {
        bestProb = p;
        bestClass = cls;
      }
    }

    const score = obj * bestProb;
    if (score < confThreshold) continue;

    const x1 = x - w / 2;
    const y1 = y - h / 2;
    const x2 = x + w / 2;
    const y2 = y + h / 2;

    detections.push({
      box: { x1, y1, x2, y2 },
      score,
      classId: bestClass,
      label: labels[bestClass] ?? `cls_${bestClass}`,
    });
  }

  return detections;
}

function decodeChannelsFirstLayout(
  data: Float32Array,
  numClasses: number,
  labels: string[],
  confThreshold: number
) {
  const stride = numClasses + 5;
  const detections: YoloDetection[] = [];

  const totalAnchors = Math.floor(data.length / stride);
  for (let anchor = 0; anchor < totalAnchors; anchor += 1) {
    const x = data[anchor];
    const y = data[anchor + totalAnchors];
    const w = data[anchor + totalAnchors * 2];
    const h = data[anchor + totalAnchors * 3];
    const obj = data[anchor + totalAnchors * 4];

    if (obj <= 0) continue;

    let bestClass = 0;
    let bestProb = 0;
    for (let cls = 0; cls < numClasses; cls += 1) {
      const p = data[anchor + totalAnchors * (5 + cls)];
      if (p > bestProb) {
        bestProb = p;
        bestClass = cls;
      }
    }

    const score = obj * bestProb;
    if (score < confThreshold) continue;

    detections.push({
      box: { x1: x - w / 2, y1: y - h / 2, x2: x + w / 2, y2: y + h / 2 },
      score,
      classId: bestClass,
      label: labels[bestClass] ?? `cls_${bestClass}`,
    });
  }

  return detections;
}

export function decodeYoloOutput(raw: Float32Array, options: DecodeOptions): YoloDetection[] {
  const {
    numClasses,
    inputWidth,
    inputHeight,
    confThreshold,
    iouThreshold,
    maxDetections,
    labels,
    layout = 'rows',
  } = options;

  const decoded =
    layout === 'channels_first'
      ? decodeChannelsFirstLayout(raw, numClasses, labels, confThreshold)
      : decodeRowsLayout(raw, numClasses, labels, confThreshold);

  const normalized = decoded.map(det => ({
    ...det,
    box: {
      x1: det.box.x1 / inputWidth,
      y1: det.box.y1 / inputHeight,
      x2: det.box.x2 / inputWidth,
      y2: det.box.y2 / inputHeight,
    },
  }));

  return nonMaxSuppression(normalized, iouThreshold, maxDetections);
}
