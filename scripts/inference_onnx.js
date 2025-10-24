'use strict';

const SIZE = 640;
let session = null;
let inFlight = false;
let lastOverlayBitmap = null;
let preprocessCanvas = null;
let preprocessCtx = null;
let prefer = 'webgpu'; // 'webgpu' | 'wasm'
let inputName = null;  // автоопределение имени входа
let outputName = null; // автоопределение выхода

// Буферы
let chwBuffer = null;      // Float32Array(1*3*640*640)
let rgbaBuffer = null;     // Uint8ClampedArray(640*640*4) — для оверлея

function makePreprocessCanvas() {
  if (preprocessCanvas) return;
  try {
    preprocessCanvas = new OffscreenCanvas(SIZE, SIZE);
  } catch {
    preprocessCanvas = document.createElement('canvas');
    preprocessCanvas.width = SIZE;
    preprocessCanvas.height = SIZE;
  }
  preprocessCtx = preprocessCanvas.getContext('2d', { willReadFrequently: true });
}

export function hasModel() { return !!session; }

export async function initSegmentation({ modelUrl, preferBackend = 'webgpu' } = {}) {
  prefer = preferBackend;
  makePreprocessCanvas();

  const ep = (prefer === 'webgpu' && ort.env.webgpu) ? 'webgpu' : 'wasm';
  const sessOpts = { executionProviders: [ep] };

  session = await ort.InferenceSession.create(modelUrl, sessOpts);

  // определим имена входа/выхода
  const inputs = session.inputNames;
  const outputs = session.outputNames;
  inputName = inputs[0];
  outputName = outputs[0];

  // буферы
  chwBuffer = new Float32Array(1 * 3 * SIZE * SIZE);
  rgbaBuffer = new Uint8ClampedArray(SIZE * SIZE * 4);

  // тёплый прогон нулями
  const dummy = new ort.Tensor('float32', chwBuffer, [1, 3, SIZE, SIZE]);
  await session.run({ [inputName]: dummy });
}

export function enqueueSegmentation(videoEl, { mirror = true } = {}) {
  if (!session || inFlight) return;
  inFlight = true;
  // fire-and-forget
  runOnce(videoEl, { mirror })
    .catch(console.error)
    .finally(() => { inFlight = false; });
}

export function getOverlayBitmap() { return lastOverlayBitmap; }

// внутреннее: один прогон
async function runOnce(videoEl, { mirror }) {
  makePreprocessCanvas();

  // препроцесс: 640x640 + зеркалирование, чтобы совпадало с рендером
  preprocessCtx.save();
  preprocessCtx.clearRect(0, 0, SIZE, SIZE);
  if (mirror) {
    preprocessCtx.setTransform(-1, 0, 0, 1, SIZE, 0);
    preprocessCtx.drawImage(videoEl, 0, 0, SIZE, SIZE);
  } else {
    preprocessCtx.setTransform(1, 0, 0, 1, 0, 0);
    preprocessCtx.drawImage(videoEl, 0, 0, SIZE, SIZE);
  }
  preprocessCtx.restore();

  const img = preprocessCtx.getImageData(0, 0, SIZE, SIZE);
  const data = img.data; // Uint8ClampedArray

  // HWC->CHW, нормализация 0..1
  const plane = SIZE * SIZE;
  const R = 0 * plane, G = 1 * plane, B = 2 * plane;
  for (let i = 0, px = 0; px < plane; px++, i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    chwBuffer[R + px] = r;
    chwBuffer[G + px] = g;
    chwBuffer[B + px] = b;
  }
  const input = new ort.Tensor('float32', chwBuffer, [1, 3, SIZE, SIZE]);

  // инференс
  const out = await session.run({ [inputName]: input });
  let mask = out[outputName];

  let maskData, shape = mask.dims;
  if (shape.length === 4) { // NCHW
    maskData = mask.data;
  } else if (shape.length === 3) { // N H W
    maskData = mask.data;
  } else {
    maskData = mask.data; // H W
  }

  // постпроцесс
  const alpha = 160;
  for (let px = 0; px < SIZE * SIZE; px++) {
    const v = maskData[px];                // 0..1
    const inv = 1 - (v > 0.5 ? 1 : 0);     // инвертированный бинарный
    const g = inv ? 255 : 0;
    const a = inv ? alpha : 0;
    const i = px * 4;
    rgbaBuffer[i + 0] = 0;   // R
    rgbaBuffer[i + 1] = g;   // G
    rgbaBuffer[i + 2] = 0;   // B
    rgbaBuffer[i + 3] = a;   // A
  }

  const overlayImageData = new ImageData(rgbaBuffer, SIZE, SIZE);
  let overlaySourceCanvas;
  try {
    overlaySourceCanvas = new OffscreenCanvas(SIZE, SIZE);
  } catch {
    overlaySourceCanvas = document.createElement('canvas');
    overlaySourceCanvas.width = SIZE;
    overlaySourceCanvas.height = SIZE;
  }
  const octx = overlaySourceCanvas.getContext('2d');
  octx.putImageData(overlayImageData, 0, 0);

  if (lastOverlayBitmap) {
    try { lastOverlayBitmap.close?.(); } catch {}
  }
  lastOverlayBitmap = await createImageBitmap(overlaySourceCanvas);
}
