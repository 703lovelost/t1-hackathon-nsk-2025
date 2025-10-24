'use strict';

const SIZE = 640;
let session = null;
let inFlight = false;
let lastOverlayBitmap = null;
let canvasW = SIZE, canvasH = SIZE;

let preprocessCanvas = null;
let preprocessCtx = null;

let prefer = 'webgpu';
let inputName = null;
let outputName = null;

let chwBuffer = null;              // Float32 [1,3,H,W]
let rgbaBuffer = null;             // Uint8Clamped [H*W*4]

// Тюнинги под датасет
const USE_LETTERBOX = true;
const CHANNELS_BGR = false;        // true, если модель обучалась на BGR
const NORM_01 = true;              // 0..1
const USE_IMAGENET_NORM = false;   // mean/std = 0.485,0.456,0.406 / 0.229,0.224,0.225
const APPLY_SIGMOID = true;        // если выход — логиты

const mean = USE_IMAGENET_NORM ? [0.485, 0.456, 0.406] : [0,0,0];
const std  = USE_IMAGENET_NORM ? [0.229, 0.224, 0.225] : [1,1,1];

function makePreprocessCanvas(w = SIZE, h = SIZE) {
  if (!preprocessCanvas) {
    try { preprocessCanvas = new OffscreenCanvas(w, h); }
    catch { preprocessCanvas = Object.assign(document.createElement('canvas'), { width: w, height: h }); }
    preprocessCtx = preprocessCanvas.getContext('2d', { willReadFrequently: true });
  } else if (preprocessCanvas.width !== w || preprocessCanvas.height !== h) {
    preprocessCanvas.width = w; preprocessCanvas.height = h;
  }
}

export function setModelInputCanvasSize(w, h) {
  canvasW = w; canvasH = h;
  makePreprocessCanvas(w, h);
}

export function hasModel() { return !!session; }

export async function initSegmentation({ modelUrl, preferBackend = 'webgpu' } = {}) {
  prefer = preferBackend;
  makePreprocessCanvas(canvasW, canvasH);

  const ep = (prefer === 'webgpu' && ort.env.webgpu) ? 'webgpu' : 'wasm';
  const sessOpts = { executionProviders: [ep] };
  session = await ort.InferenceSession.create(modelUrl, sessOpts);

  inputName = session.inputNames[0];
  outputName = session.outputNames[0];

  chwBuffer = new Float32Array(1 * 3 * canvasH * canvasW);
  rgbaBuffer = new Uint8ClampedArray(canvasH * canvasW * 4);

  // тёплый прогон
  const dummy = new ort.Tensor('float32', chwBuffer, [1, 3, canvasH, canvasW]);
  await session.run({ [inputName]: dummy });
}

export function enqueueSegmentation(videoEl, { mirror = true } = {}) {
  if (!session || inFlight) return;
  inFlight = true;
  runOnce(videoEl, { mirror })
    .catch(console.error)
    .finally(() => { inFlight = false; });
}

export function getOverlayBitmap() { return lastOverlayBitmap; }

// --- утилиты ---
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function letterboxDraw(ctx, src, dstW, dstH, mirror) {
  const iw = src.videoWidth || src.width || dstW;
  const ih = src.videoHeight || src.height || dstH;
  const r = Math.min(dstW / iw, dstH / ih);
  const nw = Math.round(iw * r);
  const nh = Math.round(ih * r);
  const dx = Math.floor((dstW - nw) / 2);
  const dy = Math.floor((dstH - nh) / 2);

  ctx.save();
  ctx.clearRect(0, 0, dstW, dstH);
  // паддинги чёрные
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, dstW, dstH);

  if (mirror) {
    ctx.setTransform(-1, 0, 0, 1, dstW, 0);
    // Для зеркала смещаем область рисования
    ctx.drawImage(src, dx, dy, nw, nh);
  } else {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(src, dx, dy, nw, nh);
  }
  ctx.restore();

  return { dx, dy, nw, nh };
}

async function runOnce(videoEl, { mirror }) {
  makePreprocessCanvas(canvasW, canvasH);

  // 1) letterbox/resize в препроцесс canvas
  const info = USE_LETTERBOX
    ? letterboxDraw(preprocessCtx, videoEl, canvasW, canvasH, mirror)
    : (preprocessCtx.save(), preprocessCtx.clearRect(0,0,canvasW,canvasH),
       mirror ? preprocessCtx.setTransform(-1,0,0,1,canvasW,0) : preprocessCtx.setTransform(1,0,0,1,0,0),
       preprocessCtx.drawImage(videoEl, 0, 0, canvasW, canvasH),
       preprocessCtx.restore(), { dx:0, dy:0, nw:canvasW, nh:canvasH });

  // 2) забираем RGBA
  const img = preprocessCtx.getImageData(0, 0, canvasW, canvasH);
  const data = img.data;
  const plane = canvasW * canvasH;

  // 3) HWC->CHW + опциональные преобразования (BGR/mean/std)
  const C0 = CHANNELS_BGR ? 2 : 0; // если BGR, то порядок B,G,R
  const C1 = 1;
  const C2 = CHANNELS_BGR ? 0 : 2;

  const mean0 = mean[0], mean1 = mean[1], mean2 = mean[2];
  const std0  = std[0],  std1  = std[1],  std2  = std[2];

  const R = 0 * plane, G = 1 * plane, B = 2 * plane;

  for (let i = 0, px = 0; px < plane; px++, i += 4) {
    let r = data[i + 0] / (NORM_01 ? 255 : 1);
    let g = data[i + 1] / (NORM_01 ? 255 : 1);
    let b = data[i + 2] / (NORM_01 ? 255 : 1);

    // mean/std
    r = (r - mean0) / std0;
    g = (g - mean1) / std1;
    b = (b - mean2) / std2;

    // кладём в CHW с нужным порядком
    chwBuffer[(C0 === 0 ? R : (C0 === 1 ? G : B)) + px] = r;
    chwBuffer[(C1 === 0 ? R : (C1 === 1 ? G : B)) + px] = g;
    chwBuffer[(C2 === 0 ? R : (C2 === 1 ? G : B)) + px] = b;
  }

  const input = new ort.Tensor('float32', chwBuffer, [1, 3, canvasH, canvasW]);

  // 4) инференс
  const out = await session.run({ [inputName]: input });
  const tensor = out[outputName];

  // 5) приводим к плоскости [H*W], учитывая возможные оси
  let flat = null, dims = tensor.dims, buf = tensor.data;
  // Нормализуем к [H,W]
  if (dims.length === 4) { // [N,C,H,W] или [N,H,W,C]
    const [N,A,B,C] = dims;
    if (A === 1) {         // [1,1,H,W]
      flat = buf; // уже линейно по H*W
    } else if (C === 1) {  // [1,H,W,1] -> нужно перестроить
      // NHWC: компактно скопируем первый канал
      flat = new Float32Array(B * C); // но B здесь H, C здесь W — см. dims
      // аккуратнее:
      const H = B, W = C;
      flat = new Float32Array(H * W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          flat[y*W + x] = buf[(0*H + y)*W*1 + x*1 + 0]; // NHWC
        }
      }
    } else {
      // неизвестный формат — допустим первый канал в NCHW
      flat = new Float32Array(B * C);
      const H = B, W = C;
      const stride = H * W;
      for (let i = 0; i < stride; i++) flat[i] = buf[i]; // C0
    }
  } else if (dims.length === 3) { // [1,H,W]
    flat = buf;
  } else { // [H,W]
    flat = buf;
  }

  // 6) активация/порог/инверсия
  let min = +Infinity, max = -Infinity, ones = 0;
  for (let i = 0; i < flat.length; i++) {
    let v = flat[i];
    if (APPLY_SIGMOID) v = sigmoid(v);
    min = Math.min(min, v); max = Math.max(max, v);
    const bin = v > 0.5 ? 1 : 0;
    const inv = 1 - bin; // инверсия
    if (inv) ones++;
    const j = i * 4;
    rgbaBuffer[j+0] = 0;
    rgbaBuffer[j+1] = inv ? 255 : 0;
    rgbaBuffer[j+2] = 0;
    rgbaBuffer[j+3] = inv ? 160 : 0;
  }
  // Диагностика в консоль (1 раз в несколько прогонов можно оставить)
  if (Math.random() < 0.05) {
    console.log(`[mask] min=${min.toFixed(3)} max=${max.toFixed(3)} green%=${(100*ones/flat.length).toFixed(1)}`);
  }

  // 7) bitmap для наложения
  const overlayImageData = new ImageData(rgbaBuffer, canvasW, canvasH);
  let overlaySourceCanvas;
  try { overlaySourceCanvas = new OffscreenCanvas(canvasW, canvasH); }
  catch {
    overlaySourceCanvas = document.createElement('canvas');
    overlaySourceCanvas.width = canvasW;
    overlaySourceCanvas.height = canvasH;
  }
  const octx = overlaySourceCanvas.getContext('2d');
  octx.putImageData(overlayImageData, 0, 0);

  if (lastOverlayBitmap) { try { lastOverlayBitmap.close?.(); } catch {} }
  lastOverlayBitmap = await createImageBitmap(overlaySourceCanvas);
}
