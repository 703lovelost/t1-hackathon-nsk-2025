// scripts/inference_onnx.js
'use strict';

const SIZE = 640;

// ---- ГЛОБАЛЬНАЯ ССЫЛКА НА onnxruntime-web (видна всем функциям модуля) ----
let ORT = null;
function getORT() {
  if (!ORT) ORT = globalThis.ort ?? null;
  return ORT;
}

// Режим работы сегментатора: 'onnx' (ORT WebGPU) или 'tfjs'
let MODE = 'onnx';
let INPUT_LAYOUT = null; // 'NCHW' | 'NHWC' (определяется автоматически при первом успешном прогоне)

// Состояние
let session = null;       // ORT session
let tfModel = null;       // TFJS GraphModel (альтернатива)
let inFlight = false;
let lastOverlayBitmap = null;
let canvasW = SIZE, canvasH = SIZE;

let preprocessCanvas = null;
let preprocessCtx = null;

let prefer = 'webgpu-onnx';
let inputName = null;
let outputName = null;

let chwBuffer = null;   // Float32 [1,3,H,W] (плоско)
let hwcBuffer = null;   // Float32 [1,H,W,3] (плоско)
let rgbaBuffer = null;  // Uint8Clamped [H*W*4]

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

export function hasModel() { return MODE === 'onnx' ? !!session : !!tfModel; }

export async function initSegmentation({ modelUrl, preferBackend = 'webgpu-onnx' } = {}) {
  prefer = preferBackend;
  MODE = (String(preferBackend).toLowerCase().includes('tfjs')) ? 'tfjs' : 'onnx';
  makePreprocessCanvas(canvasW, canvasH);

  // Буферы под вход и выход
  chwBuffer = new Float32Array(1 * 3 * canvasH * canvasW);
  hwcBuffer = new Float32Array(1 * canvasH * canvasW * 3);
  rgbaBuffer = new Uint8ClampedArray(canvasH * canvasW * 4);

  if (MODE === 'onnx') {
    const ns = getORT();
    if (!('gpu' in navigator) || !ns?.env?.webgpu) {
      throw new Error('WebGPU is not available for ORT.');
    }

    const sessOpts = { executionProviders: ['webgpu'] };
    session = await ns.InferenceSession.create(modelUrl, sessOpts);
    console.log('ORT EP picked:', session?.executionProvider ?? 'webgpu (requested)');

    inputName = session.inputNames[0];
    outputName = session.outputNames[0];

    // тёплый прогон — одновременно определим раскладку входа
    await warmupAndDetectLayout();
  } else {
    // TFJS GraphModel (например, экспорт YOLOv11m-seg в TFJS)
    tfModel = await tf.loadGraphModel(modelUrl);
    // тёплый прогон
    await tf.tidy(() => {
      const dummy = tf.zeros([1, canvasH, canvasW, 3], 'float32');
      const out = tfModel.executeAsync ? tfModel.executeAsync(dummy) : tfModel.execute(dummy);
      // обрабатываем промис, чтобы реально выполнить граф
      if (out instanceof Promise) return out.then(xs => (Array.isArray(xs) ? xs.forEach(t => t.dispose?.()) : xs.dispose?.()));
      (Array.isArray(out) ? out.forEach(t => t.dispose?.()) : out.dispose?.());
    });
  }
}

async function warmupAndDetectLayout() {
  // Подготовим оба буфера и попробуем сначала NCHW, затем NHWC.
  // Заполним нулями — это неважно для shape-чека.
  chwBuffer.fill(0);
  hwcBuffer.fill(0);

  const ns = getORT();
  const tryNCHW = async () => {
    const t = new ns.Tensor('float32', chwBuffer, [1, 3, canvasH, canvasW]);
    const o = await session.run({ [inputName]: t });
    disposeOrtOutputs(o);
    INPUT_LAYOUT = 'NCHW';
  };
  const tryNHWC = async () => {
    const t = new ns.Tensor('float32', hwcBuffer, [1, canvasH, canvasW, 3]);
    const o = await session.run({ [inputName]: t });
    disposeOrtOutputs(o);
    INPUT_LAYOUT = 'NHWC';
  };

  try {
    await tryNCHW();
  } catch (e1) {
    console.warn('[onnx] NCHW warmup failed, retry NHWC', e1);
    await tryNHWC();
  }
  console.log('[onnx] input layout detected:', INPUT_LAYOUT);
}

function disposeOrtOutputs(out) {
  // ORT JS отдаёт plain объекты с TypedArray; обычно диспоуз не требуется
  // оставлено на будущее, если появится handle
}

export function enqueueSegmentation(videoEl, { mirror = true } = {}) {
  if ((MODE === 'onnx' && !session) || (MODE === 'tfjs' && !tfModel) || inFlight) return;
  inFlight = true;
  (MODE === 'onnx' ? runOnceORT(videoEl, { mirror }) : runOnceTFJS(videoEl, { mirror }))
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
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, dstW, dstH);

  if (mirror) {
    ctx.setTransform(-1, 0, 0, 1, dstW, 0);
    ctx.drawImage(src, dx, dy, nw, nh);
  } else {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(src, dx, dy, nw, nh);
  }
  ctx.restore();
}

function fillInputBuffersFromRGBA(data, plane) {
  // Заполняем одновременно CHW и HWC (одним проходом)
  const mean0 = mean[0], mean1 = mean[1], mean2 = mean[2];
  const std0  = std[0],  std1  = std[1],  std2  = std[2];

  const Roff = 0 * plane, Goff = 1 * plane, Boff = 2 * plane;

  for (let i = 0, px = 0; px < plane; px++, i += 4) {
    let r = data[i + 0] / (NORM_01 ? 255 : 1);
    let g = data[i + 1] / (NORM_01 ? 255 : 1);
    let b = data[i + 2] / (NORM_01 ? 255 : 1);

    r = (r - mean0) / std0;
    g = (g - mean1) / std1;
    b = (b - mean2) / std2;

    // CHW (возможен BGR порядок каналов)
    const C0 = CHANNELS_BGR ? 2 : 0;
    const C1 = 1;
    const C2 = CHANNELS_BGR ? 0 : 2;
    const idx0 = (C0 === 0 ? Roff : (C0 === 1 ? Goff : Boff)) + px;
    const idx1 = (C1 === 0 ? Roff : (C1 === 1 ? Goff : Boff)) + px;
    const idx2 = (C2 === 0 ? Roff : (C2 === 1 ? Goff : Boff)) + px;
    chwBuffer[idx0] = r; chwBuffer[idx1] = g; chwBuffer[idx2] = b;

    // HWC (RGB всегда; если нужна BGR — переставим ниже при создании тензора)
    const j = px * 3;
    hwcBuffer[j + 0] = r;
    hwcBuffer[j + 1] = g;
    hwcBuffer[j + 2] = b;
  }
}

async function runOnceORT(videoEl, { mirror }) {
  const ns = getORT();
  if (!ns) return;

  makePreprocessCanvas(canvasW, canvasH);

  // 1) letterbox/resize в препроцесс canvas
  if (USE_LETTERBOX) {
    letterboxDraw(preprocessCtx, videoEl, canvasW, canvasH, mirror);
  } else {
    preprocessCtx.save();
    preprocessCtx.clearRect(0, 0, canvasW, canvasH);
    mirror ? preprocessCtx.setTransform(-1,0,0,1,canvasW,0) : preprocessCtx.setTransform(1,0,0,1,0,0);
    preprocessCtx.drawImage(videoEl, 0, 0, canvasW, canvasH);
    preprocessCtx.restore();
  }

  // 2) RGBA -> оба буфера (CHW и HWC)
  const img = preprocessCtx.getImageData(0, 0, canvasW, canvasH);
  const data = img.data;
  const plane = canvasW * canvasH;
  fillInputBuffersFromRGBA(data, plane);

  // 3) инференс (с авто-выбором раскладки один раз)
  let tensor = null, out = null;
  const t0 = performance.now();
  try {
    if (INPUT_LAYOUT === 'NHWC') {
      tensor = new ns.Tensor('float32', hwcBuffer, [1, canvasH, canvasW, 3]);
    } else { // по умолчанию пробуем NCHW
      tensor = new ns.Tensor('float32', chwBuffer, [1, 3, canvasH, canvasW]);
    }
    out = await session.run({ [inputName]: tensor });
  } catch (e) {
    if (INPUT_LAYOUT == null) {
      // Первая попытка не удалась — пробуем другую раскладку
      try {
        tensor = new ns.Tensor('float32', hwcBuffer, [1, canvasH, canvasW, 3]);
        out = await session.run({ [inputName]: tensor });
        INPUT_LAYOUT = 'NHWC';
        console.log('[onnx] switched to NHWC');
      } catch (e2) {
        console.error('[onnx] run failed for both layouts', e, e2);
        return;
      }
    } else {
      console.error('[onnx] run failed', e);
      return;
    }
  }
  if (INPUT_LAYOUT == null) INPUT_LAYOUT = 'NCHW';
  const inferMs = performance.now() - t0;

  // 4) постпроцесс выхода в flat[H*W]
  const outTensor = out[outputName];
  const flat = toFlatHW(outTensor);
  console.debug(`[onnx] run ok: ${inferMs.toFixed(1)} ms, dims=${outTensor.dims?.join('×')}`);

  // 5) порог + инверсия → RGBA
  applyThresholdToRGBA(flat);

  // 6) bitmap для наложения
  await buildOverlayBitmap();
}

async function runOnceTFJS(videoEl, { mirror }) {
  makePreprocessCanvas(canvasW, canvasH);

  // 1) letterbox/resize в препроцесс canvas
  if (USE_LETTERBOX) {
    letterboxDraw(preprocessCtx, videoEl, canvasW, canvasH, mirror);
  } else {
    preprocessCtx.save();
    preprocessCtx.clearRect(0, 0, canvasW, canvasH);
    mirror ? preprocessCtx.setTransform(-1,0,0,1,canvasW,0) : preprocessCtx.setTransform(1,0,0,1,0,0);
    preprocessCtx.drawImage(videoEl, 0, 0, canvasW, canvasH);
    preprocessCtx.restore();
  }

  // 2) RGBA -> HWC
  const img = preprocessCtx.getImageData(0, 0, canvasW, canvasH);
  const data = img.data;
  const plane = canvasW * canvasH;
  fillInputBuffersFromRGBA(data, plane);

  // 3) инференс TFJS (ожидаем [1,H,W,3] → [1,H,W,1] или совместимые)
  let flat = null;
  await tf.tidy(async () => {
    let input4d = tf.tensor4d(hwcBuffer, [1, canvasH, canvasW, 3], 'float32');
    const t0 = performance.now();
    const out = tfModel.executeAsync ? await tfModel.executeAsync(input4d) : tfModel.execute(input4d);
    const y = Array.isArray(out) ? out[0] : out;
    const inferMs = performance.now() - t0;
    console.debug(`[tfjs] run ok: ${inferMs.toFixed(1)} ms, shape=${y.shape?.join('×')}`);

    // Приводим к [H,W]
    if (y.rank === 4 && y.shape[0] === 1 && y.shape[3] === 1) {
      flat = (await y.data());
    } else if (y.rank === 3 && y.shape[0] === 1) {
      flat = (await y.data());
    } else {
      // Попытка свести к первой карте
      const y0 = y.squeeze(); // [H,W] или [H,W,C]
      const y1 = y0.rank === 3 ? y0.slice([0,0,0],[y0.shape[0], y0.shape[1], 1]).squeeze() : y0;
      flat = await y1.data();
    }
  });

  // 4) порог + инверсия → RGBA
  applyThresholdToRGBA(flat);

  // 5) bitmap для наложения
  await buildOverlayBitmap();
}

function toFlatHW(tensor) {
  const buf = tensor.data, dims = tensor.dims || [];
  if (dims.length === 4) { // [N,C,H,W] или [N,H,W,C]
    const [N,A,B,C] = dims;
    if (A === 1) {         // [1,1,H,W]
      return buf;
    } else if (C === 1) {  // [1,H,W,1] (NHWC)
      const H = B, W = C;
      const flat = new Float32Array(H * W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          flat[y*W + x] = buf[(0*H + y)*W*1 + x];
        }
      }
      return flat;
    } else {
      // допустим NCHW и берём C0
      const H = B, W = C;
      const stride = H * W;
      const flat = new Float32Array(stride);
      for (let i = 0; i < stride; i++) flat[i] = buf[i];
      return flat;
    }
  } else if (dims.length === 3) { // [1,H,W]
    return buf;
  } else { // [H,W] или неизвестно — возвращаем как есть
    return buf;
  }
}

function applyThresholdToRGBA(flat) {
  // динамический порог: если min/max сильно сжаты, берём середину диапазона
  let min = +Infinity, max = -Infinity;
  for (let i = 0; i < flat.length; i++) {
    const v = APPLY_SIGMOID ? sigmoid(flat[i]) : flat[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const th = (max - min > 1e-6) ? 0.5 * (min + max) : 0.5;

  let ones = 0;
  for (let i = 0; i < flat.length; i++) {
    let v = APPLY_SIGMOID ? sigmoid(flat[i]) : flat[i];
    const bin = v > th ? 1 : 0;
    const inv = 1 - bin; // инверсия: фон зелёный, объект прозрачен
    if (inv) ones++;
    const j = i * 4;
    rgbaBuffer[j+0] = 0;
    rgbaBuffer[j+1] = inv ? 255 : 0;
    rgbaBuffer[j+2] = 0;
    rgbaBuffer[j+3] = inv ? 160 : 0;
  }
  if (Math.random() < 0.15) {
    console.log(`[mask] min=${min.toFixed(3)} max=${max.toFixed(3)} th=${th.toFixed(3)} green(inv)%=${(100*ones/flat.length).toFixed(1)}`);
  }
}

async function buildOverlayBitmap() {
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
