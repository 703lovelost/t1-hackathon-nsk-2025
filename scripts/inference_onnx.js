// scripts/inference_onnx.js
'use strict';

import { els } from './dom.js';

const SIZE = 640;

// ---- ГЛОБАЛЬНАЯ ССЫЛКА НА onnxruntime-web (видна всем функциям модуля) ----
let ORT = null;
function getORT() {
  if (!ORT) ORT = globalThis.ort ?? null;
  return ORT;
}

// Состояние
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
let previewRGBA = null;            // Uint8Clamped [H*W*4] для <img>

let lastMaskUrl = null;            // objectURL для <img id="maskImg">

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

  const ns = getORT();
  if (!('gpu' in navigator) || !ns?.env?.webgpu) {
    throw new Error('WebGPU is not available.');
  }

  const sessOpts = { executionProviders: ['webgpu'] };
  session = await ns.InferenceSession.create(modelUrl, sessOpts);
  console.log('ORT EP picked:', session?.executionProvider ?? 'webgpu (requested)');

  inputName = session.inputNames[0];
  outputName = session.outputNames[0];

  chwBuffer = new Float32Array(1 * 3 * canvasH * canvasW);
  rgbaBuffer = new Uint8ClampedArray(canvasH * canvasW * 4);
  previewRGBA = new Uint8ClampedArray(canvasH * canvasW * 4);

  // тёплый прогон
  const dummy = new ns.Tensor('float32', chwBuffer, [1, 3, canvasH, canvasW]);
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
  return { dx, dy, nw, nh };
}

async function runOnce(videoEl, { mirror }) {
  const ns = getORT();
  if (!ns) {
    console.warn('[onnx] ORT namespace not available');
    return;
  }

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

  // 2) забираем RGBA
  const img = preprocessCtx.getImageData(0, 0, canvasW, canvasH);
  const data = img.data;
  const plane = canvasW * canvasH;

  // 3) HWC->CHW + нормализации
  const C0 = CHANNELS_BGR ? 2 : 0;
  const C1 = 1;
  const C2 = CHANNELS_BGR ? 0 : 2;

  const mean0 = mean[0], mean1 = mean[1], mean2 = mean[2];
  const std0  = std[0],  std1  = std[1],  std2  = std[2];

  const R = 0 * plane, G = 1 * plane, B = 2 * plane;

  for (let i = 0, px = 0; px < plane; px++, i += 4) {
    let r = data[i + 0] / (NORM_01 ? 255 : 1);
    let g = data[i + 1] / (NORM_01 ? 255 : 1);
    let b = data[i + 2] / (NORM_01 ? 255 : 1);

    r = (r - mean0) / std0;
    g = (g - mean1) / std1;
    b = (b - mean2) / std2;

    chwBuffer[(C0 === 0 ? R : (C0 === 1 ? G : B)) + px] = r;
    chwBuffer[(C1 === 0 ? R : (C1 === 1 ? G : B)) + px] = g;
    chwBuffer[(C2 === 0 ? R : (C2 === 1 ? G : B)) + px] = b;
  }

  // 4) инференс
  const input = new ns.Tensor('float32', chwBuffer, [1, 3, canvasH, canvasW]);
  const t0 = performance.now();
  const out = await session.run({ [inputName]: input });
  const tensor = out[outputName];
  const inferMs = performance.now() - t0;
  console.debug(`[onnx] run ok: ${inferMs.toFixed(1)} ms, dims=${tensor.dims?.join('×')}, type=${tensor.type ?? 'float32'}`);

  // 5) приводим к плоскости [H*W], учитывая возможные оси
  let flat = null, dims = tensor.dims, buf = tensor.data;
  if (dims.length === 4) { // [N,C,H,W] или [N,H,W,C]
    const [N,A,B,C] = dims;
    if (A === 1) {         // [1,1,H,W]
      flat = buf;
    } else if (C === 1) {  // [1,H,W,1] (NHWC)
      const H = B, W = C;
      flat = new Float32Array(H * W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          flat[y*W + x] = buf[(0*H + y)*W*1 + x]; // NHWC
        }
      }
    } else {
      // допустим NCHW и берём C0
      const H = B, W = C;
      const stride = H * W;
      flat = new Float32Array(stride);
      for (let i = 0; i < stride; i++) flat[i] = buf[i];
    }
  } else if (dims.length === 3) { // [1,H,W]
    flat = buf;
  } else { // [H,W]
    flat = buf;
  }

  // 6) активация/порог/ИНВЕРСИЯ (зелёный хромакей по фону) + подготовка превью
  let min = +Infinity, max = -Infinity, ones = 0;
  for (let i = 0; i < flat.length; i++) {
    let v = flat[i];
    if (APPLY_SIGMOID) v = sigmoid(v);
    min = Math.min(min, v); max = Math.max(max, v);
    const bin = v > 0.5 ? 1 : 0;
    const inv = 1 - bin; // инверсия

    // зелёный overlay
    const j = i * 4;
    rgbaBuffer[j+0] = 0;
    rgbaBuffer[j+1] = inv ? 255 : 0;
    rgbaBuffer[j+2] = 0;
    rgbaBuffer[j+3] = inv ? 160 : 0;

    // превью для <img>: белый = 255 для inv, чёрный = 0
    const g = inv ? 255 : 0;
    previewRGBA[j+0] = g;
    previewRGBA[j+1] = g;
    previewRGBA[j+2] = g;
    previewRGBA[j+3] = 255;

    if (inv) ones++;
  }
  if (Math.random() < 0.1) {
    console.log(`[mask] min=${min.toFixed(3)} max=${max.toFixed(3)} green(inv)%=${(100*ones/flat.length).toFixed(1)}`);
  }

  // 7a) bitmap для наложения на основной canvas
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

  // 7b) превью в <img> (objectURL PNG)
  const prevCanvas = document.createElement('canvas');
  prevCanvas.width = canvasW;
  prevCanvas.height = canvasH;
  const pctx = prevCanvas.getContext('2d');
  pctx.putImageData(new ImageData(previewRGBA, canvasW, canvasH), 0, 0);

  prevCanvas.toBlob(blob => {
    if (!blob) return;
    if (lastMaskUrl) URL.revokeObjectURL(lastMaskUrl);
    lastMaskUrl = URL.createObjectURL(blob);
    if (els.maskImg) els.maskImg.src = lastMaskUrl;
  }, 'image/png');
}
