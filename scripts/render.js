// scripts/render.js
'use strict';

import { els } from './dom.js';
import {
  running, lastTs, fpsSamples, cpuSamples, gpuSamples,
  setRafId, setLastTs, incFrame, frameIdx
} from './state.js';
import { avg, fmtNum, clamp01, pushSample } from './utils.js';
import { gpuProbe } from './gpu.js';
import { enqueueSegmentation, getOverlayBitmap, hasModel } from './inference_onnx.js';

const INFER_EVERY = 2;       // инференс маски раз в 2 кадра (под iGPU можно 3–4)
const GPU_PROBE_EVERY = 10;  // реже пробуем GPU, чтобы не мешать рендеру

let lastGpuProbeAt = 0;

// Создаём панель и canvas для маски, если их нет в DOM (устраняет TypeError по maskCanvas)
function ensureMaskCanvas() {
  if (els.maskCanvas && els.maskCanvas.getContext) return els.maskCanvas;

  const panes = document.querySelector('.panes');
  if (!panes) return null;

  const pane = document.createElement('div');
  pane.className = 'pane';

  const h2 = document.createElement('h2');
  h2.textContent = 'Выход модели (маска)';

  const canvas = document.createElement('canvas');
  canvas.id = 'maskCanvas';
  canvas.width = 1024;
  canvas.height = 1080;
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  canvas.style.background = '#000';
  canvas.style.borderRadius = '12px';
  canvas.style.aspectRatio = '16 / 9';

  pane.appendChild(h2);
  pane.appendChild(canvas);
  panes.appendChild(pane);

  // сохранить ссылку в общем объекте элементов
  els.maskCanvas = canvas;
  return canvas;
}

// одинаковая геометрия с препроцессингом: letterbox + зеркалирование
function drawLetterboxed(ctx, src, dstW, dstH, mirror) {
  const iw = src.videoWidth || 640;
  const ih = src.videoHeight || 480;
  const r  = Math.min(dstW / iw, dstH / ih);
  const nw = Math.round(iw * r);
  const nh = Math.round(ih * r);
  const dx = Math.floor((dstW - nw) / 2);
  const dy = Math.floor((dstH - nh) / 2);

  ctx.save();
  ctx.clearRect(0, 0, dstW, dstH);
  if (mirror) {
    ctx.setTransform(-1, 0, 0, 1, dstW, 0);
    ctx.drawImage(src, dx, dy, nw, nh);
  } else {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(src, dx, dy, nw, nh);
  }
  ctx.restore();
}

export function startLoop() {
  const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  const ctx = els.canvas.getContext('2d', { alpha: false });

  // гарантируем наличие maskCanvas перед первым кадром
  const maskCanvas = ensureMaskCanvas();
  const maskCtx = maskCanvas ? maskCanvas.getContext('2d', { alpha: false }) : null;

  const render = async () => {
    if (!running) return;

    const frameStart = performance.now();

    if (els.video.readyState >= 2) {
      // Синхронизация размеров canvas с входным видео (1:1), UI масштабируется CSS'ом
      const vw = els.video.videoWidth  || 640;
      const vh = els.video.videoHeight || 480;
      if (els.canvas.width !== vw || els.canvas.height !== vh) {
        els.canvas.width = vw; els.canvas.height = vh;
      }
      if (maskCanvas && (maskCanvas.width !== vw || maskCanvas.height !== vh)) {
        maskCanvas.width = vw; maskCanvas.height = vh;
      }

      // 1) Рисуем видео letterbox'ом + зеркалим (как в препроцессинге)
      drawLetterboxed(ctx, els.video, els.canvas.width, els.canvas.height, true);

      // 2) Фоновый инференс сегментации каждые N кадров (fire-and-forget)
      if (hasModel() && (frameIdx % INFER_EVERY) === 0) {
        enqueueSegmentation(els.video, { mirror: true });
      }

      // 3) Наложение зелёной инвертированной маски поверх видео + отдельный вывод маски
      const overlay = getOverlayBitmap();
      if (overlay) {
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1;
        ctx.drawImage(overlay, 0, 0, els.canvas.width, els.canvas.height);

        if (maskCtx) {
          maskCtx.save();
          maskCtx.globalCompositeOperation = 'source-over';
          maskCtx.globalAlpha = 1;
          maskCtx.fillStyle = '#000';
          maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
          maskCtx.drawImage(overlay, 0, 0, maskCanvas.width, maskCanvas.height);
          maskCtx.restore();
        }
      } else if (maskCtx) {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      }

      // 4) Неблокирующая оценка GPU utilization
      const now = performance.now();
      const dtMs = now - lastTs;

      if (now - lastGpuProbeAt >= GPU_PROBE_EVERY * (1000 / Math.max(1, avg(fpsSamples) || 30))) {
        lastGpuProbeAt = now;
        gpuProbe(dtMs).then(res => {
          if (res) {
            const val = res.gpuUtil;
            pushSample(gpuSamples, val);
            els.gpuNow.textContent = `GPU: ${fmtNum(val)}%`;
            const gAvg2 = avg(gpuSamples);
            els.gpuAvg.textContent = `GPUAvg: ${Number.isFinite(gAvg2) ? fmtNum(gAvg2) + '%' : '-%'}`;
          }
        }).catch(()=>{});
      }

      // 5) FPS/CPU метрики
      const afterDraw = performance.now();
      const dt = afterDraw - lastTs;
      const busy = afterDraw - frameStart;
      const fps = 1000 / Math.max(1, dt);
      const cpu = clamp01(busy / Math.max(1, dt)) * 100;

      pushSample(fpsSamples, fps);
      pushSample(cpuSamples, cpu);

      els.fpsNow.textContent = `FPS: ${fmtNum(fps)} fps`;
      els.fpsAvg.textContent = `FPSAvg: ${fmtNum(avg(fpsSamples))} fps`;
      els.cpuNow.textContent = `CPU: ${fmtNum(cpu)}%`;
      els.cpuAvg.textContent = `CPUAvg: ${fmtNum(avg(cpuSamples))}%`;

      const gAvg = avg(gpuSamples);
      els.gpuAvg.textContent = `GPUAvg: ${Number.isFinite(gAvg) ? fmtNum(gAvg) + '%' : '-%'}`;

      setLastTs(afterDraw);
      incFrame();
    }

    if (hasRVFC) {
      els.video.requestVideoFrameCallback(() => { render(); });
    } else {
      setRafId(requestAnimationFrame(render));
    }
  };

  render();
}
