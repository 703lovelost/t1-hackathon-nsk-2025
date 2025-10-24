// scripts/render.js
'use strict';

import { els } from './dom.js';
import {
  running, lastTs, fpsSamples, cpuSamples, gpuSamples,
  setRafId, setLastTs, incFrame, frameIdx
} from './state.js';
import { avg, fmtNum, clamp01, pushSample } from './utils.js';
import { gpuProbe } from './gpu.js';
import { enqueueSegmentation, getOverlayBitmap, getProbBitmap, hasModel } from './inference_onnx.js';

const INFER_EVERY = 2;       // инференс маски раз в 2 кадра
const GPU_PROBE_EVERY = 10;  // реже пробуем GPU, чтобы не мешать рендеру

let lastGpuProbeAt = 0;
let maskCanvas, maskCtx;

// Создаём мини-панель для отдельного вывода «карты вероятностей»
function ensureMaskCanvas() {
  if (maskCanvas) return;
  const wrapper = document.createElement('div');
  Object.assign(wrapper.style, {
    position: 'fixed',
    right: '16px',
    bottom: '16px',
    background: 'rgba(0,0,0,0.6)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '12px',
    padding: '8px',
    zIndex: 9999,
    backdropFilter: 'blur(6px)',
  });

  const title = document.createElement('div');
  title.textContent = 'Model output (prob)';
  Object.assign(title.style, {
    fontSize: '12px',
    color: '#d9e1ee',
    margin: '0 0 6px',
    opacity: 0.8,
  });

  maskCanvas = document.createElement('canvas');
  maskCanvas.width = 320;
  maskCanvas.height = 180; // 16:9 по умолчанию, масштабируем дальше
  Object.assign(maskCanvas.style, {
    display: 'block',
    width: '320px',
    height: '180px',
    borderRadius: '8px',
    background: '#000',
  });
  maskCtx = maskCanvas.getContext('2d', { alpha: false });

  wrapper.appendChild(title);
  wrapper.appendChild(maskCanvas);
  document.body.appendChild(wrapper);
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
  ensureMaskCanvas();

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

      // 1) Рисуем видео letterbox'ом + зеркалим (как в препроцессинге)
      drawLetterboxed(ctx, els.video, els.canvas.width, els.canvas.height, true);

      // 2) Фоновый инференс сегментации каждые N кадров (fire-and-forget)
      if (hasModel() && (frameIdx % INFER_EVERY) === 0) {
        enqueueSegmentation(els.video, { mirror: true });
      }

      // 3) Наложение зелёной инвертированной маски поверх видео
      const overlay = getOverlayBitmap();
      if (overlay) {
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1;
        ctx.drawImage(overlay, 0, 0, els.canvas.width, els.canvas.height);
      }

      // 4) Отдельный вывод «карты вероятностей» модели
      const prob = getProbBitmap();
      if (prob && maskCtx) {
        // подгоняем мини-канвас под соотношение сторон карты
        const mw = prob.width;
        const mh = prob.height;
        const targetW = 320;
        const targetH = Math.round(targetW * (mh / mw));
        if (maskCanvas.width !== targetW || maskCanvas.height !== targetH) {
          maskCanvas.width = targetW;
          maskCanvas.height = targetH;
          maskCanvas.style.width = `${targetW}px`;
          maskCanvas.style.height = `${targetH}px`;
        }
        maskCtx.globalCompositeOperation = 'source-over';
        maskCtx.globalAlpha = 1;
        maskCtx.drawImage(prob, 0, 0, maskCanvas.width, maskCanvas.height);
      }

      // 5) Неблокирующая оценка GPU utilization
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

      // 6) FPS/CPU метрики
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