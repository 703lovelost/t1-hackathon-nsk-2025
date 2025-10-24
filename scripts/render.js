'use strict';

import { els } from './dom.js';
import {
  running, lastTs, fpsSamples, cpuSamples, gpuSamples,
  setRafId, setLastTs, incFrame
} from './state.js';
import { avg, fmtNum, clamp01, pushSample } from './utils.js';
import { gpuProbe } from './gpu.js';
import { enqueueSegmentation, getOverlayBitmap } from './inference_onnx.js';

const INFER_EVERY = 2;

const GPU_PROBE_EVERY = 15;
let lastGpuProbeAt = 0;

export function startLoop() {
  const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  const ctx = els.canvas.getContext('2d', { alpha: false });
  const GPU_PROBE_EVERY = 10;

  const render = async () => {
    if (!running) return;

    const frameStart = performance.now();

    if (els.video.readyState >= 2) {
      const vw = els.video.videoWidth  || 640;
      const vh = els.video.videoHeight || 480;
      if (els.canvas.width !== vw || els.canvas.height !== vh) {
        els.canvas.width = vw; els.canvas.height = vh;
      }

      // зеркалим
      ctx.save();
      ctx.setTransform(-1, 0, 0, 1, els.canvas.width, 0);
      ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
      ctx.restore();

      if ((fpsSamples.length % INFER_EVERY) === 0) {
        enqueueSegmentation(els.video, { mirror: true });
      }

      const overlay = getOverlayBitmap();
      if (overlay) {
        ctx.drawImage(overlay, 0, 0, els.canvas.width, els.canvas.height);
      }

      let gpuUtilNow = null;
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

      // FPS/CPU
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

      if (gpuUtilNow != null) {
        els.gpuNow.textContent = `GPU: ${fmtNum(gpuUtilNow)}%`;
      }
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