import * as tf from "@tensorflow/tfjs";
import { els } from "./dom";
import { MAX_SAMPLES, fpsSamples, cpuSamples, gpuSamples, lastTs, setLastTs, running, incFrame } from "./state";

function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }
function avg(arr: number[]) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN; }
function fmtNum(n: number, d = 1) { return Number.isFinite(n) ? n.toFixed(d) : "-"; }
function pushSample(arr: number[], v: number) { arr.push(v); if (arr.length > MAX_SAMPLES) arr.shift(); }

async function gpuProbe(dtMs: number): Promise<{ gpuMs: number; gpuUtil: number } | null> {
  const b = tf.getBackend();
  if (!["webgpu", "webgl", "wasm"].includes(b)) return null;
  try {
    const res = await tf.time(() => tf.tidy(() => {
      let t = tf.browser.fromPixels(els.video);
      t = tf.image.resizeBilinear(t, [160, 90], true).toFloat().mul(1 / 255);
      const r = t.mean();
      return r;
    }));
    const gpuMs = (res.kernelMs ?? res.wallMs ?? 0) as number;
    const gpuUtil = clamp01(gpuMs / Math.max(1, dtMs)) * 100;
    return { gpuMs, gpuUtil };
  } catch { return null; }
}

export function startLoop(): void {
  const hasRVFC = typeof HTMLVideoElement !== "undefined" && "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  const GPU_PROBE_EVERY = 15;

  const render = async () => {
    if (!((window as any).___running ?? true)) {/* no-op */}
    if (!document.body) return; // крайний случай

    if (els.video && els.video.readyState >= 2) {
      const frameStart = performance.now();
      let gpuUtilNow: number | null = null;

      const now = performance.now();
      const dtMs = now - lastTs;
      if ((window as any)._frameIdx % GPU_PROBE_EVERY === 0) {
        const res = await gpuProbe(dtMs);
        if (res) { gpuUtilNow = res.gpuUtil; pushSample(gpuSamples, gpuUtilNow); }
      }

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
      if (gpuUtilNow != null) els.gpuNow.textContent = `GPU: ${fmtNum(gpuUtilNow)}%`;
      const gAvg = avg(gpuSamples);
      els.gpuAvg.textContent = `GPUAvg: ${Number.isFinite(gAvg) ? fmtNum(gAvg) + "%" : "-%"}`;

      setLastTs(afterDraw);
      (window as any)._frameIdx = ((window as any)._frameIdx || 0) + 1;
    }

    if ((window as any)._running !== false) {
      if (hasRVFC && els.video.requestVideoFrameCallback) {
        try { els.video.requestVideoFrameCallback(render as any); }
        catch { requestAnimationFrame(render); }
      } else {
        requestAnimationFrame(render);
      }
    }
  };

  (window as any)._running = true;
  render();
}