'use strict';

import { els } from './dom.js';
import { listCameras } from './devices.js';
import { pickBestBackend } from './backend.js';
import { setStatus, fmtNum } from './utils.js';
import {
  mediaStream, running, rafId,
  setRunning, setMediaStream, setRafId, setLastTs, resetSamples
} from './state.js';
import { startLoop } from './render.js';
import { initSegmentation } from './inference_onnx.js';

// Возможно, low-power валит обращение к webgpu. Пока в комментариях.

// async function canUseOrtWebGPU() {
//   if (!('gpu' in navigator) || !ort?.env?.webgpu) return false;
//   try {
//     const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
//     return !!adapter;
//   } catch { return false; }
// }

export async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('getUserMedia не поддерживается', 'warn'); return;
  }
  els.startBtn.disabled = true;
  setStatus('запрашиваю доступ к камере…');

  const deviceId = els.camSelect.value || undefined;
  const constraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
      : { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    setMediaStream(stream);
    els.video.srcObject = stream;
    els.video.setAttribute('playsinline', '');
    els.video.muted = true;
    await els.video.play();

    await listCameras();

    await new Promise(r => {
      if (els.video.readyState >= 2) return r();
      els.video.onloadeddata = () => r();
    });

    const backend = await pickBestBackend();
    console.log('TF.js backend:', backend);

    try {
      // const preferOrt = (await canUseOrtWebGPU()) ? 'webgpu' : 'wasm';
      await initSegmentation({ modelUrl: './models/best.onnx', preferBackend: 'webgpu' });
      console.log('ONNX session ready:', preferOrt);
    } catch (e) {
      console.warn('ONNX init failed:', e);
      setStatus('WebGPU для ONNX недоступен (маска выключена)', 'warn');
    }

    setRunning(true);
    els.stopBtn.disabled = false;
    setStatus('камера запущена', 'ok');
    setLastTs(performance.now());
    resetSamples();

    startLoop();
  } catch (e) {
    console.error(e);
    setStatus(`ошибка доступа: ${e?.name || e}`, 'warn');
    els.startBtn.disabled = false;
  }
}

export function stopCamera() {
  setRunning(false);
  if (rafId) cancelAnimationFrame(rafId), setRafId(null);
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  setMediaStream(null);
  els.video.srcObject = null;
  els.stopBtn.disabled = true;
  els.startBtn.disabled = false;
  setStatus('остановлено');
  els.fpsNow.textContent  = 'FPS: - fps';
  els.fpsAvg.textContent  = 'FPSAvg: - fps';
  els.cpuNow.textContent  = 'CPU: -%';
  els.cpuAvg.textContent  = 'CPUAvg: -%';
  els.gpuNow.textContent  = 'GPU: -%';
  els.gpuAvg.textContent  = 'GPUAvg: -%';
}
