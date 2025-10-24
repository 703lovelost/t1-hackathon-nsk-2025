// scripts/camera.js
'use strict';

import { els } from './dom.js';
import { listCameras } from './devices.js';
import { pickBestBackend } from './backend.js';
import { setStatus } from './utils.js';
import {
  mediaStream, running, rafId,
  setRunning, setMediaStream, setRafId, setLastTs, resetSamples
} from './state.js';
import { startLoop } from './render.js';
import { initSegmentation } from './inference_onnx.js';

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

    // Сегментация: сначала пытаемся ORT WebGPU (best.onnx), при неудаче — TFJS (например, YOLOv11m-seg в TFJS)
    let segOk = false;
    try {
      await initSegmentation({ modelUrl: './models/best.onnx', preferBackend: 'webgpu-onnx' });
      console.log('Segmentation session ready: ORT webgpu');
      segOk = true;
    } catch (e) {
      console.warn('ONNX init failed:', e);
      setStatus('WebGPU/ORT недоступен, пробую TFJS-модель…', 'warn');
      try {
        // Укажите путь к TFJS-модели сегментации (например, экспорт YOLOv11m-seg в TFJS)
        await initSegmentation({ modelUrl: './models/yolo11m_tfjs/model.json', preferBackend: 'tfjs' });
        console.log('Segmentation model ready: TFJS');
        segOk = true;
      } catch (e2) {
        console.warn('TFJS init failed:', e2);
        setStatus('Сегментация отключена (ORT/TFJS не инициализировались)', 'warn');
      }
    }

    setRunning(true);
    els.stopBtn.disabled = false;
    setStatus(segOk ? 'камера запущена (сегментация включена)' : 'камера запущена (без сегментации)', segOk ? 'ok' : '');
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
