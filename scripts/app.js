'use strict';

// DOM
const els = {
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  camSelect: document.getElementById('cameraSelect'),
  startBtn: document.getElementById('startBtn'),
  stopBtn: document.getElementById('stopBtn'),
  status: document.getElementById('status'),
  fpsNow: document.getElementById('fps'),
  fpsAvg: document.getElementById('fpsAvg'),
  cpuNow: document.getElementById('cpuNow'),
  cpuAvg: document.getElementById('cpuAvg'),
  gpuNow: document.getElementById('gpuNow'),
  gpuAvg: document.getElementById('gpuAvg'),
};

// state
let mediaStream = null;
let running = false;
let rafId = null;
let lastTs = 0;

// FPS и загрузка
const fpsSamples = [];
const cpuSamples = []; // % main thread per frame
const gpuSamples = []; // % (tf kernel time / frame time)
const MAX_SAMPLES = 120;
let frameIdx = 0;

// ==== helpers UI ====
function setStatus(text, cls = '') {
  els.status.className = '';
  if (cls) els.status.classList.add(cls);
  els.status.textContent = `Статус: ${text}`;
}
function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function avg(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0) / arr.length : NaN; }
function fmtNum(n, d=1) { return Number.isFinite(n) ? n.toFixed(d) : '-'; }
function pushSample(arr, v) { arr.push(v); if (arr.length > MAX_SAMPLES) arr.shift(); }

// ==== выбор лучшего бэкенда TF ====
async function pickBestBackend() {
  try {
    await tf.setBackend('webgpu');
    await tf.ready();
    if (tf.getBackend() === 'webgpu') return 'webgpu';
  } catch {}

  try {
    if (typeof tf.setWasmPaths === 'function') {
      tf.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4/dist/');
    }
    tf.env().set('WASM_NUM_THREADS', Math.min(4, navigator.hardwareConcurrency || 4));
    await tf.setBackend('wasm');
    await tf.ready();
    if (tf.getBackend() === 'wasm') return 'wasm';
  } catch {}

  try {
    await tf.setBackend('webgl');
    await tf.ready();
    if (tf.getBackend() === 'webgl') return 'webgl';
  } catch {}

  await tf.setBackend('cpu');
  await tf.ready();
  return 'cpu';
}

// ==== devices ====
async function listCameras() {
  const devices = await navigator.mediaDevices?.enumerateDevices?.() ?? [];
  const cams = devices.filter(d => d.kind === 'videoinput');
  els.camSelect.innerHTML = '';
  cams.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.text = cam.label || `Камера ${i + 1}`;
    els.camSelect.appendChild(opt);
  });
  els.camSelect.disabled = cams.length <= 1;
}

// ==== start/stop camera ====
async function startCamera() {
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
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    els.video.srcObject = mediaStream;
    els.video.setAttribute('playsinline', '');
    els.video.muted = true;
    await els.video.play();

    await listCameras();

    // дождаться размеров видео
    await new Promise(r => {
      if (els.video.readyState >= 2) return r();
      els.video.onloadeddata = () => r();
    });

    // бэкенд TF
    const backend = await pickBestBackend();
    console.log('TF.js backend:', backend);

    running = true;
    els.stopBtn.disabled = false;
    setStatus('камера запущена', 'ok');
    lastTs = performance.now();
    fpsSamples.length = 0; cpuSamples.length = 0; gpuSamples.length = 0; frameIdx = 0;

    startLoop();
  } catch (e) {
    console.error(e);
    setStatus(`ошибка доступа: ${e?.name || e}`, 'warn');
    els.startBtn.disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId), (rafId = null);
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  mediaStream = null;
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

// ==== GPU probe (аккуратная, раз в N кадров) ====
async function gpuProbe(dtMs) {
  const b = tf.getBackend();
  if (!['webgpu','webgl','wasm'].includes(b)) return null;
  try {
    const res = await tf.time(() => tf.tidy(() => {
      let t = tf.browser.fromPixels(els.video);
      t = tf.image.resizeBilinear(t, [160, 90], true).toFloat().mul(1/255);
      return t.mean();
    }));
    const gpuMs = (res.kernelMs ?? res.wallMs ?? 0);
    const gpuUtil = clamp01(gpuMs / Math.max(1, dtMs)) * 100;
    return { gpuMs, gpuUtil };
  } catch {
    return null;
  }
}

// ==== основной цикл отрисовки ====
function startLoop() {
  const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  const ctx = els.canvas.getContext('2d', { alpha: false });

  const GPU_PROBE_EVERY = 15; // мерять GPU примерно раз в 15 кадров

  const render = async () => {
    if (!running) return;

    const frameStart = performance.now();

    if (els.video.readyState >= 2) {
      // синхронизируем размеры 1:1 с входным видео
      const vw = els.video.videoWidth  || 640;
      const vh = els.video.videoHeight || 480;
      if (els.canvas.width !== vw || els.canvas.height !== vh) {
        els.canvas.width = vw; els.canvas.height = vh;
      }

      // зеркалим вывод
      ctx.save();
      ctx.setTransform(-1, 0, 0, 1, els.canvas.width, 0);
      ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
      ctx.restore();

      // (опционально) проба GPU раз в N кадров
      let gpuUtilNow = null;
      const now = performance.now();
      const dtMs = now - lastTs;

      if ((frameIdx % GPU_PROBE_EVERY) === 0) {
        const res = await gpuProbe(dtMs);
        if (res) {
          gpuUtilNow = res.gpuUtil;
          pushSample(gpuSamples, gpuUtilNow);
        }
      }

      // FPS/CPU расчёты
      const afterDraw = performance.now();
      const dt = afterDraw - lastTs;
      const busy = afterDraw - frameStart;             // «занятость» JS на кадр
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

      lastTs = afterDraw;
      frameIdx++;
    }

    if (hasRVFC) {
      els.video.requestVideoFrameCallback(() => { render(); });
    } else {
      rafId = requestAnimationFrame(render);
    }
  };

  render();
}

// ==== инициализация, события ====
async function init() {
  await listCameras();
  navigator.mediaDevices?.addEventListener?.('devicechange', listCameras);
  els.startBtn.addEventListener('click', startCamera);
  els.stopBtn.addEventListener('click', stopCamera);

  window.addEventListener('pagehide', stopCamera);
  window.addEventListener('beforeunload', stopCamera);
}
init();
