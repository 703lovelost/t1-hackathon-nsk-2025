'use strict';

import { listCameras } from './devices.js';
import { startCamera, stopCamera } from './camera.js';
import { els } from './dom.js';

async function init() {
  await listCameras();
  navigator.mediaDevices?.addEventListener?.('devicechange', listCameras);

  els.startBtn.addEventListener('click', startCamera);
  els.stopBtn.addEventListener('click', stopCamera);

  window.addEventListener('pagehide', stopCamera);
  window.addEventListener('beforeunload', stopCamera);

  window.debugORT = () => {
  const out = {
    hasGlobal: !!globalThis.ort,
    hasWebGPU: 'gpu' in navigator,
    ortWebGPUFlag: !!globalThis.ort?.env?.webgpu,
  };
  console.table(out);
  return out;
};
}

init();