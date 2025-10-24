'use strict';

import { els } from './dom.js';

export async function listCameras() {
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