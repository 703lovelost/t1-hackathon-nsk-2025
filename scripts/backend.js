'use strict';

async function canUseWebGPU() {
  if (!('gpu' in navigator)) return false;
  try { return !!(await navigator.gpu.requestAdapter()); }
  catch { return false; }
}

export async function pickBestBackend() {
  try {
    if (await canUseWebGPU()) {
      await tf.setBackend('webgpu');
      await tf.ready();
      if (tf.getBackend() === 'webgpu') return 'webgpu';
    }
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