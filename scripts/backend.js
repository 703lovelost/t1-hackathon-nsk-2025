'use strict';

async function canUseWebGPU() {
  if (!('gpu' in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
    return !!adapter;
  } catch { return false; }
}

export async function pickBestBackend() {
  try {
    if (await canUseWebGPU()) {
      await tf.setBackend('webgpu');
      await tf.ready();
      if (tf.getBackend() === 'webgpu') return 'webgpu';
    }
  } catch {} // тихо откатываемся

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
