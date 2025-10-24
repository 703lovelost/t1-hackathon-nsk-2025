'use strict';

import { els } from './dom.js';
import { clamp01 } from './utils.js';

export async function gpuProbe(dtMs) {
  const b = tf.getBackend();
  if (!['webgpu','webgl'].includes(b)) return null;
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