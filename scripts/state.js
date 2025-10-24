'use strict';

export let mediaStream = null;
export let running = false;
export let rafId = null;
export let lastTs = 0;

export const fpsSamples = [];
export const cpuSamples = [];
export const gpuSamples = [];
export const MAX_SAMPLES = 120;
export let frameIdx = 0;

export function setRunning(v) { running = v; }
export function setMediaStream(v) { mediaStream = v; }
export function setRafId(v) { rafId = v; }
export function setLastTs(v) { lastTs = v; }
export function resetSamples() {
  fpsSamples.length = 0;
  cpuSamples.length = 0;
  gpuSamples.length = 0;
  frameIdx = 0;
}
export function incFrame() { frameIdx++; }
