'use strict';

import { els } from './dom.js';
import { MAX_SAMPLES } from './state.js';

export function setStatus(text, cls = '') {
  els.status.className = '';
  if (cls) els.status.classList.add(cls);
  els.status.textContent = `Статус: ${text}`;
}
export function clamp01(x) { return Math.max(0, Math.min(1, x)); }
export function avg(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0) / arr.length : NaN; }
export function fmtNum(n, d=1) { return Number.isFinite(n) ? n.toFixed(d) : '-'; }
export function pushSample(arr, v) { arr.push(v); if (arr.length > MAX_SAMPLES) arr.shift(); }