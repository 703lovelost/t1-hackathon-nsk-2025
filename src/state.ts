import type { BadgeSettings } from "./types";

export const MAX_SAMPLES = 120;

export let mediaStream: MediaStream | null = null;
export let running = false;
export let rafId: number | null = null;
export let lastTs = 0;
export const fpsSamples: number[] = [];
export const cpuSamples: number[] = [];
export const gpuSamples: number[] = [];
export let frameIdx = 0;

export function setRunning(v: boolean) { running = v; }
export function setRafId(id: number | null) { rafId = id; }
export function setLastTs(ts: number) { lastTs = ts; }
export function incFrame() { frameIdx++; }

export let badgeSettings: BadgeSettings = {} as BadgeSettings;
export function setBadgeSettings(s: BadgeSettings) { badgeSettings = s; }