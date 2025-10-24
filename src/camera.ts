import * as tf from "@tensorflow/tfjs";
import { els } from "./dom";
import { pickBestBackend } from "./backend";
import { mediaStream, setRunning, running, setRafId, setLastTs, fpsSamples, cpuSamples, gpuSamples, frameIdx as frameIndexRef } from "./state";
import { startLoop } from "./render";
import { setStatus, resetMetrics } from "./modal";

export async function listCameras(): Promise<void> {
  try {
    const devices = (await navigator.mediaDevices?.enumerateDevices?.()) ?? [];
    const cams = devices.filter((d) => d.kind === "videoinput");

    const currentCamId = els.camSelect.value;
    els.camSelect.innerHTML = "";

    cams.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.text = cam.label || `Камера ${i + 1}`;
      els.camSelect.appendChild(opt);
    });

    if (cams.some((c) => c.deviceId === currentCamId)) {
      els.camSelect.value = currentCamId;
    }
    els.camSelect.disabled = cams.length <= 1;
  } catch (e) {
    console.error("Ошибка при получении списка камер:", e);
    setStatus("ошибка списка камер", "warn");
    els.camSelect.disabled = true;
  }
}

export async function startCamera(): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("getUserMedia не поддерживается", "warn");
    return;
  }

  els.toggleCamBtn.disabled = true;
  setStatus("запрашиваю доступ к камере…");

  const videoConstraints = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  const deviceId = els.camSelect.value || undefined;
  const videoBase = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" as const };
  const constraints: MediaStreamConstraints = { audio: false, video: { ...videoBase, ...videoConstraints } };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    (window as any)._streamRef = stream; // отладка, если нужно

    els.video.srcObject = stream;
    els.video.setAttribute("playsinline", "");
    els.video.muted = true;
    await els.video.play();

    await listCameras();

    await new Promise<void>((resolve, reject) => {
      if (els.video.readyState >= 2) return resolve();
      els.video.onloadeddata = () => resolve();
      els.video.onerror = (e) => reject(new Error("Ошибка загрузки видео: " + (e as any)?.message));
    });

    const backend = await pickBestBackend();
    console.log("TF.js backend:", backend);
    console.log("Using constraints:", constraints);

    setRunning(true);
    els.toggleCamBtn.disabled = false;
    els.toggleCamBtn.classList.add("is-active");
    setStatus("камера запущена", "ok");

    setLastTs(performance.now());
    fpsSamples.length = 0;
    cpuSamples.length = 0;
    gpuSamples.length = 0;
    (window as any)._frameIdx = 0;

    startLoop();
  } catch (e: any) {
    console.error("Ошибка доступа к камере или запуска видео:", e);
    setStatus(`ошибка: ${e?.name || e?.message || e}`, "warn");
    els.toggleCamBtn.disabled = false;
    els.toggleCamBtn.classList.remove("is-active");
    await stopCamera();
  }
}

export async function stopCamera(): Promise<void> {
  setRunning(false);
  if (typeof cancelAnimationFrame === "function") {
    const id = (window as any)._rafId as number | null;
    if (id) cancelAnimationFrame(id);
    setRafId(null);
  }
  const stream = els.video.srcObject as MediaStream | null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  els.video.pause();
  els.toggleCamBtn.classList.remove("is-active");
  setStatus("остановлено");
  resetMetrics();
}