import { els } from "./dom";
import { bindModal, setStatus } from "./modal";
import { listCameras, startCamera, stopCamera } from "./camera";
import { bindBadgeControls, loadBadgeSettings } from "./badge";

import type { SegmentationRunner } from "./segmentation/runner";
import { OnnxSegmentationRunner } from "./segmentation/onnxRunner";
import { TFLiteSegmentationRunner } from "./segmentation/tfliteRunner";
import { attachMaskOverlay } from "./overlay";

async function init() {
  loadBadgeSettings();
  bindBadgeControls();

  await listCameras();
  navigator.mediaDevices?.addEventListener?.("devicechange", listCameras);

  // ==== СЕГМЕНТАЦИЯ (инициализация) ====
  const segBack = (new URLSearchParams(location.search).get("seg") || "auto") as "auto" | "onnx" | "tflite";
  let runner: SegmentationRunner | null = null;
  // Пробуем ONNX WebGPU (лучший вариант), затем TFLite
  try {
    if (segBack === "auto" || segBack === "onnx") {
      if (!("gpu" in navigator)) throw new Error("WebGPU недоступен");
      runner = new OnnxSegmentationRunner("/models/yolo.onnx", { inputSize: 640, threshold: 0.5 });
      await runner.init();
      console.log("[seg] ONNX WebGPU готов");
    }
  } catch (e) {
    console.warn("[seg] ONNX WebGPU не инициализировался:", e);
    runner = null;
  }
  if (!runner && (segBack === "auto" || segBack === "tflite")) {
    try {
      runner = new TFLiteSegmentationRunner("/models/yolo.tflite", { inputSize: 640, threshold: 0.5 });
      await runner.init();
      console.log("[seg] tfjs-tflite (WASM) готов");
    } catch (e) {
      console.warn("[seg] tfjs-tflite не инициализировался:", e);
    }
  }

  // Оверлей
  const pane = els.video.parentElement as HTMLElement;
  const overlay = attachMaskOverlay(pane, els.video);

  // Кнопка камеры
  els.toggleCamBtn.addEventListener("click", async () => {
    const running = (window as any)._running === true;
    if (running) {
      stopCamera();
      (window as any)._running = false;
    } else {
      await startCamera();
      (window as any)._running = true;

      // Тёплый прогон сегментации
      if (runner) {
        try { await runner.warmup?.(els.video); } catch {}
      }
    }
  });

  // Модалка и смена камеры
  bindModal();
  const handleVideoSettingsChange = () => {
    if ((window as any)._running) {
      stopCamera();
      setTimeout(startCamera, 50);
    }
  };
  els.camSelect.addEventListener("change", handleVideoSettingsChange);

  // Грейсфул стоп
  window.addEventListener("pagehide", stopCamera);
  window.addEventListener("beforeunload", stopCamera);

  // ==== Цикл сегментации с троттлингом ====
  let busy = false;
  let lastTs = 0;
  const TARGET_FPS = 24;
  const MIN_DT = 1000 / TARGET_FPS;

  const tick = async (ts: number) => {
    if ((window as any)._running && runner && !busy) {
      if (ts - lastTs >= MIN_DT) {
        busy = true;
        try {
          const res = await runner.run(els.video);
          if (res) overlay.draw(res.data, res.width, res.height);
        } catch (e) {
          // тихо, чтобы не заспамить
        } finally {
          lastTs = ts;
          busy = false;
        }
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  // Ресайз оверлея при изменении размера
  const ro = new ResizeObserver(() => overlay.resizeToVideo());
  ro.observe(pane);
  setStatus("готов");
}

init().catch((e) => {
  console.error("Ошибка при инициализации приложения:", e);
  setStatus("Ошибка инициализации", "warn");
});
