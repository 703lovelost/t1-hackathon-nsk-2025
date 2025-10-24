import { els } from "./dom";
import { bindModal, setStatus } from "./modal";
import { listCameras, startCamera, stopCamera } from "./camera";
import { bindBadgeControls, loadBadgeSettings } from "./badge";
import { engine } from "./inference";
import { pickBestBackend } from "./backend";

/**
 * Точка входа приложения.
 * Порядок:
 * 1) Инициализация TFJS-бэкенда (WebGPU → WASM → WebGL → CPU)
 * 2) Загрузка настроек бейджа и биндинг контролов
 * 3) Параллельная загрузка модели сегментации
 * 4) Инициализация камер и обработчиков devicechange
 * 5) UI/модалка/хоткеи
 * 6) Грейсфул остановка камеры при уходе со страницы
 */
async function init() {
  // 1) Гарантируем готовность TFJS-бэкенда до любых операций с моделью
  const backend = await pickBestBackend();
  console.log("[TFJS] backend:", backend);

  // 2) Бейдж: загрузка настроек и биндинг контролов
  loadBadgeSettings();
  bindBadgeControls();

  // 3) Модель — грузим/прогреваем заранее (параллельно остальному)
  engine.load("/models/yolo11m-seg_web_model_tfjs/model.json").catch((e) =>
    console.error("Model load error:", e)
  );

  // 4) Камеры
  await listCameras();
  if ((navigator.mediaDevices as any)?.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", listCameras);
  }

  // 5) Кнопка старта/стопа камеры
  els.toggleCamBtn.addEventListener("click", () => {
    const running = (window as any)._running === true;
    if (running) {
      stopCamera();
      (window as any)._running = false;
      engine.clearOverlay(); // очистим оверлей при остановке
    } else {
      startCamera();
      (window as any)._running = true;
    }
  });

  // 6) Модальное окно и табы
  bindModal();

  // Перезапуск камеры при смене устройства
  const handleVideoSettingsChange = () => {
    if ((window as any)._running) {
      console.log("Настройки видео изменились, перезапускаем камеру…");
      stopCamera();
      setTimeout(startCamera, 50);
    }
  };
  els.camSelect.addEventListener("change", handleVideoSettingsChange);

  // Хоткеи: Space — старт/стоп, Esc — закрыть модалку (если открыта)
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      els.toggleCamBtn.click();
    } else if (e.code === "Escape") {
      const modal = els.settingsModal as HTMLElement;
      if (getComputedStyle(modal).display !== "none") {
        (document.getElementById("closeSettingsBtn") as HTMLElement)?.click();
      }
    }
  });

  // Грейсфул стоп камеры при уходе со страницы
  window.addEventListener("pagehide", stopCamera);
  window.addEventListener("beforeunload", stopCamera);

  setStatus("готов");
}

init().catch((e) => {
  console.error("Ошибка при инициализации приложения:", e);
  setStatus("Ошибка инициализации", "warn");
});
