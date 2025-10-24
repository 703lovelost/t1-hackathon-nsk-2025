"use strict";

// DOM Elements
const els = {
  video: document.getElementById("video"),
  camSelect: document.getElementById("cameraSelect"),
  status: document.getElementById("status"),
  fpsNow: document.getElementById("fps"),
  fpsAvg: document.getElementById("fpsAvg"),
  cpuNow: document.getElementById("cpuNow"),
  cpuAvg: document.getElementById("cpuAvg"),
  gpuNow: document.getElementById("gpuNow"),
  gpuAvg: document.getElementById("gpuAvg"),
  settingsBtn: document.getElementById("settingsBtn"),
  settingsModal: document.getElementById("settingsModal"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  toggleCamBtn: document.getElementById("toggleCamBtn"), // === НОВЫЙ ОБЪЕКТ: Все элементы для бейджа ===

  badge: {
    // --- Элементы оверлея ---
    overlay: document.getElementById("smartBadgeOverlay"),
    badge: document.querySelector(".smart-badge"),
    logoContainer: document.getElementById("badge-logo-container"),
    logoImg: document.getElementById("badge-logo-img"),

    mainInfo: document.querySelector(".badge-main"),
    nameText: document.getElementById("badge-name-text"),
    companyText: document.getElementById("badge-company-text"),

    details: document.querySelector(".badge-details"), // Элементы .detail-item
    itemPosition: document.getElementById("badge-item-position"),
    itemDepartment: document.getElementById("badge-item-department"),
    itemLocation: document.getElementById("badge-item-location"),
    itemTelegram: document.getElementById("badge-item-telegram"),
    itemEmail: document.getElementById("badge-item-email"),
    itemSlogan: document.getElementById("badge-item-slogan"), // Текст/ссылки внутри .detail-item
    positionText: document.getElementById("badge-position-text"),
    departmentText: document.getElementById("badge-department-text"),
    locationText: document.getElementById("badge-location-text"),
    telegramLink: document.getElementById("badge-telegram-link"),
    emailLink: document.getElementById("badge-email-link"),
    sloganText: document.getElementById("badge-slogan-text"), // --- Элементы управления в модальном окне ---

    toggleShow: document.getElementById("badge-toggle-show"),

    // === ДОБАВЛЕНО: Ссылка на группу настроек ===
    settingsGroup: document.getElementById("badge-settings-group"),
    // === КОНЕЦ ДОБАВЛЕНИЯ ===

    positionRadios: document.querySelectorAll('input[name="badge-position"]'),
    logoTypeRadios: document.querySelectorAll('input[name="badge-logo-type"]'),

    logoUrl: document.getElementById("badge-logo-url"),
    logoUpload: document.getElementById("badge-logo-upload"),
    logoWarning: document.getElementById("badge-logo-warning"),

    // === ИЗМЕНЕНО: Ссылки на элементы выбора цвета ===
    // Старые colorPrimary/Secondary теперь указывают на ТЕКСТОВЫЕ поля
    colorPrimary: document.getElementById("badge-field-color-primary"),
    colorSecondary: document.getElementById("badge-field-color-secondary"),
    // Новые элементы для самих палитр
    pickerColorPrimary: document.getElementById("badge-picker-color-primary"),
    pickerColorSecondary: document.getElementById(
      "badge-picker-color-secondary",
    ),
    // === КОНЕЦ ИЗМЕНЕНИЯ ===

    // Поля ввода
    fieldName: document.getElementById("badge-field-name"),

    fieldCompany: document.getElementById("badge-field-company"),
    fieldPosition: document.getElementById("badge-field-position"),
    fieldDepartment: document.getElementById("badge-field-department"),
    fieldLocation: document.getElementById("badge-field-location"),
    fieldTelegram: document.getElementById("badge-field-telegram"),
    fieldEmail: document.getElementById("badge-field-email"),
    fieldSlogan: document.getElementById("badge-field-slogan"), // Чекбоксы для полей

    toggleName: document.getElementById("badge-toggle-name"),
    toggleCompany: document.getElementById("badge-toggle-company"),
    togglePosition: document.getElementById("badge-toggle-position"),
    toggleDepartment: document.getElementById("badge-toggle-department"),
    toggleLocation: document.getElementById("badge-toggle-location"),
    toggleTelegram: document.getElementById("badge-toggle-telegram"),
    toggleEmail: document.getElementById("badge-toggle-email"),
    toggleSlogan: document.getElementById("badge-toggle-slogan"),
  },
};

// State variables
let mediaStream = null;
let running = false;
let rafId = null;
let lastTs = 0;
const fpsSamples = [];
const cpuSamples = [];
const gpuSamples = [];
const MAX_SAMPLES = 120;
let frameIdx = 0;

// === НОВОЕ: Глобальный объект настроек бейджа ===
let badgeSettings = {};

// ==== UI Helpers ====
function setStatus(text, cls = "") {
  if (!els.status) return; // Добавим проверку на всякий случай
  els.status.className = "";
  if (cls) els.status.classList.add(cls);
  els.status.textContent = `Статус: ${text}`;
}
function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}
function avg(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN;
}
function fmtNum(n, d = 1) {
  return Number.isFinite(n) ? n.toFixed(d) : "-";
}
function pushSample(arr, v) {
  arr.push(v);
  if (arr.length > MAX_SAMPLES) arr.shift();
}

// ==== TF.js backend selection (без изменений) ====
async function pickBestBackend() {
  try {
    await tf.setBackend("webgpu");
    await tf.ready();
    if (tf.getBackend() === "webgpu") return "webgpu";
  } catch {}
  try {
    if (typeof tf.setWasmPaths === "function") {
      tf.setWasmPaths(
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4/dist/",
      );
    }
    tf.env().set(
      "WASM_NUM_THREADS",
      Math.min(4, navigator.hardwareConcurrency || 4),
    );
    await tf.setBackend("wasm");
    await tf.ready();
    if (tf.getBackend() === "wasm") return "wasm";
  } catch {}
  try {
    await tf.setBackend("webgl");
    await tf.ready();
    if (tf.getBackend() === "webgl") return "webgl";
  } catch {}
  await tf.setBackend("cpu");
  await tf.ready();
  return "cpu";
}

// ==== Device listing (без изменений) ====
async function listCameras() {
  try {
    // Добавим try-catch для надежности
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
    if (els.camSelect) els.camSelect.disabled = true; // Проверка els.camSelect
  }
}

// ==== Start/stop camera functions (без изменений) ====
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("getUserMedia не поддерживается", "warn");
    return;
  }

  if (els.toggleCamBtn) els.toggleCamBtn.disabled = true; // Проверка
  setStatus("запрашиваю доступ к камере…");

  const videoConstraints = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  const deviceId = els.camSelect ? els.camSelect.value || undefined : undefined; // Проверка
  const videoBase = deviceId
    ? { deviceId: { exact: deviceId } }
    : { facingMode: "user" };
  const constraints = {
    audio: false,
    video: { ...videoBase, ...videoConstraints },
  };

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    if (!els.video) throw new Error("Элемент <video> не найден."); // Проверка
    els.video.srcObject = mediaStream;
    els.video.setAttribute("playsinline", "");
    els.video.muted = true;
    await els.video.play();

    await listCameras(); // Обновляем список ПОСЛЕ получения доступа

    await new Promise((r, reject) => {
      // Добавим reject
      if (!els.video)
        return reject("Элемент <video> не найден для loadeddata.");
      if (els.video.readyState >= 2) return r();
      els.video.onloadeddata = () => r();
      els.video.onerror = (e) => reject("Ошибка загрузки видео: " + e); // Обработка ошибки
    });

    const backend = await pickBestBackend();
    console.log("TF.js backend:", backend);
    console.log("Using constraints:", constraints);

    running = true;
    if (els.toggleCamBtn) {
      // Проверка
      els.toggleCamBtn.disabled = false;
      els.toggleCamBtn.classList.add("is-active");
    }
    setStatus("камера запущена", "ok");

    lastTs = performance.now();
    fpsSamples.length = 0;
    cpuSamples.length = 0;
    gpuSamples.length = 0;
    frameIdx = 0;
    startLoop();
  } catch (e) {
    console.error("Ошибка доступа к камере или запуска видео:", e);
    setStatus(`ошибка: ${e?.name || e?.message || e}`, "warn");
    if (els.toggleCamBtn) {
      // Проверка
      els.toggleCamBtn.disabled = false; // Разблокируем кнопку в любом случае
      els.toggleCamBtn.classList.remove("is-active");
    }
    stopCamera(); // Убедимся, что все остановлено
  }
}

function stopCamera() {
  running = false;
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (els.video) {
    els.video.srcObject = null;
    els.video.pause(); // Добавим паузу на всякий случай
  }
  if (els.toggleCamBtn) {
    // Проверка на существование элемента
    els.toggleCamBtn.classList.remove("is-active");
  }
  setStatus("остановлено"); // Сброс метрик (добавим проверки на null)
  if (els.fpsNow) els.fpsNow.textContent = "FPS: - fps";
  if (els.fpsAvg) els.fpsAvg.textContent = "FPSAvg: - fps";
  if (els.cpuNow) els.cpuNow.textContent = "CPU: -%";
  if (els.cpuAvg) els.cpuAvg.textContent = "CPUAvg: -%";
  if (els.gpuNow) els.gpuNow.textContent = "GPU: -%";
  if (els.gpuAvg) els.gpuAvg.textContent = "GPUAvg: -%";
}

// ==== GPU probe (без изменений) ====
async function gpuProbe(dtMs) {
  if (!els.video || els.video.paused || els.video.ended || !tf) return null; // Доп. проверки
  const b = tf.getBackend();
  if (!["webgpu", "webgl", "wasm"].includes(b)) return null;
  try {
    const res = await tf.time(() =>
      tf.tidy(() => {
        let t = tf.browser.fromPixels(els.video);
        t = tf.image
          .resizeBilinear(t, [160, 90], true)
          .toFloat()
          .mul(1 / 255);
        const result = t.mean(); // Сохраним результат перед dispose
        return result; // Возвращаем тензор
      }),
    ); // Тензор результата будет удален tf.tidy, поэтому используем kernelMs/wallMs
    const gpuMs = res.kernelMs ?? res.wallMs ?? 0;
    const gpuUtil = clamp01(gpuMs / Math.max(1, dtMs)) * 100;
    return { gpuMs, gpuUtil };
  } catch (e) {
    // console.warn("Ошибка при gpuProbe:", e); // Можно раскомментировать для отладки
    return null;
  }
}

// ==== Main render loop (без изменений) ====
function startLoop() {
  const hasRVFC =
    typeof HTMLVideoElement !== "undefined" &&
    "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  const GPU_PROBE_EVERY = 15;
  const render = async () => {
    if (!running) return;
    const frameStart = performance.now();
    if (els.video && els.video.readyState >= 2) {
      // Проверка els.video
      let gpuUtilNow = null;
      const now = performance.now();
      const dtMs = now - lastTs;
      if (frameIdx % GPU_PROBE_EVERY === 0) {
        const res = await gpuProbe(dtMs);
        if (res) {
          gpuUtilNow = res.gpuUtil;
          pushSample(gpuSamples, gpuUtilNow);
        }
      }
      const afterDraw = performance.now();
      const dt = afterDraw - lastTs;
      const busy = afterDraw - frameStart;
      const fps = 1000 / Math.max(1, dt);
      const cpu = clamp01(busy / Math.max(1, dt)) * 100;
      pushSample(fpsSamples, fps);
      pushSample(cpuSamples, cpu); // Обновление метрик с проверками
      if (els.fpsNow) els.fpsNow.textContent = `FPS: ${fmtNum(fps)} fps`;
      if (els.fpsAvg)
        els.fpsAvg.textContent = `FPSAvg: ${fmtNum(avg(fpsSamples))} fps`;
      if (els.cpuNow) els.cpuNow.textContent = `CPU: ${fmtNum(cpu)}%`;
      if (els.cpuAvg)
        els.cpuAvg.textContent = `CPUAvg: ${fmtNum(avg(cpuSamples))}%`;
      if (gpuUtilNow != null && els.gpuNow) {
        els.gpuNow.textContent = `GPU: ${fmtNum(gpuUtilNow)}%`;
      }
      const gAvg = avg(gpuSamples);
      if (els.gpuAvg)
        els.gpuAvg.textContent = `GPUAvg: ${
          Number.isFinite(gAvg) ? fmtNum(gAvg) + "%" : "-%"
        }`;
      lastTs = afterDraw;
      frameIdx++;
    }
    if (running) {
      // Добавим проверку running перед рекурсивным вызовом
      if (hasRVFC && els.video) {
        // Проверка els.video
        try {
          // Используем bind, чтобы сохранить контекст this для requestVideoFrameCallback
          els.video.requestVideoFrameCallback(render.bind(this));
        } catch (e) {
          // Обработаем возможную ошибку если видео уже недоступно
          console.warn("Ошибка в requestVideoFrameCallback:", e);
          rafId = requestAnimationFrame(render); // Фоллбэк на rAF
        }
      } else {
        rafId = requestAnimationFrame(render);
      }
    }
  };
  render();
}

// ==== Settings Modal (без изменений) ====
function openSettings() {
  if (els.settingsModal) els.settingsModal.style.display = "flex";
}
function closeSettings() {
  if (els.settingsModal) els.settingsModal.style.display = "none";
}

// === НОВЫЕ ФУНКЦИИ: Управление бейджем ===

// === ДОБАВЛЕНО: Новая функция для скрытия/показа ===
/**
 * Показывает или скрывает группу настроек бейджа
 * в зависимости от главного переключателя.
 */
function toggleBadgeSettingsVisibility() {
  if (!els.badge.toggleShow || !els.badge.settingsGroup) return; // Проверка

  if (els.badge.toggleShow.checked) {
    els.badge.settingsGroup.classList.remove("hidden");
  } else {
    els.badge.settingsGroup.classList.add("hidden");
  }
}
// === КОНЕЦ ДОБАВЛЕНИЯ ===

/**
 * Рендерит бейдж на основе текущего объекта badgeSettings
 */
function renderBadge() {
  const b = els.badge;
  const s = badgeSettings; // Проверка, что элементы бейджа существуют

  if (
    !b ||
    !b.overlay ||
    !b.badge ||
    !b.logoContainer ||
    !b.logoImg ||
    !b.mainInfo ||
    !b.nameText ||
    !b.companyText ||
    !b.details
  ) {
    console.error("Не все элементы бейджа найдены в DOM для рендеринга.");
    return;
  } // 1. Показать/Скрыть оверлей

  b.overlay.style.display = s.show ? "block" : "none";
  if (!s.show) return; // Если скрыт, дальше не рендерим
  // 2. Позиция

  b.overlay.className = "badge-overlay-container"; // Сброс
  //
  // === ИСПРАВЛЕНИЕ 1 ===
  //
  if (s.badgePosition && typeof s.badgePosition === "string") {
    b.overlay.classList.add(s.badgePosition);
  } else {
    b.overlay.classList.add("pos-bottom-left");
  } // 3. Цвета

  b.badge.style.backgroundColor = s.colorPrimary;
  b.badge.style.outlineColor = s.colorSecondary; // 4. Логотип

  const logoSrc = s.logoType === "upload" ? s.logoDataUrl : s.logoUrl;
  if (logoSrc) {
    b.logoImg.src = logoSrc;
    b.logoContainer.style.display = "block";
  } else {
    b.logoContainer.style.display = "none";
    b.logoImg.src = ""; // Очистим src, если лого нет
  } // 5. Поля
  // ФИО

  b.nameText.textContent = s.name;
  b.nameText.style.display = s.showName && s.name ? "block" : "none"; // Компания
  b.companyText.textContent = s.company;
  b.companyText.style.display = s.showCompany && s.company ? "block" : "none"; // Скрываем .badge-main, если оба поля пустые

  b.mainInfo.style.display =
    (s.showName && s.name) || (s.showCompany && s.company) ? "block" : "none"; //
  // === ИСПРАВЛЕНИЕ 2 ===
  //
  // Должность (с проверками элементов)

  if (b.positionText) b.positionText.textContent = s.jobPosition;
  if (b.itemPosition)
    b.itemPosition.style.display =
      s.showPosition && s.jobPosition ? "flex" : "none"; // Департамент
  if (b.departmentText) b.departmentText.textContent = s.department;
  if (b.itemDepartment)
    b.itemDepartment.style.display =
      s.showDepartment && s.department ? "flex" : "none"; // Локация
  if (b.locationText) b.locationText.textContent = s.location;
  if (b.itemLocation)
    b.itemLocation.style.display =
      s.showLocation && s.location ? "flex" : "none"; // Слоган
  if (b.sloganText) b.sloganText.textContent = s.slogan;
  if (b.itemSlogan)
    b.itemSlogan.style.display = s.showSlogan && s.slogan ? "flex" : "none"; // Telegram

  if (b.itemTelegram && b.telegramLink) {
    // Проверка элементов
    if (s.showTelegram && s.telegram) {
      const username = s.telegram.replace(/^@/, "");
      b.telegramLink.textContent = s.telegram;
      b.telegramLink.href = `https://t.me/${username}`;
      b.itemTelegram.style.display = "flex"; // Динамическая покраска акцентных элементов
      b.telegramLink.style.color = s.colorSecondary;
      const telegramIcon = b.itemTelegram.querySelector("i");
      if (telegramIcon) {
        telegramIcon.style.color = s.colorSecondary;
        telegramIcon.style.opacity = "1";
      }
    } else {
      b.itemTelegram.style.display = "none";
    }
  } // Email

  if (b.itemEmail && b.emailLink) {
    // Проверка элементов
    if (s.showEmail && s.email) {
      b.emailLink.textContent = s.email;
      b.emailLink.href = `mailto:${s.email}`;
      b.itemEmail.style.display = "flex";
    } else {
      b.itemEmail.style.display = "none";
    }
  } //
  // === ИСПРАВЛЕНИЕ 3 ===
  //
  // Проверяем, есть ли вообще детали для показа

  const hasDetails =
    (s.showPosition && s.jobPosition) ||
    (s.showDepartment && s.department) ||
    (s.showLocation && s.location) ||
    (s.showTelegram && s.telegram) ||
    (s.showEmail && s.email) ||
    (s.showSlogan && s.slogan);

  b.details.style.display = hasDetails ? "block" : "none";
}

/**
 * Собирает все настройки из модального окна, сохраняет в localStorage и вызывает renderBadge
 */
function handleBadgeSettingChange() {
  const b = els.badge;
  const currentPositionRadio = document.querySelector(
    'input[name="badge-position"]:checked',
  );
  const currentLogoTypeRadio = document.querySelector(
    'input[name="badge-logo-type"]:checked',
  ); // Проверка, что все нужные элементы управления существуют

  if (!b || !b.toggleShow || !currentPositionRadio || !currentLogoTypeRadio) {
    console.error(
      "Не найдены основные элементы управления бейджем в DOM при сохранении.",
    );
    return;
  } //
  // === ИСПРАВЛЕНИЕ 4 (КЛЮЧЕВОЕ) ===
  //
  // Собираем все значения в объект

  badgeSettings = {
    show: b.toggleShow.checked,
    badgePosition: currentPositionRadio.value, // <--- ИЗМЕНЕНО (было 'position')
    logoType: currentLogoTypeRadio.value, // Берем значение из найденного элемента
    logoUrl: b.logoUrl ? b.logoUrl.value : "", // Проверки на null
    logoDataUrl: badgeSettings.logoDataUrl || "", // Сохраняем старое значение
    colorPrimary: b.colorPrimary ? b.colorPrimary.value : "#0052CC",
    colorSecondary: b.colorSecondary ? b.colorSecondary.value : "#00B8D9",

    name: b.fieldName ? b.fieldName.value : "",
    company: b.fieldCompany ? b.fieldCompany.value : "",
    jobPosition: b.fieldPosition ? b.fieldPosition.value : "", // <--- ИЗМЕНЕНО (было 'position')
    department: b.fieldDepartment ? b.fieldDepartment.value : "",
    location: b.fieldLocation ? b.fieldLocation.value : "",
    telegram: b.fieldTelegram ? b.fieldTelegram.value : "",
    email: b.fieldEmail ? b.fieldEmail.value : "",
    slogan: b.fieldSlogan ? b.fieldSlogan.value : "",

    showName: b.toggleName ? b.toggleName.checked : true,
    showCompany: b.toggleCompany ? b.toggleCompany.checked : true,
    showPosition: b.togglePosition ? b.togglePosition.checked : false,
    showDepartment: b.toggleDepartment ? b.toggleDepartment.checked : false,
    showLocation: b.toggleLocation ? b.toggleLocation.checked : false,
    showTelegram: b.toggleTelegram ? b.toggleTelegram.checked : false,
    showEmail: b.toggleEmail ? b.toggleEmail.checked : false,
    showSlogan: b.toggleSlogan ? b.toggleSlogan.checked : false,
  }; // Сохраняем в localStorage

  try {
    localStorage.setItem("badgeSettings", JSON.stringify(badgeSettings));
  } catch (e) {
    console.error("Ошибка сохранения настроек бейджа в localStorage:", e);
  } // Перерисовываем бейдж

  renderBadge();

  // === ДОБАВЛЕНО: Обновляем видимость настроек ===
  toggleBadgeSettingsVisibility();
  // === КОНЕЦ ДОБАВЛЕНИЯ ===
}

/**
 * Обрабатывает загрузку файла логотипа (без изменений)
 */
function handleLogoUpload(event) {
  const file = event.target.files[0];
  const warningElement = els.badge.logoWarning; // Сохраним ссылку

  if (warningElement) warningElement.textContent = ""; // Очистим предупреждение сразу

  if (!file) return; // Проверка типа файла (добавлено)

  if (
    !["image/png", "image/jpeg", "image/gif", "image/webp"].includes(file.type)
  ) {
    // Расширим типы
    if (warningElement)
      warningElement.textContent = "Неверный тип файла (PNG, JPG, GIF, WEBP).";
    return;
  } // Проверка размера файла (добавлено, например, < 1MB)

  if (file.size > 1 * 1024 * 1024) {
    if (warningElement)
      warningElement.textContent = "Файл слишком большой (макс. 1MB).";
    return;
  }

  const reader = new FileReader();

  reader.onload = (e) => {
    const dataUrl = e.target.result; // Валидация

    const img = new Image();
    img.onload = () => {
      let warning = "";
      if (img.naturalWidth < 48 || img.naturalHeight < 48) {
        // Уменьшил требование до 48px
        warning = "Лого маловато (реком. 48x48+). ";
      }
      if (img.naturalWidth !== img.naturalHeight) {
        warning += "Лого не квадратное.";
      }
      if (warningElement) warningElement.textContent = warning; // Сохраняем DataURL и обновляем

      badgeSettings.logoDataUrl = dataUrl;
      handleBadgeSettingChange();
    };
    img.onerror = () => {
      // Добавим обработку ошибки загрузки
      if (warningElement)
        warningElement.textContent = "Не удалось загрузить изображение.";
      badgeSettings.logoDataUrl = ""; // Сбрасываем URL
      handleBadgeSettingChange(); // Обновляем, чтобы убрать старое лого
    };
    img.src = dataUrl;
  };

  reader.onerror = () => {
    // Добавим обработку ошибки чтения файла
    if (warningElement)
      warningElement.textContent = "Не удалось прочитать файл.";
  };

  reader.readAsDataURL(file);
}

/**
 * Загружает настройки бейджа из localStorage при запуске
 */
function loadBadgeSettings() {
  const b = els.badge;
  const defaults = {
    show: false,
    badgePosition: "pos-bottom-left",
    logoType: "url",
    logoUrl: "",
    logoDataUrl: "",
    colorPrimary: "#0052CC",
    colorSecondary: "#00B8D9",
    name: "Иванов Сергей",
    company: "ООО «Рога и Копыта»",
    jobPosition: "",
    department: "",
    location: "",
    telegram: "",
    email: "",
    slogan: "",
    showName: true,
    showCompany: true,
    showPosition: false,
    showDepartment: false,
    showLocation: false,
    showTelegram: false,
    showEmail: false,
    showSlogan: false,
  };

  try {
    const storedSettings = localStorage.getItem("badgeSettings");
    if (storedSettings && typeof storedSettings === "string") {
      const parsed = JSON.parse(storedSettings);
      if (
        !parsed.badgePosition ||
        typeof parsed.badgePosition !== "string" ||
        !parsed.badgePosition.startsWith("pos-")
      ) {
        console.warn(
          "Загруженная позиция некорректна, используется дефолтная.",
        );
        parsed.badgePosition = defaults.badgePosition;
      }
      badgeSettings = { ...defaults, ...parsed };
    } else {
      badgeSettings = { ...defaults };
    }
  } catch (e) {
    console.error("Ошибка загрузки настроек бейджа из localStorage:", e);
    badgeSettings = { ...defaults };
  }

  if (b.toggleShow) b.toggleShow.checked = badgeSettings.show;

  const currentPositionRadio = document.querySelector(
    `input[name="badge-position"][value="${badgeSettings.badgePosition}"]`,
  );
  if (currentPositionRadio) {
    currentPositionRadio.checked = true;
  } else {
    console.warn(
      `Не найдена радио-кнопка для позиции '${badgeSettings.badgePosition}', устанавливается дефолтная.`,
    );
    const defaultPositionRadio = document.querySelector(
      `input[name="badge-position"][value="${defaults.badgePosition}"]`,
    );
    if (defaultPositionRadio) defaultPositionRadio.checked = true;
  }

  const currentLogoTypeRadio = document.querySelector(
    `input[name="badge-logo-type"][value="${badgeSettings.logoType}"]`,
  );
  if (currentLogoTypeRadio) {
    currentLogoTypeRadio.checked = true;
  } else {
    const defaultLogoTypeRadio = document.querySelector(
      `input[name="badge-logo-type"][value="${defaults.logoType}"]`,
    );
    if (defaultLogoTypeRadio) defaultLogoTypeRadio.checked = true;
  }

  if (b.logoUrl) b.logoUrl.value = badgeSettings.logoUrl;

  // Обновляем ТЕКСТОВОЕ ПОЛЕ для Фона
  if (b.colorPrimary) b.colorPrimary.value = badgeSettings.colorPrimary;
  // Обновляем ПАЛИТРУ для Фона
  if (b.pickerColorPrimary)
    b.pickerColorPrimary.value = badgeSettings.colorPrimary;

  // Обновляем ТЕКСТОВОЕ ПОЛЕ для Акцента
  if (b.colorSecondary) b.colorSecondary.value = badgeSettings.colorSecondary;
  // Обновляем ПАЛИТРУ для Акцента
  if (b.pickerColorSecondary)
    b.pickerColorSecondary.value = badgeSettings.colorSecondary;

  if (b.fieldName) b.fieldName.value = badgeSettings.name;
  if (b.fieldCompany) b.fieldCompany.value = badgeSettings.company;
  if (b.fieldPosition) b.fieldPosition.value = badgeSettings.jobPosition;
  if (b.fieldDepartment) b.fieldDepartment.value = badgeSettings.department;
  if (b.fieldLocation) b.fieldLocation.value = badgeSettings.location;
  if (b.fieldTelegram) b.fieldTelegram.value = badgeSettings.telegram;
  if (b.fieldEmail) b.fieldEmail.value = badgeSettings.email;
  if (b.fieldSlogan) b.fieldSlogan.value = badgeSettings.slogan;

  if (b.toggleName) b.toggleName.checked = badgeSettings.showName;
  if (b.toggleCompany) b.toggleCompany.checked = badgeSettings.showCompany;
  if (b.togglePosition) b.togglePosition.checked = badgeSettings.showPosition;
  if (b.toggleDepartment)
    b.toggleDepartment.checked = badgeSettings.showDepartment;
  if (b.toggleLocation) b.toggleLocation.checked = badgeSettings.showLocation;
  if (b.toggleTelegram) b.toggleTelegram.checked = badgeSettings.showTelegram;
  if (b.toggleEmail) b.toggleEmail.checked = badgeSettings.showEmail;
  if (b.toggleSlogan) b.toggleSlogan.checked = badgeSettings.showSlogan;

  if (b.logoUrl && b.logoUpload) {
    if (badgeSettings.logoType === "url") {
      b.logoUrl.classList.remove("hidden");
      b.logoUpload.classList.add("hidden");
    } else {
      b.logoUrl.classList.add("hidden");
      b.logoUpload.classList.remove("hidden");
    } // <--- ВОТ ЗДЕСЬ БЫЛА ОПЕЧАТКА "D"
  }

  if (b.logoWarning) b.logoWarning.textContent = "";

  renderBadge();

  toggleBadgeSettingsVisibility();
}

// ==== Initialization and Event Listeners ====
async function init() {
  // Ждем загрузки DOM перед поиском элементов
  if (document.readyState === "loading") {
    await new Promise((resolve) =>
      document.addEventListener("DOMContentLoaded", resolve),
    );
  } // === ИЗМЕНЕНИЕ: Сначала загружаем настройки бейджа, потом вешаем обработчики ===

  els.badge.settingsGroup = document.getElementById("badge-settings-group");
  loadBadgeSettings();

  await listCameras();
  if (navigator.mediaDevices?.addEventListener) {
    // Проверка на addEventListener
    navigator.mediaDevices.addEventListener("devicechange", listCameras);
  } // --- Обработчик: Переключение камеры ---

  if (els.toggleCamBtn) {
    els.toggleCamBtn.addEventListener("click", () => {
      if (running) stopCamera();
      else startCamera();
    });
  } // --- Обработчики модального окна ---

  if (els.settingsBtn) els.settingsBtn.addEventListener("click", openSettings);
  if (els.closeSettingsBtn)
    els.closeSettingsBtn.addEventListener("click", closeSettings);
  if (els.settingsModal) {
    els.settingsModal.addEventListener("click", (e) => {
      if (e.target === els.settingsModal) closeSettings();
    });
  }

  window.addEventListener("pagehide", stopCamera);
  window.addEventListener("beforeunload", stopCamera); // --- Обработчик: Переключение табов ---

  const tabLinks = document.querySelectorAll(".tab-link");
  const tabPanes = document.querySelectorAll(".tab-pane");
  if (tabLinks.length > 0 && tabPanes.length > 0) {
    // Проверка
    tabLinks.forEach((link) => {
      link.addEventListener("click", () => {
        const tabId = link.getAttribute("data-tab");
        if (!tabId) return; // Проверка
        tabLinks.forEach((btn) => btn.classList.remove("active"));
        link.classList.add("active");
        tabPanes.forEach((pane) => pane.classList.remove("active"));
        const activePane = document.getElementById(tabId);
        if (activePane) activePane.classList.add("active");
      });
    });
  } else {
    console.warn("Не найдены элементы табов для настроек.");
  } // --- Обработчик: Смена камеры "на лету" ---

  function handleVideoSettingsChange() {
    // Переименовал для ясности
    if (running) {
      console.log("Настройки видео изменились, перезапускаем камеру...");
      stopCamera();
      setTimeout(startCamera, 50); // Небольшая задержка
    }
  }
  if (els.camSelect)
    els.camSelect.addEventListener("change", handleVideoSettingsChange); // === УСТАНОВКА ОБРАБОТЧИКОВ ДЛЯ БЕЙДЖА ===
  // Массив текстовых инпутов, чекбоксов и колор-пикеров
  // === ИЗМЕНЕНИЕ 1: Я УБРАЛ ОТСЮДА colorPrimary и colorSecondary ===
  const directUpdateControls = [
    els.badge.toggleShow,
    els.badge.logoUrl,
    // els.badge.colorPrimary,   <-- УБРАНО
    // els.badge.colorSecondary, <-- УБРАНО
    els.badge.fieldName,
    els.badge.fieldCompany,
    els.badge.fieldPosition,
    els.badge.fieldDepartment,
    els.badge.fieldLocation,
    els.badge.fieldTelegram,
    els.badge.fieldEmail,
    els.badge.fieldSlogan,
    els.badge.toggleName,
    els.badge.toggleCompany,
    els.badge.togglePosition,
    els.badge.toggleDepartment,
    els.badge.toggleLocation,
    els.badge.toggleTelegram,
    els.badge.toggleEmail,
    els.badge.toggleSlogan,
  ];

  directUpdateControls.forEach((el) => {
    if (el) {
      // Проверка на null
      // Используем 'input' для текстовых полей и 'change' для остальных
      const eventType =
        el.type === "text" || el.tagName === "TEXTAREA" ? "input" : "change";
      el.addEventListener(eventType, handleBadgeSettingChange);
    } else {
      // console.warn("Элемент управления бейджем не найден, пропуск добавления обработчика.");
    }
  }); // Обработчики на радиокнопки позиций

  if (els.badge.positionRadios) {
    els.badge.positionRadios.forEach((radio) =>
      radio.addEventListener("change", handleBadgeSettingChange),
    );
  } // Обработчики на радиокнопки типа лого

  if (els.badge.logoTypeRadios) {
    els.badge.logoTypeRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        // Показываем/скрываем нужный инпут (с проверками)
        if (els.badge.logoUrl && els.badge.logoUpload) {
          if (radio.value === "url") {
            els.badge.logoUrl.classList.remove("hidden");
            els.badge.logoUpload.classList.add("hidden");
          } else {
            els.badge.logoUrl.classList.add("hidden");
            els.badge.logoUpload.classList.remove("hidden");
          }
        }
        if (els.badge.logoWarning) els.badge.logoWarning.textContent = ""; // Очищаем предупреждение
        // Вызываем основной обработчик, чтобы сохранить новое значение logoType
        handleBadgeSettingChange();
      });
    });
  } // Обработчик на загрузку файла

  if (els.badge.logoUpload) {
    els.badge.logoUpload.addEventListener("change", handleLogoUpload);
  }

  // === ИЗМЕНЕНИЕ 2: ВОТ ЭТОТ КОД ДОБАВЛЕН ===
  // === ОН СВЯЗЫВАЕТ ПАЛИТРУ И ПОЛЕ ===

  /**
   * Вспомогательная функция для синхронизации текстового поля и палитры
   * @param {HTMLInputElement} textField - Текстовое поле (input type="text")
   * @param {HTMLInputElement} colorPicker - Палитра (input type="color")
   */
  function syncColorInputs(textField, colorPicker) {
    if (!textField || !colorPicker) return; // Проверка

    // 1. При вводе в ТЕКСТОВОЕ ПОЛЕ
    textField.addEventListener("input", (e) => {
      const newColor = e.target.value;
      // Пытаемся применить цвет к палитре.
      // Браузер сам проверит, валидный ли это HEX.
      try {
        colorPicker.value = newColor;
      } catch (err) {
        // Игнорируем ошибку, если введен невалидный HEX
      }

      // Вызываем главный обработчик, чтобы бейдж обновился в реальном времени
      handleBadgeSettingChange();
    });

    // 2. При выборе в ПАЛИТРЕ
    colorPicker.addEventListener("input", (e) => {
      const newColor = e.target.value.toUpperCase();
      // Обновляем текстовое поле
      textField.value = newColor;

      // Вызываем главный обработчик для обновления бейджа
      handleBadgeSettingChange();
    });
  }

  // Синхронизируем обе пары
  syncColorInputs(els.badge.colorPrimary, els.badge.pickerColorPrimary);
  syncColorInputs(els.badge.colorSecondary, els.badge.pickerColorSecondary);

  // === КОНЕЦ ИЗМЕНЕНИЯ 2 ===
}

// Запуск
init().catch((e) => {
  // Добавим обработку ошибок инициализации
  console.error("Ошибка при инициализации приложения:", e);
  setStatus("Ошибка инициализации", "warn");
});
