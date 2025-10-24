import { els } from "./dom";
import { BadgeSettings } from "./types";
import { setBadgeSettings, badgeSettings as bsRef } from "./state";

export function toggleBadgeSettingsVisibility(): void {
  if (!els.badge.settingsGroup) return;
  if (els.badge.toggleShow.checked) {
    els.badge.settingsGroup.classList.remove("hidden");
  } else {
    els.badge.settingsGroup.classList.add("hidden");
  }
}

export function renderBadge(): void {
  const b = els.badge;
  const s = bsRef;

  b.overlay.style.display = s.show ? "block" : "none";
  if (!s.show) return;

  b.overlay.className = "badge-overlay-container";
  b.overlay.classList.add(s.badgePosition || "pos-bottom-left");

  b.badge.style.backgroundColor = s.colorPrimary;
  (b.badge.style as any).outlineColor = s.colorSecondary;

  const logoSrc = s.logoType === "upload" ? s.logoDataUrl : s.logoUrl;
  if (logoSrc) {
    b.logoImg.src = logoSrc;
    b.logoContainer.style.display = "block";
  } else {
    b.logoContainer.style.display = "none";
    b.logoImg.src = "";
  }

  b.nameText.textContent = s.name;
  (b.nameText.style as any).display = s.showName && s.name ? "block" : "none";

  b.companyText.textContent = s.company;
  (b.companyText.style as any).display = s.showCompany && s.company ? "block" : "none";

  (b.mainInfo.style as any).display = (s.showName && s.name) || (s.showCompany && s.company) ? "block" : "none";

  b.positionText.textContent = s.jobPosition;
  (b.itemPosition.style as any).display = s.showPosition && s.jobPosition ? "flex" : "none";

  b.departmentText.textContent = s.department;
  (b.itemDepartment.style as any).display = s.showDepartment && s.department ? "flex" : "none";

  b.locationText.textContent = s.location;
  (b.itemLocation.style as any).display = s.showLocation && s.location ? "flex" : "none";

  b.sloganText.textContent = s.slogan;
  (b.itemSlogan.style as any).display = s.showSlogan && s.slogan ? "flex" : "none";

  if (s.showTelegram && s.telegram) {
    const username = s.telegram.replace(/^@/, "");
    b.telegramLink.textContent = s.telegram;
    b.telegramLink.href = `https://t.me/${username}`;
    (b.itemTelegram.style as any).display = "flex";
    b.telegramLink.style.color = s.colorSecondary;
    const icon = b.itemTelegram.querySelector("i") as HTMLElement | null;
    if (icon) {
      icon.style.color = s.colorSecondary;
      icon.style.opacity = "1";
    }
  } else {
    (b.itemTelegram.style as any).display = "none";
  }

  if (s.showEmail && s.email) {
    b.emailLink.textContent = s.email;
    b.emailLink.href = `mailto:${s.email}`;
    (b.itemEmail.style as any).display = "flex";
  } else {
    (b.itemEmail.style as any).display = "none";
  }

  const hasDetails = (
    (s.showPosition && s.jobPosition) ||
    (s.showDepartment && s.department) ||
    (s.showLocation && s.location) ||
    (s.showTelegram && s.telegram) ||
    (s.showEmail && s.email) ||
    (s.showSlogan && s.slogan)
  );
  (b.details.style as any).display = hasDetails ? "block" : "none";
}

export function handleBadgeSettingChange(): void {
  const currentPositionRadio = document.querySelector('input[name="badge-position"]:checked') as HTMLInputElement | null;
  const currentLogoTypeRadio = document.querySelector('input[name="badge-logo-type"]:checked') as HTMLInputElement | null;
  if (!currentPositionRadio || !currentLogoTypeRadio) return;

  const next: BadgeSettings = {
    show: els.badge.toggleShow.checked,
    badgePosition: currentPositionRadio.value as BadgeSettings["badgePosition"],
    logoType: currentLogoTypeRadio.value as BadgeSettings["logoType"],
    logoUrl: els.badge.logoUrl.value || "",
    logoDataUrl: (bsRef.logoDataUrl ?? ""),
    colorPrimary: els.badge.colorPrimary.value || "#0052CC",
    colorSecondary: els.badge.colorSecondary.value || "#00B8D9",
    name: els.badge.fieldName.value || "",
    company: els.badge.fieldCompany.value || "",
    jobPosition: els.badge.fieldPosition.value || "",
    department: els.badge.fieldDepartment.value || "",
    location: els.badge.fieldLocation.value || "",
    telegram: els.badge.fieldTelegram.value || "",
    email: els.badge.fieldEmail.value || "",
    slogan: els.badge.fieldSlogan.value || "",
    showName: els.badge.toggleName.checked,
    showCompany: els.badge.toggleCompany.checked,
    showPosition: els.badge.togglePosition.checked,
    showDepartment: els.badge.toggleDepartment.checked,
    showLocation: els.badge.toggleLocation.checked,
    showTelegram: els.badge.toggleTelegram.checked,
    showEmail: els.badge.toggleEmail.checked,
    showSlogan: els.badge.toggleSlogan.checked,
  };

  setBadgeSettings(next);
  try { localStorage.setItem("badgeSettings", JSON.stringify(next)); } catch {}
  renderBadge();
  toggleBadgeSettingsVisibility();
}

export function handleLogoUpload(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  const warn = els.badge.logoWarning;
  warn.textContent = "";
  if (!file) return;

  if (!["image/png", "image/jpeg", "image/gif", "image/webp"].includes(file.type)) {
    warn.textContent = "Неверный тип файла (PNG, JPG, GIF, WEBP).";
    return;
  }
  if (file.size > 1024 * 1024) {
    warn.textContent = "Файл слишком большой (макс. 1MB).";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = String(reader.result || "");
    const img = new Image();
    img.onload = () => {
      let msg = "";
      if (img.naturalWidth < 48 || img.naturalHeight < 48) msg = "Лого маловато (реком. 48x48+). ";
      if (img.naturalWidth !== img.naturalHeight) msg += "Лого не квадратное.";
      warn.textContent = msg;
      setBadgeSettings({ ...bsRef, logoDataUrl: dataUrl });
      handleBadgeSettingChange();
    };
    img.onerror = () => {
      warn.textContent = "Не удалось загрузить изображение.";
      setBadgeSettings({ ...bsRef, logoDataUrl: "" });
      handleBadgeSettingChange();
    };
    img.src = dataUrl;
  };
  reader.onerror = () => { warn.textContent = "Не удалось прочитать файл."; };
  reader.readAsDataURL(file);
}

export function loadBadgeSettings(): void {
  const defaults: BadgeSettings = {
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
    const raw = localStorage.getItem("badgeSettings");
    const parsed = raw ? (JSON.parse(raw) as Partial<BadgeSettings>) : null;
    const merged: BadgeSettings = { ...defaults, ...(parsed || {}) };
    if (!merged.badgePosition?.startsWith("pos-")) merged.badgePosition = defaults.badgePosition;
    setBadgeSettings(merged);
  } catch {
    setBadgeSettings({ ...defaults });
  }

  els.badge.toggleShow.checked = bsRef.show;

  const posInput = document.querySelector(`input[name="badge-position"][value="${bsRef.badgePosition}"]`) as HTMLInputElement | null;
  if (posInput) posInput.checked = true;

  const logoTypeInput = document.querySelector(`input[name="badge-logo-type"][value="${bsRef.logoType}"]`) as HTMLInputElement | null;
  if (logoTypeInput) logoTypeInput.checked = true;

  els.badge.logoUrl.value = bsRef.logoUrl;

  els.badge.colorPrimary.value = bsRef.colorPrimary;
  els.badge.pickerColorPrimary.value = bsRef.colorPrimary;
  els.badge.colorSecondary.value = bsRef.colorSecondary;
  els.badge.pickerColorSecondary.value = bsRef.colorSecondary;

  els.badge.fieldName.value = bsRef.name;
  els.badge.fieldCompany.value = bsRef.company;
  els.badge.fieldPosition.value = bsRef.jobPosition;
  els.badge.fieldDepartment.value = bsRef.department;
  els.badge.fieldLocation.value = bsRef.location;
  els.badge.fieldTelegram.value = bsRef.telegram;
  els.badge.fieldEmail.value = bsRef.email;
  els.badge.fieldSlogan.value = bsRef.slogan;

  els.badge.toggleName.checked = bsRef.showName;
  els.badge.toggleCompany.checked = bsRef.showCompany;
  els.badge.togglePosition.checked = bsRef.showPosition;
  els.badge.toggleDepartment.checked = bsRef.showDepartment;
  els.badge.toggleLocation.checked = bsRef.showLocation;
  els.badge.toggleTelegram.checked = bsRef.showTelegram;
  els.badge.toggleEmail.checked = bsRef.showEmail;
  els.badge.toggleSlogan.checked = bsRef.showSlogan;

  // Переключение URL/Upload
  if (bsRef.logoType === "url") {
    els.badge.logoUrl.classList.remove("hidden");
    els.badge.logoUpload.classList.add("hidden");
  } else {
    els.badge.logoUrl.classList.add("hidden");
    els.badge.logoUpload.classList.remove("hidden");
  }

  els.badge.logoWarning.textContent = "";
  renderBadge();
  toggleBadgeSettingsVisibility();
}

export function bindBadgeControls(): void {
  // текстовые поля + чекбоксы
  const directUpdateControls: (HTMLElement | null)[] = [
    els.badge.toggleShow,
    els.badge.logoUrl,
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
    if (!el) return;
    const inputEl = el as HTMLInputElement;
    const eventType = inputEl.type === "text" ? "input" : "change";
    inputEl.addEventListener(eventType, handleBadgeSettingChange);
  });

  els.badge.positionRadios.forEach((r) => r.addEventListener("change", handleBadgeSettingChange));

  els.badge.logoTypeRadios.forEach((r) => {
    r.addEventListener("change", () => {
      if (r.value === "url") {
        els.badge.logoUrl.classList.remove("hidden");
        els.badge.logoUpload.classList.add("hidden");
      } else {
        els.badge.logoUrl.classList.add("hidden");
        els.badge.logoUpload.classList.remove("hidden");
      }
      els.badge.logoWarning.textContent = "";
      handleBadgeSettingChange();
    });
  });

  els.badge.logoUpload.addEventListener("change", handleLogoUpload);

  // Синхронизация текстового HEX и color input
  function syncColorInputs(textField: HTMLInputElement, colorPicker: HTMLInputElement) {
    textField.addEventListener("input", (e) => {
      const val = (e.target as HTMLInputElement).value;
      try { colorPicker.value = val; } catch {}
      handleBadgeSettingChange();
    });
    colorPicker.addEventListener("input", (e) => {
      const val = (e.target as HTMLInputElement).value.toUpperCase();
      textField.value = val;
      handleBadgeSettingChange();
    });
  }

  syncColorInputs(els.badge.colorPrimary, els.badge.pickerColorPrimary);
  syncColorInputs(els.badge.colorSecondary, els.badge.pickerColorSecondary);
}