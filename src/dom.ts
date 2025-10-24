import type { Els } from "./types";

function byId<T extends HTMLElement>(id: string) {
  const el = document.getElementById(id) as T | null;
  if (!el) throw new Error(`Не найден элемент #${id}`);
  return el;
}

export const els: Els = {
  video: byId<HTMLVideoElement>("video"),
  camSelect: byId<HTMLSelectElement>("cameraSelect"),
  status: byId<HTMLElement>("status"),
  fpsNow: byId<HTMLElement>("fps"),
  fpsAvg: byId<HTMLElement>("fpsAvg"),
  cpuNow: byId<HTMLElement>("cpuNow"),
  cpuAvg: byId<HTMLElement>("cpuAvg"),
  gpuNow: byId<HTMLElement>("gpuNow"),
  gpuAvg: byId<HTMLElement>("gpuAvg"),
  settingsBtn: byId<HTMLButtonElement>("settingsBtn"),
  settingsModal: byId<HTMLElement>("settingsModal"),
  closeSettingsBtn: byId<HTMLElement>("closeSettingsBtn"),
  toggleCamBtn: byId<HTMLButtonElement>("toggleCamBtn"),

  badge: {
    overlay: byId<HTMLElement>("smartBadgeOverlay"),
    badge: document.querySelector(".smart-badge") as HTMLElement,
    logoContainer: byId<HTMLElement>("badge-logo-container"),
    logoImg: byId<HTMLImageElement>("badge-logo-img"),

    mainInfo: document.querySelector(".badge-main") as HTMLElement,
    nameText: byId<HTMLElement>("badge-name-text"),
    companyText: byId<HTMLElement>("badge-company-text"),

    details: document.querySelector(".badge-details") as HTMLElement,
    itemPosition: byId<HTMLElement>("badge-item-position"),
    itemDepartment: byId<HTMLElement>("badge-item-department"),
    itemLocation: byId<HTMLElement>("badge-item-location"),
    itemTelegram: byId<HTMLElement>("badge-item-telegram"),
    itemEmail: byId<HTMLElement>("badge-item-email"),
    itemSlogan: byId<HTMLElement>("badge-item-slogan"),

    positionText: byId<HTMLElement>("badge-position-text"),
    departmentText: byId<HTMLElement>("badge-department-text"),
    locationText: byId<HTMLElement>("badge-location-text"),
    telegramLink: byId<HTMLAnchorElement>("badge-telegram-link"),
    emailLink: byId<HTMLAnchorElement>("badge-email-link"),
    sloganText: byId<HTMLElement>("badge-slogan-text"),

    toggleShow: byId<HTMLInputElement>("badge-toggle-show"),
    settingsGroup: document.getElementById("badge-settings-group"),

    positionRadios: document.querySelectorAll('input[name="badge-position"]') as NodeListOf<HTMLInputElement>,
    logoTypeRadios: document.querySelectorAll('input[name="badge-logo-type"]') as NodeListOf<HTMLInputElement>,

    logoUrl: byId<HTMLInputElement>("badge-logo-url"),
    logoUpload: byId<HTMLInputElement>("badge-logo-upload"),
    logoWarning: byId<HTMLElement>("badge-logo-warning"),

    colorPrimary: byId<HTMLInputElement>("badge-field-color-primary"),
    colorSecondary: byId<HTMLInputElement>("badge-field-color-secondary"),
    pickerColorPrimary: byId<HTMLInputElement>("badge-picker-color-primary"),
    pickerColorSecondary: byId<HTMLInputElement>("badge-picker-color-secondary"),

    fieldName: byId<HTMLInputElement>("badge-field-name"),
    fieldCompany: byId<HTMLInputElement>("badge-field-company"),
    fieldPosition: byId<HTMLInputElement>("badge-field-position"),
    fieldDepartment: byId<HTMLInputElement>("badge-field-department"),
    fieldLocation: byId<HTMLInputElement>("badge-field-location"),
    fieldTelegram: byId<HTMLInputElement>("badge-field-telegram"),
    fieldEmail: byId<HTMLInputElement>("badge-field-email"),
    fieldSlogan: byId<HTMLInputElement>("badge-field-slogan"),

    toggleName: byId<HTMLInputElement>("badge-toggle-name"),
    toggleCompany: byId<HTMLInputElement>("badge-toggle-company"),
    togglePosition: byId<HTMLInputElement>("badge-toggle-position"),
    toggleDepartment: byId<HTMLInputElement>("badge-toggle-department"),
    toggleLocation: byId<HTMLInputElement>("badge-toggle-location"),
    toggleTelegram: byId<HTMLInputElement>("badge-toggle-telegram"),
    toggleEmail: byId<HTMLInputElement>("badge-toggle-email"),
    toggleSlogan: byId<HTMLInputElement>("badge-toggle-slogan"),
  },
};