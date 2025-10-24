export interface BadgeSettings {
  show: boolean;
  badgePosition: "pos-top-left" | "pos-top-center" | "pos-top-right" | "pos-bottom-left" | "pos-bottom-center" | "pos-bottom-right";
  logoType: "url" | "upload";
  logoUrl: string;
  logoDataUrl: string;
  colorPrimary: string;
  colorSecondary: string;
  name: string;
  company: string;
  jobPosition: string;
  department: string;
  location: string;
  telegram: string;
  email: string;
  slogan: string;
  showName: boolean;
  showCompany: boolean;
  showPosition: boolean;
  showDepartment: boolean;
  showLocation: boolean;
  showTelegram: boolean;
  showEmail: boolean;
  showSlogan: boolean;
}

export interface Els {
  video: HTMLVideoElement;
  camSelect: HTMLSelectElement;
  status: HTMLElement;
  fpsNow: HTMLElement;
  fpsAvg: HTMLElement;
  cpuNow: HTMLElement;
  cpuAvg: HTMLElement;
  gpuNow: HTMLElement;
  gpuAvg: HTMLElement;
  settingsBtn: HTMLButtonElement;
  settingsModal: HTMLElement;
  closeSettingsBtn: HTMLElement;
  toggleCamBtn: HTMLButtonElement;
  badge: {
    overlay: HTMLElement;
    badge: HTMLElement;
    logoContainer: HTMLElement;
    logoImg: HTMLImageElement;

    mainInfo: HTMLElement;
    nameText: HTMLElement;
    companyText: HTMLElement;

    details: HTMLElement;
    itemPosition: HTMLElement;
    itemDepartment: HTMLElement;
    itemLocation: HTMLElement;
    itemTelegram: HTMLElement;
    itemEmail: HTMLElement;
    itemSlogan: HTMLElement;

    positionText: HTMLElement;
    departmentText: HTMLElement;
    locationText: HTMLElement;
    telegramLink: HTMLAnchorElement;
    emailLink: HTMLAnchorElement;
    sloganText: HTMLElement;

    toggleShow: HTMLInputElement;
    settingsGroup: HTMLElement | null;

    positionRadios: NodeListOf<HTMLInputElement>;
    logoTypeRadios: NodeListOf<HTMLInputElement>;

    logoUrl: HTMLInputElement;
    logoUpload: HTMLInputElement;
    logoWarning: HTMLElement;

    colorPrimary: HTMLInputElement;          // text input
    colorSecondary: HTMLInputElement;        // text input
    pickerColorPrimary: HTMLInputElement;    // color input
    pickerColorSecondary: HTMLInputElement;  // color input

    fieldName: HTMLInputElement;
    fieldCompany: HTMLInputElement;
    fieldPosition: HTMLInputElement;
    fieldDepartment: HTMLInputElement;
    fieldLocation: HTMLInputElement;
    fieldTelegram: HTMLInputElement;
    fieldEmail: HTMLInputElement;
    fieldSlogan: HTMLInputElement;

    toggleName: HTMLInputElement;
    toggleCompany: HTMLInputElement;
    togglePosition: HTMLInputElement;
    toggleDepartment: HTMLInputElement;
    toggleLocation: HTMLInputElement;
    toggleTelegram: HTMLInputElement;
    toggleEmail: HTMLInputElement;
    toggleSlogan: HTMLInputElement;
  };
}

export type Nullable<T> = T | null;

// Дополняем типизацию для requestVideoFrameCallback (если не объявлено в lib.dom)
declare global {
  type VideoFrameRequestCallback = (now: number, metadata: { mediaTime: number; presentedFrames: number }) => void;
  interface HTMLVideoElement {
    requestVideoFrameCallback?: (callback: VideoFrameRequestCallback) => number;
  }
}