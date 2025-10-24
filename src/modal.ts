import { els } from "./dom";

export function openSettings() { els.settingsModal.style.display = "flex"; }
export function closeSettings() { els.settingsModal.style.display = "none"; }

export function bindModal(): void {
  els.settingsBtn.addEventListener("click", openSettings);
  els.closeSettingsBtn.addEventListener("click", closeSettings);
  els.settingsModal.addEventListener("click", (e) => { if (e.target === els.settingsModal) closeSettings(); });

  const tabLinks = document.querySelectorAll(".tab-link");
  const tabPanes = document.querySelectorAll(".tab-pane");
  tabLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const tabId = link.getAttribute("data-tab");
      if (!tabId) return;
      tabLinks.forEach((btn) => btn.classList.remove("active"));
      link.classList.add("active");
      tabPanes.forEach((pane) => pane.classList.remove("active"));
      const activePane = document.getElementById(tabId);
      activePane?.classList.add("active");
    });
  });
}

export function setStatus(text: string, cls = ""): void {
  els.status.className = "";
  if (cls) els.status.classList.add(cls);
  els.status.textContent = `Статус: ${text}`;
}

export function resetMetrics(): void {
  const set = (id: string, text: string) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };
  set("fps", "FPS: - fps");
  set("fpsAvg", "FPSAvg: - fps");
  set("cpuNow", "CPU: -%");
  set("cpuAvg", "CPUAvg: -%");
  set("gpuNow", "GPU: -%");
  set("gpuAvg", "GPUAvg: -%");
}