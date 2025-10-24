export type OverlayHandle = {
  draw(mask: Float32Array, mw: number, mh: number): void;
  resizeToVideo(): void;
};

export function attachMaskOverlay(container: HTMLElement, video: HTMLVideoElement, color = "rgba(0,0,0,0.6)"): OverlayHandle {
  const canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.inset = "0";
  canvas.style.pointerEvents = "none";
  canvas.style.zIndex = "6"; // поверх метрик, но под кнопками
  container.appendChild(canvas);

  const ctx = canvas.getContext("2d")!;
  const tmp = document.createElement("canvas");
  const tctx = tmp.getContext("2d")!;

  function resizeToVideo() {
    const w = (video as any).clientWidth || container.clientWidth;
    const h = (video as any).clientHeight || container.clientHeight;
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function draw(mask: Float32Array, mw: number, mh: number) {
    if (!mw || !mh) return;
    // 1) Рисуем mask->ImageData, ИНВЕРТИРУЕМ (1 - mask)
    tmp.width = mw; tmp.height = mh;
    const img = tctx.createImageData(mw, mh);
    const alpha = 180; // плотность затемнения фона после инверсии
    for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
      const inv = 1 - (mask[i] > 0.5 ? 1 : 0); // бинарная инверсия
      // Подложка чёрная: RGBA = 0,0,0, inv*alpha
      img.data[j] = 0;
      img.data[j + 1] = 0;
      img.data[j + 2] = 0;
      img.data[j + 3] = inv * alpha;
    }
    tctx.putImageData(img, 0, 0);

    // 2) Апскейл до размера видео и отрисовка поверх
    resizeToVideo();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Сглаживание можно выключить для "жёсткой" маски:
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  }

  // Инициализация размеров
  resizeToVideo();
  return { draw, resizeToVideo };
}
