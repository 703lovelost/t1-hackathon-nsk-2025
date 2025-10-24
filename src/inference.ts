import * as tf from "@tensorflow/tfjs";
import type { Tensor, Tensor3D, Tensor4D, GraphModel } from "@tensorflow/tfjs";
import { els } from "./dom";

type MaybeTensor = Tensor | Tensor[];

/**
 * Движок сегментации: загрузка, warmup и инференс маски с наложением RGBA-оверлея.
 * Главное:
 *  - никаких await/Promise внутри tf.tidy()
 *  - для статического графа используем model.execute()
 *  - fallback на model.executeAsync() с явным dispose выходов
 */
class SegmentationEngine {
  private model: GraphModel | null = null;
  private inputH = 0;
  private inputW = 0;
  private loaded = false;

  private overlayCanvas: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;
  private busy = false;
  private thresh = 0.5;
  private alpha = 0.45;

  constructor() {
    // создаём канвас-оверлей поверх <video>
    this.overlayCanvas = document.createElement("canvas");
    const parent = els.video.parentElement as HTMLElement; // .main-video-pane
    this.overlayCanvas.style.position = "absolute";
    this.overlayCanvas.style.inset = "0";
    this.overlayCanvas.style.pointerEvents = "none";
    this.overlayCanvas.style.zIndex = "6";
    parent.appendChild(this.overlayCanvas);

    const ctx = this.overlayCanvas.getContext("2d");
    if (!ctx) throw new Error("Не удалось создать 2D контекст для маски");
    this.overlayCtx = ctx;
  }

  get isReady() { return this.loaded; }
  setThreshold(t: number) { this.thresh = Math.min(1, Math.max(0, t)); }
  setAlpha(a: number)     { this.alpha = Math.min(1, Math.max(0, a)); }

  /**
   * Грузим граф-модель. Требует предварительной инициализации бэкенда (см. main.ts).
   */
  async load(url = "/models/yolo11m-seg_web_model_tfjs/model.json") {
    if (this.loaded) return;

    // гарантируем, что выбранный бэкенд готов
    await tf.ready();

    this.model = await tf.loadGraphModel(url);
    const inShape = this.model.inputs?.[0]?.shape ?? [1, 640, 640, 3];
    this.inputH = Number(inShape?.[1] ?? 640);
    this.inputW = Number(inShape?.[2] ?? 640);

    // warmup: СИНХРОННО через execute (без async/await внутри tidy)
    const dummy = tf.tidy(() =>
      tf.zeros([1, this.inputH, this.inputW, 3], "float32")
    );
    try {
      this.model.execute(dummy as Tensor4D); // если граф статический, этого достаточно
    } catch {
      // если модель требует async-выполнения (редкие случаи) — один раз выполним вне tidy
      await this.model.executeAsync(dummy as Tensor4D);
    } finally {
      dummy.dispose();
    }

    this.loaded = true;
    this.resizeOverlayToVideo();
  }

  private resizeOverlayToVideo() {
    const w = els.video.clientWidth;
    const h = els.video.clientHeight;
    if (w <= 0 || h <= 0) return;
    if (this.overlayCanvas.width !== w || this.overlayCanvas.height !== h) {
      this.overlayCanvas.width = w;
      this.overlayCanvas.height = h;
      this.overlayCtx.clearRect(0, 0, w, h);
    }
  }

  /**
   * Выбор тензора маски из выходов модели: допускаем [1,h,w,1] | [h,w,1] | [1,h,w] | [h,w].
   */
  private pickMaskTensor(out: MaybeTensor): Tensor3D {
    const isCandidate = (t: Tensor) => {
      const r = t.rank;
      const s = t.shape;
      return (
        (r === 4 && s[0] === 1 && (s[3] === 1 || s[3] === 0)) ||
        (r === 3 && (s[2] === 1 || s[2] === 0)) ||
        (r === 2)
      );
    };

    let mask: Tensor | null = null;

    if (Array.isArray(out)) {
      for (const t of out) {
        if (isCandidate(t)) { mask = t; break; }
      }
      if (!mask && out.length) mask = out[out.length - 1];
    } else {
      mask = out;
    }

    if (!mask) throw new Error("Выход модели пуст, маска не найдена");

    if (mask.rank === 4) {
      mask = mask.squeeze([0]) as Tensor3D; // [1,h,w,1] -> [h,w,1]
    } else if (mask.rank === 2) {
      mask = (mask as Tensor).expandDims(2) as Tensor3D; // [h,w] -> [h,w,1]
    }

    return mask as Tensor3D;
  }

  /**
   * Инференс + отрисовка RGBA-оверлея на canvas.
   * Важно: ни одного await внутри tf.tidy().
   */
  async inferAndOverlay(video: HTMLVideoElement) {
    if (!this.loaded || this.busy) return;
    if (video.readyState < 2) return;

    this.resizeOverlayToVideo();
    const outH = this.overlayCanvas.height;
    const outW = this.overlayCanvas.width;

    this.busy = true;
    try {
      // ВАРИАНТ 1 — синхронный путь (предпочтительный и быстрый)
      let rgba: Tensor3D | null = null;
      let syncOk = true;

      try {
        rgba = tf.tidy(() => {
          const frame = tf.browser.fromPixels(video);
          const input = tf.image
            .resizeBilinear(frame, [this.inputH, this.inputW], true)
            .toFloat()
            .mul(1 / 255)
            .expandDims(0) as Tensor4D;

          const out = this.model!.execute(input) as Tensor | Tensor[];
          let mask = this.pickMaskTensor(out); // [h,w,1]
          // Если модель возвращает логиты, можно включить сигмоиду:
          // mask = tf.sigmoid(mask) as Tensor3D;

          const binary = mask.greater(this.thresh).toFloat();
          const inverted = tf.sub(1, binary);
          const up = tf.image.resizeNearestNeighbor(
            inverted as Tensor3D,
            [outH, outW],
            true
          );
          const a = (up as Tensor3D).mul(this.alpha);
          const z = tf.zerosLike(a);
          const o = tf.onesLike(a);
          return tf.concat([z, o, z, a], 2) as Tensor3D; // RGBA
        });
      } catch {
        syncOk = false;
      }

      // ВАРИАНТ 2 — fallback на async-граф (await ВНЕ tidy)
      if (!syncOk) {
        const input = tf.tidy(() => {
          const f = tf.browser.fromPixels(video);
          return tf.image
            .resizeBilinear(f, [this.inputH, this.inputW], true)
            .toFloat()
            .mul(1 / 255)
            .expandDims(0) as Tensor4D;
        });

        const out = await this.model!.executeAsync(input);
        input.dispose();

        const outArr = Array.isArray(out) ? out : [out];

        const rgbaAsync = tf.tidy(() => {
          // используем выходы, созданные ВНЕ tidy — tidy их не удалит,
          // поэтому ниже вручную вызовем dispose для outArr
          let mask = this.pickMaskTensor(Array.isArray(out) ? out : outArr);
          // mask = tf.sigmoid(mask) as Tensor3D; // если нужно
          const binary = mask.greater(this.thresh).toFloat();
          const inverted = tf.sub(1, binary);
          const up = tf.image.resizeNearestNeighbor(
            inverted as Tensor3D,
            [outH, outW],
            true
          );
          const a = (up as Tensor3D).mul(this.alpha);
          const z = tf.zerosLike(a);
          const o = tf.onesLike(a);
          return tf.concat([z, o, z, a], 2) as Tensor3D;
        });

        // ручной dispose выходов async-выполнения
        outArr.forEach(t => t.dispose());
        rgba = rgbaAsync;
      }

      if (rgba) {
        await tf.browser.toPixels(rgba, this.overlayCanvas);
        rgba.dispose();
      }
    } catch {
      // пропускаем кадр без падения
    } finally {
      this.busy = false;
    }
  }

  clearOverlay() {
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }
}

export const engine = new SegmentationEngine();
