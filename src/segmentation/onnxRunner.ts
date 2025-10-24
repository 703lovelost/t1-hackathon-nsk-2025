// ONNX Runtime Web + WebGPU
import * as ort from "onnxruntime-web/webgpu";
import type { SegmentationRunner, MaskResult } from "./runner";

// Небольшой помощник для CPU-препроцессинга без зависимостей
function drawToSizedRGBA(video: HTMLVideoElement, w: number, h: number): Uint8ClampedArray {
  const can = document.createElement("canvas");
  can.width = w; can.height = h;
  const ctx = can.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(video, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

/** Преобразование RGBA -> CHW Float32 [1,3,H,W], нормализация [0..1] */
function rgbaToCHWFloat(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const out = new Float32Array(1 * 3 * h * w);
  const plane = h * w;
  for (let i = 0, px = 0; i < rgba.length; i += 4, px++) {
    const r = rgba[i] / 255, g = rgba[i + 1] / 255, b = rgba[i + 2] / 255;
    out[px] = r;
    out[plane + px] = g;
    out[2 * plane + px] = b;
  }
  return out;
}

export class OnnxSegmentationRunner implements SegmentationRunner {
  private modelPath: string;
  private session!: ort.InferenceSession;
  private inputName!: string;
  private inW = 640;
  private inH = 640;
  private outName!: string;
  private threshold = 0.5;

  constructor(modelPath = "/models/yolo.onnx", opts?: { inputSize?: number; threshold?: number }) {
    this.modelPath = modelPath;
    if (opts?.inputSize) this.inW = this.inH = opts.inputSize;
    if (opts?.threshold != null) this.threshold = opts.threshold;
  }

  async init() {
    // Включаем WebGPU EP
    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
      preferredOutputLocation: "gpu-buffer", // пусть результаты остаются на GPU, если возможно
    });
    // Имя входа и его размер
    this.inputName = this.session.inputNames[0];
    const meta: any = (this.session as any).inputMetadata?.[this.inputName];
    const dims: number[] | undefined = meta?.dimensions || meta?.dims;
    // Если в модели зашит статический размер — используем его
    if (Array.isArray(dims) && dims.length >= 4 && dims.every(n => typeof n === "number" && n !== -1)) {
      // Ожидаем NCHW
      this.inH = dims[dims.length - 2];
      this.inW = dims[dims.length - 1];
    }
    // Выберем первый выход как маску по умолчанию
    this.outName = this.session.outputNames[0];
  }

  async warmup(video: HTMLVideoElement) {
    const rgba = drawToSizedRGBA(video, this.inW, this.inH);
    const chw = rgbaToCHWFloat(rgba, this.inW, this.inH);
    const input = new ort.Tensor("float32", chw, [1, 3, this.inH, this.inW]);
    await this.session.run({ [this.inputName]: input });
  }

  async run(video: HTMLVideoElement): Promise<MaskResult | null> {
    if (!video.videoWidth || !video.videoHeight) return null;

    // CPU препроцесс (быстро, без лишних копий); для ещё большего ускорения можно добавить IO-binding.
    const rgba = drawToSizedRGBA(video, this.inW, this.inH);
    const chw = rgbaToCHWFloat(rgba, this.inW, this.inH);
    const feeds: Record<string, ort.Tensor> = {
      [this.inputName]: new ort.Tensor("float32", chw, [1, 3, this.inH, this.inW]),
    };

    const results = await this.session.run(feeds);
    const out = results[this.outName] as ort.Tensor;

    // Унифицируем форму выхода до [H, W]
    let oh = 0, ow = 0;
    if (out.dims.length === 4) { // [1,1,H,W] или [1,H,W,1]
      const [n, c, h, w] = out.dims;
      if (n === 1 && (c === 1 || w === 1)) {
        oh = h; ow = w;
      } else {
        // Если форма другая, попробуем найти 2 последние размерности как HxW
        oh = out.dims[out.dims.length - 2];
        ow = out.dims[out.dims.length - 1];
      }
    } else if (out.dims.length === 3) { // [1,H,W] или [H,W,1]
      oh = out.dims[out.dims.length - 2];
      ow = out.dims[out.dims.length - 1];
    } else if (out.dims.length === 2) { // [H,W]
      oh = out.dims[0]; ow = out.dims[1];
    } else {
      // Неизвестный формат — выходим
      return null;
    }

    const raw = (await out.data()) as Float32Array; // массив вероятностей/логитов
    // Превращаем в бинарную маску и возвращаем
    const bin = new Float32Array(oh * ow);
    if (raw.length === bin.length) {
      for (let i = 0; i < raw.length; i++) bin[i] = raw[i] > this.threshold ? 1 : 0;
    } else {
      // Если каналов >1, берём первый
      const stride = Math.floor(raw.length / bin.length);
      for (let i = 0, j = 0; i < raw.length && j < bin.length; i += stride, j++) {
        bin[j] = raw[i] > this.threshold ? 1 : 0;
      }
    }
    // NB: инверсию делаем уже при отрисовке, как просили
    return { data: bin, width: ow, height: oh };
  }

  dispose() {}
}
