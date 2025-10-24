import * as tf from "@tensorflow/tfjs";
import * as tflite from '@tensorflow/tfjs-tflite';
import type { SegmentationRunner, MaskResult } from "./runner";

export class TFLiteSegmentationRunner implements SegmentationRunner {
  private modelPath: string;
  private model!: tflite.TFLiteModel;
  private inW = 640;
  private inH = 640;
  private threshold = 0.5;

  constructor(modelPath = "/models/yolo11m-seg_saved_model/yolo11m-seg_float32.tflite", opts?: { inputSize?: number; threshold?: number }) {
    this.modelPath = modelPath;
    if (opts?.inputSize) this.inW = this.inH = opts.inputSize;
    if (opts?.threshold != null) this.threshold = opts.threshold;
  }

  async init() {
    // Многопотоный WASM ускоряет инференс на CPU
    const numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
    this.model = await tflite.loadTFLiteModel(this.modelPath, { numThreads });
  }

  async warmup(video: HTMLVideoElement) {
    await tf.tidy(() => {
      const x = tf.browser.fromPixels(video).resizeBilinear([this.inH, this.inW]).toFloat().div(255).expandDims(0); // [1,H,W,3]
      const y = this.model.predict(x) as tf.Tensor;
      y.dataSync();
    });
  }

  async run(video: HTMLVideoElement): Promise<MaskResult | null> {
    if (!video.videoWidth) return null;

    const out = tf.tidy(() => {
      const x = tf.browser.fromPixels(video).resizeBilinear([this.inH, this.inW]).toFloat().div(255).expandDims(0); // NHWC
      const y = this.model.predict(x) as tf.Tensor; // форма может быть [1,H,W,1] / [1,1,H,W] — зависит от модели
      // Приведём к [H,W,1]
      let probs = y;
      if (y.shape.length === 4 && y.shape[0] === 1) {
        probs = y.squeeze([0]);
      }
      if (probs.shape.length === 3 && probs.shape[2] !== 1) {
        // если каналов >1 — берём первый
        probs = probs.slice([0, 0, 0], [probs.shape[0], probs.shape[1], 1]);
      }
      const bin = probs.greater(this.threshold).toFloat(); // бинаризация
      return bin;
    });

    const [oh, ow] = out.shape as [number, number, number];
    const data = new Float32Array(oh * ow);
    const tmp = await out.data();
    // tmp может быть [H*W*1]
    for (let i = 0; i < oh * ow; i++) data[i] = tmp[i];
    out.dispose();
    return { data, width: ow, height: oh };
  }

  dispose() {}
}
