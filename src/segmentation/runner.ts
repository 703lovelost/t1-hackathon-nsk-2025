export type MaskResult = {
  data: Float32Array;
  width: number;
  height: number;
};

export interface SegmentationRunner {
  init(): Promise<void>;
  warmup?(video: HTMLVideoElement): Promise<void>;
  run(video: HTMLVideoElement): Promise<MaskResult | null>;
  dispose?(): void;
}
