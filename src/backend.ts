import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import "@tensorflow/tfjs-backend-wasm";

export async function pickBestBackend(): Promise<string> {
  try {
    await tf.setBackend("webgpu");
    await tf.ready();
    if (tf.getBackend() === "webgpu") return "webgpu";
  } catch {}

  try {
    if (typeof (tf as any).setWasmPaths === "function") {
      // грузим wasm из CDN, чтобы не настраивать копирование артефактов
      (tf as any).setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4/dist/");
    }
    tf.env().set("WASM_NUM_THREADS", Math.min(4, navigator.hardwareConcurrency || 4));
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