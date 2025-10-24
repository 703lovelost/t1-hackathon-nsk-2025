// BlurryCamDemo.jsx
import React, { useRef, useEffect, useLayoutEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const BlurryCamDemo = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null); // основной видимый canvas
  const maskCanvasRef = useRef(typeof document !== 'undefined' ? document.createElement('canvas') : null);
  const overlayCanvasRef = useRef(typeof document !== 'undefined' ? document.createElement('canvas') : null);
  const [segmenter, setSegmenter] = useState(null);
  const rafId = useRef(null);

  const loadSegmentation = async () => {
    const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
    const segmenterConfig = { runtime: 'tfjs', modelType: 'general' };
    const s = await bodySegmentation.createSegmenter(model, segmenterConfig);
    setSegmenter(s);
  };

  const loop = async () => {
    const video = webcamRef.current?.video;
    const canvas = canvasRef.current;
    if (!video || !canvas || !segmenter) {
      rafId.current = requestAnimationFrame(loop);
      return;
    }

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw; canvas.height = vh;
    }
    const ctx = canvas.getContext('2d');

    tf.engine().startScope();
    try {
      // 1) Сырое видео — всегда сначала
      ctx.globalCompositeOperation = 'source-over';
      ctx.clearRect(0, 0, vw, vh);
      ctx.drawImage(video, 0, 0, vw, vh);

      // 2) Сегментация -> ч/б maskCanvas
      let haveMask = false;
      try {
        const segmentations = await segmenter.segmentPeople(video);
        const subject = segmentations?.[0];
        if (subject?.mask) {
          const imageData = await subject.mask.toImageData(); // RGBA, 0/255
          const data = imageData.data;
          // Приводим к чистой маске: белый (255) — человек, прозрачный — фон
          for (let i = 0; i < data.length; i += 4) {
            const v = data[i];         // 0 или 255
            data[i + 0] = 255;         // делаем белый для наглядности операции IN
            data[i + 1] = 255;
            data[i + 2] = 255;
            data[i + 3] = v > 127 ? 255 : 0; // альфа фона = 0 (полная прозрачность)
          }

          const m = maskCanvasRef.current;
          if (m.width !== imageData.width || m.height !== imageData.height) {
            m.width = imageData.width; m.height = imageData.height;
          }
          m.getContext('2d').putImageData(imageData, 0, 0);

          // 3) Готовим overlay: зелёный слой, вырезанный по маске
          const o = overlayCanvasRef.current;
          if (o.width !== vw || o.height !== vh) {
            o.width = vw; o.height = vh;
          }
          const octx = o.getContext('2d');
          octx.clearRect(0, 0, vw, vh);

          // 3.1 Заливаем зелёным с нужной прозрачностью
          octx.globalCompositeOperation = 'source-over';
          octx.fillStyle = 'rgba(0,255,0,0.5)'; // полупрозрачный зелёный
          octx.fillRect(0, 0, vw, vh);

          // 3.2 Оставляем только область маски
          octx.globalCompositeOperation = 'destination-in';
          // Масштабируем маску до размеров видео при отрисовке
          octx.drawImage(m, 0, 0, vw, vh);

          // 4) Кладём overlay поверх видео
          ctx.globalCompositeOperation = 'source-over';
          ctx.drawImage(o, 0, 0, vw, vh);

          haveMask = true;
        }
      } catch {
        // пропускаем кадр
      }

      // если маски нет — остаётся только сырое видео
      // ===== КОД БЛЮРА ОТКЛЮЧЁН =====
    } finally {
      tf.engine().endScope();
    }

    rafId.current = requestAnimationFrame(loop);
  };

  useLayoutEffect(() => {
    (async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      await loadSegmentation();
    })();
  }, []);

  useEffect(() => {
    if (!segmenter) return;
    const start = () => {
      if (rafId.current) cancelAnimationFrame(rafId.current);
      rafId.current = requestAnimationFrame(loop);
    };
    const v = webcamRef.current?.video;
    if (v && v.readyState >= 2) start();
    else if (v) v.onloadeddata = start;

    return () => {
      if (rafId.current) cancelAnimationFrame(rafId.current);
      segmenter?.dispose?.();
    };
  }, [segmenter]);

  return (
    <div>
      <Webcam ref={webcamRef} width={640} height={480} style={{ display: 'none' }} videoConstraints={{ facingMode: 'user' }} />
      <canvas ref={canvasRef} style={{ width: '100%', height: 'auto', display: 'block' }} />
    </div>
  );
};

export default BlurryCamDemo;
