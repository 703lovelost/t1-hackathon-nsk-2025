import React, { useRef, useEffect, useLayoutEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const BlurryCamDemo = () => {
  const webcamRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const rafRef = useRef(null);
  const [segmenter, setSegmenter] = useState(null);

  // Загрузка модели сегментации
  const loadSegmentation = async () => {
    const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
    const segmenterConfig = { runtime: 'tfjs', modelType: 'general' };
    const seg = await bodySegmentation.createSegmenter(model, segmenterConfig);
    setSegmenter(seg);
  };

  // Применяем маску и заменяем фон на зелёный
  const applyMaskWithGreenBackground = (video, maskImageData, outputCanvas) => {
    const width = video.videoWidth;
    const height = video.videoHeight;
    const ctx = outputCanvas.getContext('2d');

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(video, 0, 0, width, height);
    const videoImageData = ctx.getImageData(0, 0, width, height);
    const videoPixels = videoImageData.data;
    const maskPixels = maskImageData.data;

    for (let i = 0; i < videoPixels.length; i += 4) {
      const maskValue = maskPixels[i];
      if (maskValue <= 127) {
        // фон — зелёный
        videoPixels[i] = 0;
        videoPixels[i + 1] = 255;
        videoPixels[i + 2] = 0;
        videoPixels[i + 3] = 255;
      }
    }
    ctx.putImageData(videoImageData, 0, 0);
  };

  // Основной цикл
  const processVideo = async () => {
    if (!segmenter) return;
    const video = webcamRef.current.video;
    const outputCanvas = outputCanvasRef.current;

    let active = true;

    const updateCanvasLoop = async () => {
      if (!active) return;
      if (!video || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(updateCanvasLoop);
        return;
      }

      try {
        const segmentations = await segmenter.segmentPeople(video);
        const ctx = outputCanvas.getContext('2d');
        const mask = await segmentations[0].mask.toImageData();
        applyMaskWithGreenBackground(video, mask, outputCanvas);
        (await segmentations[0].mask.toTensor()).dispose();
      } catch (e) {
        console.warn('Segmentation error:', e);
      }

      rafRef.current = requestAnimationFrame(updateCanvasLoop);
    };

    updateCanvasLoop();

    // Возврат функции очистки
    return () => {
      active = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  };

  useLayoutEffect(() => {
    (async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      await loadSegmentation();
    })();
  }, []);

  useEffect(() => {
    let cleanupFn = null;

    if (segmenter && webcamRef.current && webcamRef.current.video) {
      processVideo().then(fn => (cleanupFn = fn));
    }

    return () => {
      if (cleanupFn) cleanupFn();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      segmenter?.dispose?.();
      tf.disposeVariables();
    };
  }, [segmenter]);

  return (
    <div style={{ position: 'relative', display: 'flex', justifyContent: 'center' }}>
      <Webcam ref={webcamRef} width={640} height={480} />
      <canvas
        ref={outputCanvasRef}
        width={640}
        height={480}
        style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)' }}
      />
    </div>
  );
};

export default BlurryCamDemo;
