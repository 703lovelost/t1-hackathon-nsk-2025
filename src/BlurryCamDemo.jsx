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
  const [backgroundImage, setBackgroundImage] = useState(null);
  const backgroundRef = useRef(null);

  // Загрузка модели сегментации
  const loadSegmentation = async () => {
    const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
    const segmenterConfig = { runtime: 'tfjs', modelType: 'general' };
    const seg = await bodySegmentation.createSegmenter(model, segmenterConfig);
    setSegmenter(seg);
  };

  // Загрузка изображения из input
  const handleBackgroundChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const img = new Image();
      img.onload = () => {
        backgroundRef.current = img;
        setBackgroundImage(URL.createObjectURL(file));
      };
      img.src = URL.createObjectURL(file);
    }
  };

  // Применение маски и вставка пользовательского фона
  const applyMaskWithCustomBackground = (video, maskImageData, outputCanvas) => {
    const width = video.videoWidth;
    const height = video.videoHeight;
    const ctx = outputCanvas.getContext('2d');

    ctx.clearRect(0, 0, width, height);

    // Сначала нарисуем фон
    if (backgroundRef.current) {
      ctx.drawImage(backgroundRef.current, 0, 0, width, height);
    } else {
      ctx.fillStyle = 'green';
      ctx.fillRect(0, 0, width, height);
    }

    // Затем применим маску и добавим изображение с камеры поверх
    const videoImageData = new ImageData(width, height);
    const tempCtx = document.createElement('canvas').getContext('2d');
    tempCtx.canvas.width = width;
    tempCtx.canvas.height = height;
    tempCtx.drawImage(video, 0, 0, width, height);
    const videoPixels = tempCtx.getImageData(0, 0, width, height).data;
    const maskPixels = maskImageData.data;

    const outputPixels = ctx.getImageData(0, 0, width, height);
    const outputData = outputPixels.data;

    for (let i = 0; i < outputData.length; i += 4) {
      const maskValue = maskPixels[i]; // 0 = фон, 255 = человек
      if (maskValue > 127) {
        outputData[i] = videoPixels[i];
        outputData[i + 1] = videoPixels[i + 1];
        outputData[i + 2] = videoPixels[i + 2];
        outputData[i + 3] = 255;
      }
    }

    ctx.putImageData(outputPixels, 0, 0);
  };

  // Основной цикл обработки видео
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

        if (!segmentations.length) {
          if (backgroundRef.current) {
            ctx.drawImage(backgroundRef.current, 0, 0, outputCanvas.width, outputCanvas.height);
          } else {
            ctx.fillStyle = 'green';
            ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
          }
        } else {
          const mask = await segmentations[0].mask.toImageData();
          applyMaskWithCustomBackground(video, mask, outputCanvas);
        }
      } catch (e) {
        console.warn('Segmentation error:', e);
      }

      rafRef.current = requestAnimationFrame(updateCanvasLoop);
    };

    updateCanvasLoop();

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
    <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ marginBottom: '16px' }}>
        <label htmlFor="bgUpload" style={{ cursor: 'pointer', fontSize: '16px', color: '#61dafb' }}>
          Выберите фоновое изображение:
        </label>
        <input id="bgUpload" type="file" accept="image/*" onChange={handleBackgroundChange} />
      </div>

      <div style={{ position: 'relative', display: 'flex', justifyContent: 'center' }}>
        <Webcam ref={webcamRef} width={640} height={480} />
        <canvas
          ref={outputCanvasRef}
          width={640}
          height={480}
          style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)' }}
        />
      </div>

      {backgroundImage && (
        <p style={{ marginTop: '10px', fontSize: '14px', color: '#ccc' }}>
          Выбран фон: {backgroundImage.split('/').pop()}
        </p>
      )}
    </div>
  );
};

export default BlurryCamDemo;
