import React, { useRef, useEffect, useLayoutEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const BlurryCamDemo = () => {
  const webcamRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const [segmenter, setSegmenter] = useState(null);

  // Загрузка модели сегментации
  const loadSegmentation = async () => {
    const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
    const segmenterConfig = {
      runtime: 'tfjs',
      modelType: 'general',
    };
    const segmenter = await bodySegmentation.createSegmenter(model, segmenterConfig);
    setSegmenter(segmenter);
  };

  // Функция для применения маски и замены фона на зеленый
  const applyMaskWithGreenBackground = (video, maskImageData, outputCanvas) => {
    const width = video.videoWidth;
    const height = video.videoHeight;

    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    // Берем исходное изображение
    ctx.drawImage(video, 0, 0, width, height);
    const videoImageData = ctx.getImageData(0, 0, width, height);

    const videoPixels = videoImageData.data;
    const maskPixels = maskImageData.data;

    for (let i = 0; i < videoPixels.length; i += 4) {
      // Берем значение маски с одного канала (красного)
      const maskValue = maskPixels[i];

      if (maskValue > 127) {
        // Пиксель принадлежит человеку — оставляем исходный цвет (можно затемнять по желанию)
        continue;
      } else {
        // Фон — красим в зеленый
        videoPixels[i] = 0;       // R
        videoPixels[i + 1] = 255; // G
        videoPixels[i + 2] = 0;   // B
        videoPixels[i + 3] = 255; // A (полностью непрозрачный)
      }
    }

    ctx.putImageData(videoImageData, 0, 0);
  };

  // Основной цикл обработки видео
  const processVideo = async () => {
    if (!segmenter) return;
    const video = webcamRef.current.video;
    const outputCanvas = outputCanvasRef.current;

    const updateCanvasLoop = async () => {
      if (video.readyState < 2) {
        // Видео еще не готово — ждем
        requestAnimationFrame(updateCanvasLoop);
        return;
      }

      const segmentations = await segmenter.segmentPeople(video);
      if (segmentations.length === 0) {
        // Человека не обнаружено — просто зеленый экран
        const ctx = outputCanvas.getContext('2d');
        ctx.fillStyle = 'green';
        ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
        requestAnimationFrame(updateCanvasLoop);
        return;
      }

      const subject = segmentations[0];
      const maskImageData = await subject.mask.toImageData();

      applyMaskWithGreenBackground(video, maskImageData, outputCanvas);

      requestAnimationFrame(updateCanvasLoop);
    };

    updateCanvasLoop();
  };

  useLayoutEffect(() => {
    tf.setBackend('webgl');
    loadSegmentation();
  }, []);

  useEffect(() => {
    if (segmenter && webcamRef.current && webcamRef.current.video) {
      processVideo();
    }
  }, [segmenter, webcamRef.current]);

  return (
    <div>
      <Webcam ref={webcamRef} width={640} height={480} />
      <canvas ref={outputCanvasRef} width={640} height={480} style={{position: 'absolute', top: 0, left: 0}} />
    </div>
  );
};

export default BlurryCamDemo;