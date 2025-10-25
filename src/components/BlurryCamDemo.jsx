import { useRef, useEffect, useLayoutEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const BlurryCamDemo = ({ videoRef, mediaStream, useYolo }) => {
  const webcamRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const rafRef = useRef(null);
  const [segmenter, setSegmenter] = useState(null);

  const loadSegmentation = async () => {
    try {
      const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
      const segmenterConfig = { runtime: 'tfjs', modelType: 'general' };
      const seg = await bodySegmentation.createSegmenter(model, segmenterConfig);
      setSegmenter(seg);
    } catch (e) {
      console.error('Ошибка загрузки модели сегментации:', e);
    }
  };

  const applyMaskWithGreenBackground = (video, maskImageData, outputCanvas) => {
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    const ctx = outputCanvas.getContext('2d');

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(video, 0, 0, width, height);
    const videoImageData = ctx.getImageData(0, 0, width, height);
    const videoPixels = videoImageData.data;
    const maskPixels = maskImageData?.data;

    if (maskPixels) {
      for (let i = 0; i < videoPixels.length; i += 4) {
        const maskValue = maskPixels[i];
        if (maskValue <= 127) {
          videoPixels[i] = 0; // R
          videoPixels[i + 1] = 255; // G
          videoPixels[i + 2] = 0; // B
          videoPixels[i + 3] = 255; // A
        }
      }
      ctx.putImageData(videoImageData, 0, 0);
    } else {
      // Если маска отсутствует, просто рисуем видео
      ctx.drawImage(video, 0, 0, width, height);
    }
  };

  const processVideo = async () => {
    if (!segmenter || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    const outputCanvas = outputCanvasRef.current;

    if (!video || !outputCanvas) return;

    let active = true;

    const updateCanvasLoop = async () => {
      if (!active || !video || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(updateCanvasLoop);
        return;
      }

      try {
        const segmentations = await segmenter.segmentPeople(video);
        const mask = segmentations[0] ? await segmentations[0].mask.toImageData() : null;
        applyMaskWithGreenBackground(video, mask, outputCanvas);
        if (segmentations[0]) {
          (await segmentations[0].mask.toTensor()).dispose();
        }
      } catch (e) {
        console.warn('Ошибка сегментации:', e);
        // Рисуем видео без маски в случае ошибки
        const ctx = outputCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 640, 480);
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
    if (segmenter && webcamRef.current?.video) {
      processVideo().then((fn) => (cleanupFn = fn));
    }
    return () => {
      if (cleanupFn) cleanupFn();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      segmenter?.dispose?.();
      tf.disposeVariables();
    };
  }, [segmenter]);

  useEffect(() => {
    if (webcamRef.current && mediaStream) {
      webcamRef.current.video.srcObject = mediaStream;
      webcamRef.current.video.play().catch((e) => console.error('Ошибка воспроизведения видео:', e));
    }
  }, [mediaStream]);

  return (
    <>
      <Webcam
        ref={(node) => {
          webcamRef.current = node;
          if (node) videoRef.current = node.video;
        }}
        width={640}
        height={480}
        style={{ display: 'none' }}
        videoConstraints={{ width: 640, height: 480 }}
      />
      <canvas
        ref={outputCanvasRef}
        width={640}
        height={480}
        style={{
          width: '640px',
          height: '480px',
          position: 'absolute',
          top: 0,
          left: 0,
          transform: 'scaleX(-1)', // Зеркальное отображение
        }}
      />
    </>
  );
};

export default BlurryCamDemo;