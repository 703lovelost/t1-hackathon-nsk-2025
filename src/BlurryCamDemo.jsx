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
  const [currentIndex, setCurrentIndex] = useState(0);
  const [customBackgrounds, setCustomBackgrounds] = useState([]);

  // дефолтные фоны
  const defaultBackgrounds = [
    '/backgrounds/back1.png',
    '/backgrounds/back2.png',
    '/backgrounds/back3.png',
    '/backgrounds/back4.png',
    '/backgrounds/back5.png',
  ];

  // объединённый список фонов
  const allBackgrounds = [...defaultBackgrounds, ...customBackgrounds];

  // переключение карусели
  const nextSlide = () => setCurrentIndex((prev) => (prev + 2) % allBackgrounds.length);
  const prevSlide = () => setCurrentIndex((prev) => (prev - 2 + allBackgrounds.length) % allBackgrounds.length);

  // загрузка модели
  const loadSegmentation = async () => {
    const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
    const segmenterConfig = { runtime: 'tfjs', modelType: 'general' };
    const seg = await bodySegmentation.createSegmenter(model, segmenterConfig);
    setSegmenter(seg);
  };

  // пользователь загружает фон
  const handleBackgroundChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const objectURL = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        backgroundRef.current = img;
        setBackgroundImage(objectURL);
        setCustomBackgrounds((prev) => [...prev, objectURL]); // добавляем в карусель
      };
      img.src = objectURL;
    }
  };

  // единый обработчик выбора фона
  const handleBackgroundSelect = (src) => {
    const img = new Image();
    img.crossOrigin = 'anonymous'; // важно для blob и локальных путей
    img.onload = () => {
      backgroundRef.current = img;
      setBackgroundImage(src);
    };
    img.src = src;
  };

  // применение маски
  const applyMaskWithCustomBackground = (video, maskImageData, outputCanvas) => {
    const width = video.videoWidth;
    const height = video.videoHeight;
    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    if (backgroundRef.current) {
      ctx.drawImage(backgroundRef.current, 0, 0, width, height);
    } else {
      ctx.fillStyle = 'green';
      ctx.fillRect(0, 0, width, height);
    }

    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = width;
    tempCanvas.height = height;
    tempCtx.drawImage(video, 0, 0, width, height);
    const videoData = tempCtx.getImageData(0, 0, width, height).data;
    const maskData = maskImageData.data;

    const outputImageData = ctx.getImageData(0, 0, width, height);
    const outputPixels = outputImageData.data;

    for (let i = 0; i < outputPixels.length; i += 4) {
      const maskValue = maskData[i];
      if (maskValue > 127) {
        outputPixels[i] = videoData[i];
        outputPixels[i + 1] = videoData[i + 1];
        outputPixels[i + 2] = videoData[i + 2];
        outputPixels[i + 3] = 255;
      }
    }

    ctx.putImageData(outputImageData, 0, 0);
  };

  // цикл отрисовки
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
      processVideo().then((fn) => (cleanupFn = fn));
    }
    return () => {
      if (cleanupFn) cleanupFn();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      segmenter?.dispose?.();
      tf.disposeVariables();
    };
  }, [segmenter]);

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        color: '#fff',
        overflow: 'hidden',
        fontFamily: 'Inter, sans-serif',
      }}
    >
      {/* Видео */}
      <div
        style={{
          position: 'relative',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
        }}
      >
        <Webcam ref={webcamRef} width={640} height={480} />
        <canvas
          ref={outputCanvasRef}
          width={640}
          height={480}
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        />
      </div>

      {/* Блок выбора фона + карусель */}
      <div
        style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          background: 'rgba(255, 255, 255, 0.08)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          padding: '16px 20px',
          borderRadius: '20px',
          boxShadow: '0 8px 20px rgba(0,0,0,0.5)',
          backdropFilter: 'blur(12px)',
        }}
      >
        {/* Кнопка загрузки */}
        <label
          htmlFor="bgUpload"
          style={{
            background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
            color: 'white',
            padding: '10px 20px',
            borderRadius: '12px',
            cursor: 'pointer',
            fontSize: '15px',
            fontWeight: '500',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 10px rgba(59,130,246,0.4)',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
          onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1.0)')}
        >
          Загрузите свой фон
        </label>
        <input id="bgUpload" type="file" accept="image/*" onChange={handleBackgroundChange} style={{ display: 'none' }} />

        {/* Карусель */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={prevSlide}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              fontSize: '28px',
            }}
          >
            &lt;
          </button>

          <div style={{ display: 'flex', overflow: 'hidden', width: '300px', gap: '12px' }}>
            {[
              ...allBackgrounds.slice(currentIndex, currentIndex + 2),
              ...(currentIndex + 2 > allBackgrounds.length
                ? allBackgrounds.slice(0, (currentIndex + 2) % allBackgrounds.length)
                : []),
            ].map((src, idx) => (
              <img
                key={idx}
                src={src}
                alt={`background-${idx}`}
                onClick={() => handleBackgroundSelect(src)}
                style={{
                  width: '140px',
                  height: '100px',
                  objectFit: 'cover',
                  borderRadius: '12px',
                  border:
                    backgroundImage === src
                      ? '3px solid #3b82f6'
                      : '2px solid rgba(255,255,255,0.2)',
                  cursor: 'pointer',
                  transition: 'transform 0.2s, border 0.2s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
                onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1.0)')}
              />
            ))}
          </div>

          <button
            onClick={nextSlide}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              fontSize: '28px',
            }}
          >
            &gt;
          </button>
        </div>
      </div>
    </div>
  );
};

export default BlurryCamDemo;
