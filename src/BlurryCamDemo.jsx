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

let recentImageDataList = [];

  function processImageData(imageData) {
      // Add the new ImageData to the list
      recentImageDataList.push(imageData);
      
      // Remove the oldest one if we have more than 3
      if (recentImageDataList.length > 3) {
          // Clean up the oldest ImageData by removing it from the array
          recentImageDataList.shift();
      }
      
      // If we have exactly 3, compute the average
      if (recentImageDataList.length === 3) {
          return calculateAverageImageData(recentImageDataList);
      }

      const width = imageData.width;
      const height = imageData.height;
      
      return new ImageData(width, height); // Return null if we don't have 3 images yet
  }

  function calculateAverageImageData(imageDataArray) {
      // All ImageData objects should have the same dimensions
      const width = imageDataArray[0].width;
      const height = imageDataArray[0].height;
      
      // Create a new ImageData object for the result
      const resultImageData = new ImageData(width, height);
      const resultData = resultImageData.data;
      
      const numImages = imageDataArray.length;

      const weights = [0.1, 0.35, 1.0];
      const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
      
      // Iterate through each pixel (4 channels: RGBA)
      for (let i = 0; i < resultData.length; i += 4) {
          let rSum = 0, gSum = 0, bSum = 0, aSum = 0;
          
          // Sum values from all images for this pixel
          for (let j = 0; j < numImages; j++) {
              const currentData = imageDataArray[j].data;
              const weight = weights[j];
              rSum += currentData[i] * weight;     // Red
              gSum += currentData[i + 1] * weight; // Green
              bSum += currentData[i + 2] * weight; // Blue
              aSum += currentData[i + 3] * weight; // Alpha
          }
          
          // Calculate averages
          resultData[i] = rSum / totalWeight;     // Average Red
          resultData[i + 1] = gSum / totalWeight; // Average Green
          resultData[i + 2] = bSum / totalWeight; // Average Blue
          resultData[i + 3] = aSum / totalWeight; // Average Alpha
      }
      
      return resultImageData;
  }

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
        const raw_mask = await segmentations[0].mask.toImageData();
        const mask = processImageData(raw_mask);

        // var toType = function(obj) {
        //   return ({}).toString.call(obj).match(/\s([a-zA-Z]+)/)[1].toLowerCase()
        // }
        // console.log('Mask type: ', toType(mask), toType(mask_raw));

        // console.log(`Shape: ${mask.height} x ${mask.width} x ${4}`);
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
