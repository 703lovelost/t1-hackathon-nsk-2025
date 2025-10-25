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
          return calculateWeightedAverageImageData(recentImageDataList);
      }

      const width = imageData.width;
      const height = imageData.height;
      
      return new ImageData(width, height); // Return null if we don't have 3 images yet
  }

  function calculateWeightedAverageImageData(imageDataArray) {
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

  // function applyBilateralFilter(imageData, sigmaSpace = 3, sigmaColor = 0.1) {
  //     const width = imageData.width;
  //     const height = imageData.height;
  //     const data = imageData.data;
      
  //     // Создаем новый ImageData для результата
  //     const resultImageData = new ImageData(width, height);
  //     const resultData = resultImageData.data;
      
  //     // Вычисляем радиус окна на основе sigmaSpace
  //     const radius = Math.floor(sigmaSpace * 1.5);
      
  //     // Предварительно вычисляем пространственное гауссово ядро
  //     const spatialKernel = createSpatialKernel(radius, sigmaSpace);
      
  //     // Копируем исходные данные (для альфа-канала и чтобы не портить исходные)
  //     const sourceData = new Float32Array(data.length);
  //     for (let i = 0; i < data.length; i++) {
  //         sourceData[i] = data[i] / 255; // Нормализуем в [0, 1]
  //         resultData[i] = data[i]; // Копируем исходные значения
  //     }
      
  //     // Применяем двусторонний фильтр к каждому пикселю
  //     for (let y = 0; y < height; y++) {
  //         for (let x = 0; x < width; x++) {
  //             const idx = (y * width + x) * 4;
              
  //             // Пропускаем полностью прозрачные пиксели
  //             if (sourceData[idx + 3] === 0) continue;
              
  //             // Получаем интенсивность текущего пикселя (берем красный канал для маски)
  //             const centerIntensity = sourceData[idx];
              
  //             let sumWeights = 0;
  //             let sumValues = 0;
              
  //             // Проходим по окрестности
  //             for (let dy = -radius; dy <= radius; dy++) {
  //                 for (let dx = -radius; dx <= radius; dx++) {
  //                     const nx = x + dx;
  //                     const ny = y + dy;
                      
  //                     // Проверяем границы
  //                     if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                      
  //                     const nidx = (ny * width + nx) * 4;
                      
  //                     // Пропускаем полностью прозрачные пиксели
  //                     if (sourceData[nidx + 3] === 0) continue;
                      
  //                     // Получаем интенсивность соседнего пикселя
  //                     const neighborIntensity = sourceData[nidx];
                      
  //                     // Вычисляем пространственный вес
  //                     const spatialWeight = spatialKernel[dy + radius][dx + radius];
                      
  //                     // Вычисляем цветовой вес (разность интенсивностей)
  //                     const intensityDiff = Math.abs(centerIntensity - neighborIntensity);
  //                     const colorWeight = Math.exp(-(intensityDiff * intensityDiff) / (2 * sigmaColor * sigmaColor));
                      
  //                     // Общий вес = пространственный * цветовой
  //                     const totalWeight = spatialWeight * colorWeight;
                      
  //                     sumWeights += totalWeight;
  //                     sumValues += neighborIntensity * totalWeight;
  //                 }
  //             }
              
  //             // Вычисляем отфильтрованное значение
  //             if (sumWeights > 0) {
  //                 const filteredIntensity = sumValues / sumWeights;
  //                 // Записываем результат во все цветовые каналы (для градаций серого)
  //                 const intensity = Math.round(filteredIntensity * 255);
  //                 resultData[idx] = intensity;     // R
  //                 resultData[idx + 1] = intensity; // G
  //                 resultData[idx + 2] = intensity; // B
  //                 // Alpha оставляем без изменений
  //             }
  //         }
  //     }
      
  //     return resultImageData;
  // }

  // function createSpatialKernel(radius, sigma) {
  //     const size = radius * 2 + 1;
  //     const kernel = Array(size);
  //     const sigma2 = sigma * sigma;
      
  //     for (let i = 0; i < size; i++) {
  //         kernel[i] = Array(size);
  //         for (let j = 0; j < size; j++) {
  //             const dx = j - radius;
  //             const dy = i - radius;
  //             const distance2 = dx * dx + dy * dy;
  //             kernel[i][j] = Math.exp(-distance2 / (2 * sigma2));
  //         }
  //     }
      
  //     return kernel;
  // }

  // // Оптимизированная версия для бинарных масок (только 0 и 255)
  // function applyBilateralFilterBinary(imageData, sigmaSpace = 3, sigmaColor = 0.1) {
  //     const width = imageData.width;
  //     const height = imageData.height;
  //     const data = imageData.data;
      
  //     const resultImageData = new ImageData(width, height);
  //     const resultData = resultImageData.data;
      
  //     const radius = Math.floor(sigmaSpace * 1.5);
  //     const spatialKernel = createSpatialKernel(radius, sigmaSpace);
      
  //     // Копируем исходные данные и нормализуем в [0, 1]
  //     const sourceData = new Float32Array(width * height);
  //     for (let y = 0; y < height; y++) {
  //         for (let x = 0; x < width; x++) {
  //             const idx = (y * width + x) * 4;
  //             const maskIdx = y * width + x;
  //             // Используем красный канал и нормализуем
  //             sourceData[maskIdx] = data[idx] / 255;
  //             // Копируем исходные данные
  //             resultData[idx] = data[idx];
  //             resultData[idx + 1] = data[idx + 1];
  //             resultData[idx + 2] = data[idx + 2];
  //             resultData[idx + 3] = data[idx + 3];
  //         }
  //     }
      
  //     // Применяем фильтр
  //     for (let y = 0; y < height; y++) {
  //         for (let x = 0; x < width; x++) {
  //             const idx = (y * width + x) * 4;
  //             const maskIdx = y * width + x;
              
  //             // Пропускаем полностью прозрачные пиксели
  //             if (data[idx + 3] === 0) continue;
              
  //             const centerIntensity = sourceData[maskIdx];
              
  //             let sumWeights = 0;
  //             let sumValues = 0;
              
  //             for (let dy = -radius; dy <= radius; dy++) {
  //                 for (let dx = -radius; dx <= radius; dx++) {
  //                     const nx = x + dx;
  //                     const ny = y + dy;
                      
  //                     if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                      
  //                     const nidx = (ny * width + nx) * 4;
  //                     const nmaskIdx = ny * width + nx;
                      
  //                     if (data[nidx + 3] === 0) continue;
                      
  //                     const neighborIntensity = sourceData[nmaskIdx];
  //                     const spatialWeight = spatialKernel[dy + radius][dx + radius];
  //                     const intensityDiff = Math.abs(centerIntensity - neighborIntensity);
  //                     const colorWeight = Math.exp(-(intensityDiff * intensityDiff) / (2 * sigmaColor * sigmaColor));
  //                     const totalWeight = spatialWeight * colorWeight;
                      
  //                     sumWeights += totalWeight;
  //                     sumValues += neighborIntensity * totalWeight;
  //                 }
  //             }
              
  //             if (sumWeights > 0) {
  //                 const filteredIntensity = sumValues / sumWeights;
  //                 const intensity = Math.round(filteredIntensity * 255);
  //                 resultData[idx] = intensity;
  //                 resultData[idx + 1] = intensity;
  //                 resultData[idx + 2] = intensity;
  //             }
  //         }
  //     }
      
  //     return resultImageData;
  // }

  // // Функция для последовательной обработки с сохранением последних кадров
  // let recentSegmentationMasks = [];

  // function processSegmentationMask(segmentationMask) {
  //     // Добавляем новую маску в список
  //     // recentSegmentationMasks.push(segmentationMask);
      
  //     // // Удаляем самую старую, если больше 3
  //     // if (recentSegmentationMasks.length > 3) {
  //     //     recentSegmentationMasks.shift();
  //     // }
      
  //     // // Если накопилось 3 маски, применяем взвешенное усреднение
  //     // if (recentSegmentationMasks.length === 3) {
  //     //     const averagedMask = calculateWeightedAverageImageData(recentSegmentationMasks);
  //     //     // Затем применяем двусторонний фильтр к усредненной маске
  //     //     return applyBilateralFilter(averagedMask);
  //     // }
      
  //     // Если меньше 3 масок, применяем фильтр к текущей
  //     return applyBilateralFilter(segmentationMask);
  // }

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
        // const mask = processSegmentationMask(raw_mask);

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
