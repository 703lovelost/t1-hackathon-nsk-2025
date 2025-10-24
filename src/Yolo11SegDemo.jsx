// Yolo11SegDemo.jsx
import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import { loadGraphModel } from '@tensorflow/tfjs-converter'; // <-- ВАЖНО: из converter!

const MODEL_PATH = '/yolo11m-seg-web_model/model.json';
const INPUT_SIZE = 640;
const PERSON_CLASS_ID = 0;
const CONFIDENCE_THRESHOLD = 0.3;
const IOU_THRESHOLD = 0.45;
const BLUR_RADIUS = 12;

const Yolo11SegDemo = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const animationId = useRef(null);

  // === Загрузка модели ===
  useEffect(() => {
    const loadModel = async () => {
      try {
        setLoading(true);
        await tf.setBackend('webgl');
        await tf.ready();

        const loadedModel = await loadGraphModel(MODEL_PATH); // <-- из converter
        setModel(loadedModel);
        console.log('YOLO11m-seg загружен');
      } catch (err) {
        console.error('Ошибка загрузки:', err);
        setError('Не удалось загрузить YOLO11m-seg. Проверьте путь к модели.');
      } finally {
        setLoading(false);
      }
    };

    loadModel();
  }, []);

  // === Предобработка ===
  const preprocess = (video) => {
    return tf.tidy(() => {
      const canvas = document.createElement('canvas');
      canvas.width = INPUT_SIZE;
      canvas.height = INPUT_SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);

      const img = tf.browser.fromPixels(canvas);
      return img.toFloat().div(255.0).expandDims(0); // [1, 640, 640, 3]
    });
  };

  // === NMS ===
  const nonMaxSuppression = (boxes, scores, iouThreshold) => {
    const indices = [];
    const sorted = Array.from(scores.dataSync())
      .map((s, i) => ({ score: s, index: i }))
      .sort((a, b) => b.score - a.score);

    const keep = new Array(scores.shape[0]).fill(true);

    for (const { index: i } of sorted) {
      if (!keep[i]) continue;

      for (let j = i + 1; j < scores.shape[0]; j++) {
        if (!keep[j]) continue;

        const box1 = boxes.slice([i, 0], [1, 4]).dataSync();
        const box2 = boxes.slice([j, 0], [1, 4]).dataSync();
        if (computeIOU(box1, box2) > iouThreshold) {
          keep[j] = false;
        }
      }
      indices.push(i);
    }
    return indices;
  };

  const computeIOU = (box1, box2) => {
    const [x1, y1, x2, y2] = box1;
    const [x1_, y1_, x2_, y2_] = box2;

    const xi1 = Math.max(x1, x1_);
    const yi1 = Math.max(y1, y1_);
    const xi2 = Math.min(x2, x2_);
    const yi2 = Math.min(y2, y2_);

    const inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
    const area1 = (x2 - x1) * (y2 - y1);
    const area2 = (x2_ - x1_) * (y2_ - y1_);
    return inter / (area1 + area2 - inter + 1e-6);
  };

  // === Постобработка YOLO ===
  const postprocess = async (outputs, width, height) => {
    return tf.tidy(() => {
      // YOLOv11-seg: [boxes, scores, classes, mask_coeffs, proto]
      const [boxes, scores, classes, maskCoeffs, proto] = outputs;

      const personMask = classes.squeeze().equal(PERSON_CLASS_ID);
      const confMask = scores.squeeze().greater(CONFIDENCE_THRESHOLD);
      const valid = personMask.and(confMask);

      const indices = tf.where(valid).squeeze();
      if (indices.shape[0] === 0) return tf.zeros([height, width]);

      const validBoxes = tf.gather(boxes.squeeze(), indices);
      const validScores = tf.gather(scores.squeeze(), indices);
      const validCoeffs = tf.gather(maskCoeffs.squeeze(), indices);

      const nmsIndices = nonMaxSuppression(validBoxes, validScores, IOU_THRESHOLD);
      if (nmsIndices.length === 0) return tf.zeros([height, width]);

      const selectedCoeffs = tf.gather(validCoeffs, nmsIndices);
      const selectedProto = tf.gather(proto.squeeze(), nmsIndices);

      // Маска: sigmoid(coeffs @ proto)
      const protoResized = tf.image.resizeBilinear(selectedProto, [height, width]);
      const mask = tf.matMul(selectedCoeffs, protoResized.reshape([selectedCoeffs.shape[0], -1]));
      const sigmoid = tf.sigmoid(mask).reshape([-1, height, width]);
      return tf.max(sigmoid, 0).greater(0.5).toFloat(); // Бинарная маска
    });
  };

  // === Эффект размытия ===
  const applyBokehEffect = (canvas, video, mask) => {
    const ctx = canvas.getContext('2d');
    const w = video.videoWidth;
    const h = video.videoHeight;

    // 1. Размытый фон
    ctx.filter = `blur(${BLUR_RADIUS}px)`;
    ctx.drawImage(video, 0, 0, w, h);
    ctx.filter = 'none';

    // 2. Четкий человек (по маске)
    const maskData = mask.dataSync();
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      if (maskData[i >> 2] > 0.5) {
        // Прозрачный фон → останется размытие
        data[i + 3] = 0;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // 3. Четкое изображение поверх
    ctx.globalCompositeOperation = 'destination-over';
    ctx.drawImage(video, 0, 0, w, h);
    ctx.globalCompositeOperation = 'source-over';
  };

  // === Основной цикл ===
  const runInference = async () => {
    if (!model || !webcamRef.current?.video || !canvasRef.current) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const loop = async () => {
      const input = preprocess(video);
      const outputs = await model.executeAsync(input);
      tf.dispose(input);

      const mask = await postprocess(outputs, video.videoWidth, video.videoHeight);
      applyBokehEffect(canvas, video, mask);
      tf.dispose([mask, ...outputs]);

      animationId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  useEffect(() => {
    if (model && webcamRef.current?.video) {
      runInference();
    }
    return () => {
      if (animationId.current) cancelAnimationFrame(animationId.current);
    };
  }, [model]);

  if (loading) return <p style={{ color: '#fff' }}>Загрузка YOLO11m-seg...</p>;
  if (error) return <p style={{ color: 'red' }}>Ошибка: {error}</p>;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <Webcam
        ref={webcamRef}
        width={640}
        height={480}
        videoConstraints={{ facingMode: 'user' }}
        style={{ display: 'none' }}
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{
          border: '3px solid #61dafb',
          borderRadius: '12px',
          boxShadow: '0 0 25px rgba(97, 218, 251, 0.4)',
        }}
      />
    </div>
  );
};

export default Yolo11SegDemo;