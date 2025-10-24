// src/Yolo11SegDemo.jsx
import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';

// ВАЖНО: пакет — CJS/UMD. Для ESM-окружения берём конструктор из default.
import * as VSM from 'video-stream-merger';

import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import { loadGraphModel } from '@tensorflow/tfjs-converter';

// ====== настройки ======
const MODEL_PATH = '/yolo11m-seg_web_model/model.json';
const INPUT_SIZE = 640;
const FPS = 25;
const PERSON_CLASS_ID = 0;
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

const VideoStreamMerger = VSM.default || VSM;

export default function Yolo11SegDemo() {
  // источники и вывод
  const webcamRef = useRef(null);     // скрытый <video> react-webcam
  const outVideoRef = useRef(null);   // видимый <video> с результатом merger

  // offscreen-canvas'ы
  const preprocCanvasRef = useRef(document.createElement('canvas')); // ресайз в 640x640
  const maskCanvasRef    = useRef(document.createElement('canvas')); // альфа-маска
  const overlayCanvasRef = useRef(document.createElement('canvas')); // зелёный ∩ маска

  // merger
  const mergerRef = useRef(null);
  const overlayStreamRef = useRef(null);

  // tfjs / цикл
  const rafId = useRef(null);
  const [model, setModel] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [error, setError] = useState(null);

  // ====== загрузка модели ======
  useEffect(() => {
    (async () => {
      try {
        await tf.setBackend('webgl'); await tf.ready();
        const m = await loadGraphModel(MODEL_PATH);
        setModel(m);
      } catch (e) {
        console.error(e);
        setError('Не удалось загрузить TFJS-модель. Проверьте путь/формат.');
      }
    })();
    return () => {
      if (rafId.current) cancelAnimationFrame(rafId.current);
      try { model?.dispose?.(); } catch {}
      try { mergerRef.current?.destroy?.(); } catch {}
      try { overlayStreamRef.current?.getTracks?.().forEach(t => t.stop()); } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ====== утилиты ======
  const disposeAny = useCallback((x) => {
    if (!x) return;
    if (x instanceof tf.Tensor) { x.dispose(); return; }
    if (Array.isArray(x)) { x.forEach(disposeAny); return; }
    if (typeof x === 'object') { Object.values(x).forEach(disposeAny); }
  }, []);

  const nmsCPU = (boxesArr, scoresArr, iouThr = IOU_THRESHOLD) => {
    const order = scoresArr.map((s,i)=>({s,i})).sort((a,b)=>b.s-a.s).map(o=>o.i);
    const keep = new Array(scoresArr.length).fill(true), out=[];
    const iou=(a,b)=>{const[x1,y1,x2,y2]=a,[X1,Y1,X2,Y2]=b,xi1=Math.max(x1,X1),yi1=Math.max(y1,Y1),xi2=Math.min(x2,X2),yi2=Math.min(y2,Y2),inter=Math.max(0,xi2-xi1)*Math.max(0,yi2-yi1),A=(x2-x1)*(y2-y1),B=(X2-X1)*(Y2-Y1);return inter/(A+B-inter+1e-6);};
    for (const i of order){ if(!keep[i]) continue; for (let j=0;j<scoresArr.length;j++){ if(!keep[j]||j===i) continue; if(iou(boxesArr[i],boxesArr[j])>iouThr) keep[j]=false; } out.push(i); }
    return out;
  };

  // ====== препроцесс ======
  const preprocess = useCallback((video) => {
    const c = preprocCanvasRef.current;
    c.width = INPUT_SIZE; c.height = INPUT_SIZE;
    c.getContext('2d').drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    return tf.tidy(() => {
      const u8  = tf.browser.fromPixels(c);    // Uint8 [H,W,3]
      const f32 = tf.cast(u8, 'float32');
      const nor = tf.mul(f32, 1/255);
      return tf.keep(tf.expandDims(nor, 0));   // [1,H,W,3]
    });
  }, []);

  // ====== нормализация выходов ======
  const asMap = (outs) => {
    if (outs instanceof tf.Tensor) return { out0: outs };
    if (Array.isArray(outs)) { const m={}; outs.forEach((t,i)=>m[`out${i}`]=t); return m; }
    return outs;
  };

  const parseOutputs = (map) => {
    // прямой выход маски
    for (const [k,t] of Object.entries(map)) {
      const s = t.shape;
      if (s.length===2 || (s.length===3&&(s[0]===1||s[2]===1)) || (s.length===4&&s[0]===1&&s[3]===1)) {
        return { type:'direct', mask:t };
      }
    }
    // yolo-seg: proto + coeffs (+ boxes/scores/classes)
    let protoKey, coeffKey, boxesKey, scoresKey, classesKey;
    for (const [k,t] of Object.entries(map)) {
      const s=t.shape;
      if (s.length===4){const kdim=Math.max(s[1],s[2],s[3]); if(kdim>=16&&kdim<=512){protoKey=k;break;}}
      else if (s.length===3){const kdim=Math.max(s[0],s[1],s[2]); if(kdim>=16&&kdim<=512){protoKey=k;break;}}
    }
    if (protoKey){
      const ps = map[protoKey].shape;
      const kdim = ps.length===4 ? Math.max(ps[1],ps[2],ps[3]) : Math.max(ps[0],ps[1],ps[2]);
      for (const [k,t] of Object.entries(map)) {
        if (k===protoKey) continue;
        const s=t.shape;
        if ((s.length===3&&s[2]===kdim)||(s.length===2&&s[1]===kdim)) { coeffKey=k; break; }
      }
    }
    for (const [k,t] of Object.entries(map)) {
      const s=t.shape;
      if (!boxesKey && s.length>=2 && s[s.length-1]===4) boxesKey=k;
      else if (!scoresKey && s.length>=1 && s.length<=2 && s[s.length-1]>4) scoresKey=k;
      else if (!classesKey && s.length>=1 && s.length<=2 && s[s.length-1]>4 && k!==scoresKey) classesKey=k;
    }
    return {
      type: protoKey && coeffKey ? 'yolo' : 'none',
      proto: protoKey && map[protoKey],
      coeffs: coeffKey && map[coeffKey],
      boxes: boxesKey && map[boxesKey],
      scores: scoresKey && map[scoresKey],
      classes: classesKey && map[classesKey],
    };
  };

  // ====== сборка бинарной маски [H,W] ======
  const buildBinaryMask = (parsed, W, H) => {
    if (parsed.type === 'direct' && parsed.mask) {
      return tf.tidy(() => {
        let m = parsed.mask;
        if (m.shape.length===4) m = tf.squeeze(m);
        if (m.shape.length===3 && m.shape[2]===1) m = tf.squeeze(m);
        if (m.shape.length===3 && m.shape[0]===1) m = tf.squeeze(m);
        const resized = (m.shape[0]!==H || m.shape[1]!==W)
          ? tf.squeeze(tf.image.resizeBilinear(tf.expandDims(m,-1), [H,W]))
          : m;
        return tf.keep(tf.cast(tf.greater(resized, 0.5), 'float32')); // [H,W]
      });
    }
    if (parsed.type === 'yolo' && parsed.proto && parsed.coeffs) {
      return tf.tidy(() => {
        let p = parsed.proto; // -> [k,h,w]
        if (p.shape.length===4) {
          const kIsLast = p.shape[3] >= Math.max(p.shape[1],p.shape[2]);
          p = kIsLast ? tf.squeeze(tf.transpose(p,[0,3,1,2])) : tf.squeeze(p);
        }
        const pRes = tf.image.resizeBilinear(p, [H, W]);  // [k,H,W]
        const k = pRes.shape[0];
        const proto2D = tf.reshape(pRes, [k, H*W]);       // [k,HW]

        let c = parsed.coeffs;                            // [N,K] | [1,N,K] | [K]
        if (c.shape.length===3) c = tf.squeeze(c);
        if (c.shape.length===1) c = tf.expandDims(c, 0);

        if (parsed.scores) {
          const scores = tf.squeeze(parsed.scores);
          const keep = tf.squeeze(tf.where(tf.greater(scores, CONFIDENCE_THRESHOLD)));
          if (keep.size) c = tf.gather(c, keep);
        }
        if (parsed.classes) {
          const classes = tf.squeeze(parsed.classes);
          const keep = tf.squeeze(tf.where(tf.equal(classes, PERSON_CLASS_ID)));
          if (keep.size) c = tf.gather(c, keep);
        }
        if (c.size===0) return tf.keep(tf.zeros([H,W]));

        const logits = tf.matMul(c, proto2D);             // [n,HW]
        const prob = tf.sigmoid(logits);
        const masks = tf.reshape(prob, [-1, H, W]);       // [n,H,W]
        return tf.keep(tf.cast(tf.greater(tf.max(masks,0), 0.5), 'float32')); // [H,W]
      });
    }
    return tf.keep(tf.zeros([H, W]));
  };

  // ====== построить overlayCanvas по маске ======
  const paintOverlay = async (overlayCanvas, mask, w, h) => {
    if (overlayCanvas.width !== w || overlayCanvas.height !== h) {
      overlayCanvas.width = w; overlayCanvas.height = h;
    }

    // maskCanvas: альфа=1 внутри маски
    const maskCanvas = maskCanvasRef.current;
    if (maskCanvas.width !== w || maskCanvas.height !== h) {
      maskCanvas.width = w; maskCanvas.height = h;
    }
    const mctx = maskCanvas.getContext('2d');
    const data = await mask.data(); // [H*W]
    const img = mctx.createImageData(w, h);
    for (let i=0;i<w*h;i++){
      const on = data[i] > 0.5; const p = i*4;
      img.data[p+0]=255; img.data[p+1]=255; img.data[p+2]=255;
      img.data[p+3]= on ? 255 : 0;
    }
    mctx.putImageData(img, 0, 0);

    // overlayCanvas: зелёный ∩ маска
    const octx = overlayCanvas.getContext('2d');
    octx.clearRect(0, 0, w, h);
    octx.globalCompositeOperation = 'source-over';
    octx.fillStyle = 'rgba(0,255,0,0.5)';
    octx.fillRect(0, 0, w, h);
    octx.globalCompositeOperation = 'destination-in';
    octx.drawImage(maskCanvas, 0, 0);
  };

  // ====== инициализация VideoStreamMerger (камера + overlay) ======
  const ensureMerger = useCallback((video) => {
    if (mergerRef.current) return mergerRef.current;

    const W = video.videoWidth || 640;
    const H = video.videoHeight || 480;

    const overlayCanvas = overlayCanvasRef.current;
    overlayStreamRef.current = overlayCanvas.captureStream(FPS);

    const merger = new VideoStreamMerger({ width: W, height: H, fps: FPS });

    // нижний слой — камера (через addMediaElement; библиотека сама возьмёт медиа-поток элемента)
    merger.addMediaElement('camera', video, {
      x: 0, y: 0, width: W, height: H, mute: true,
      draw: (ctx, frame, done) => { ctx.drawImage(frame, 0, 0, W, H); done(); }
    });

    // верхний слой — наш overlayCanvas как MediaStream
    merger.addStream(overlayStreamRef.current, {
      x: 0, y: 0, width: W, height: H, index: 1, mute: true,
      draw: (ctx, _frame, done) => { ctx.drawImage(overlayCanvas, 0, 0, W, H); done(); }
    });

    merger.start(); // создаёт merger.result
    mergerRef.current = merger;

    // подключаем в видимый <video>
    if (outVideoRef.current) {
      outVideoRef.current.srcObject = merger.result;
      outVideoRef.current.muted = true;
      outVideoRef.current.play().catch(()=>{});
    }
    return merger;
  }, []);

  // ====== покадровая обработка ======
  const step = useCallback(async () => {
    const video = webcamRef.current?.video;
    if (!model || !video || video.readyState < 2) {
      rafId.current = requestAnimationFrame(step);
      return;
    }

    // инициализируем merger, когда появились размеры видео
    ensureMerger(video);
    const W = video.videoWidth || 640;
    const H = video.videoHeight || 480;

    tf.engine().startScope();
    try {
      const x = preprocess(video);
      const outs = model.outputNodes?.length
        ? model.execute(x, model.outputNodes)
        : model.execute(x);

      const parsed = parseOutputs(asMap(outs));
      const mask = buildBinaryMask(parsed, W, H);

      // обновляем overlayCanvas — merger подхватит новый кадр
      await paintOverlay(overlayCanvasRef.current, mask, W, H);
    } catch (e) {
      // console.warn('inference error', e);
    } finally {
      tf.engine().endScope();
      rafId.current = requestAnimationFrame(step);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, preprocess, ensureMerger]);

  useEffect(() => {
    if (!model || !cameraReady) return;
    let active = true;
    const run = () => {
      if (!active) return;
      if (rafId.current) cancelAnimationFrame(rafId.current);
      rafId.current = requestAnimationFrame(step);
    };
    run();
    return () => { active = false; if (rafId.current) cancelAnimationFrame(rafId.current); };
  }, [model, cameraReady, step]);

  // ====== UI ======
  if (error) return <p style={{ color:'red' }}>{error}</p>;
  return (
    <div style={{ position:'relative', display:'inline-block' }}>
      {/* источник: вебка (скрыта визуально, но обязана играть) */}
      <Webcam
        ref={webcamRef}
        audio={false}
        muted
        playsInline
        onUserMedia={() => setCameraReady(true)}
        videoConstraints={{ facingMode: 'user' }}
        style={{ position:'absolute', opacity:0, width:1, height:1, pointerEvents:'none' }}
      />
      {/* результат мерджа */}
      <video
        ref={outVideoRef}
        autoPlay
        playsInline
        muted
        style={{
          display:'block',
          border:'3px solid #61dafb',
          borderRadius:12,
          boxShadow:'0 0 25px rgba(97,218,251,.4)',
          width:'640px', height:'480px', background:'#000'
        }}
      />
    </div>
  );
}
