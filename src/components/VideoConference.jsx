import { useState, useEffect, useRef } from 'react';
import VideoPane from './VideoPane';
import VideoMetrics from './VideoMetrics';
import BottomControls from './BottomControls';
import BadgeOverlay from './BadgeOverlay';
import SettingsModal from './SettingsModal';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const VideoConference = ({ useYolo, setUseYolo }) => {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [cameraId, setCameraId] = useState('');
  const [cameras, setCameras] = useState([]);
  const [metrics, setMetrics] = useState({
    status: 'готов',
    fps: '-',
    fpsAvg: '-',
    cpu: '-',
    cpuAvg: '-',
    gpu: '-',
    gpuAvg: '-',
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [badgeSettings, setBadgeSettings] = useState({
    show: false,
    badgePosition: 'pos-bottom-left',
    logoType: 'url',
    logoUrl: '',
    logoDataUrl: '',
    colorPrimary: '#0052CC',
    colorSecondary: '#00B8D9',
    name: 'Иванов Сергей',
    company: 'ООО «Рога и Копыта»',
    jobPosition: '',
    department: '',
    location: '',
    telegram: '',
    email: '',
    slogan: '',
    showName: true,
    showCompany: true,
    showPosition: false,
    showDepartment: false,
    showLocation: false,
    showTelegram: false,
    showEmail: false,
    showSlogan: false,
  });

  const mediaStreamRef = useRef(null);
  const rafIdRef = useRef(null);
  const lastTsRef = useRef(0);
  const fpsSamplesRef = useRef([]);
  const cpuSamplesRef = useRef([]);
  const gpuSamplesRef = useRef([]);
  const frameIdxRef = useRef(0);
  const MAX_SAMPLES = 120;

  const setStatus = (text, cls = '') => {
    setMetrics((prev) => ({ ...prev, status: text, statusClass: cls }));
  };

  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN);
  const fmtNum = (n, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : '-');
  const pushSample = (arr, v) => {
    arr.push(v);
    if (arr.length > MAX_SAMPLES) arr.shift();
  };

  const pickBestBackend = async () => {
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      return tf.getBackend();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
      return 'cpu';
    }
  };

  const listCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === 'videoinput');
      setCameras(cams);
      if (cams.length > 0 && !cameraId) setCameraId(cams[0].deviceId);
    } catch (e) {
      console.error('Ошибка при получении списка камер:', e);
      setStatus('ошибка списка камер', 'warn');
    }
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('getUserMedia не поддерживается', 'warn');
      return;
    }

    setStatus('запрашиваю доступ к камере…');
    const videoConstraints = { width: { ideal: 640 }, height: { ideal: 480 } }; // Фиксированные размеры
    const videoBase = cameraId ? { deviceId: { exact: cameraId } } : { facingMode: 'user' };
    const constraints = { audio: false, video: { ...videoBase, ...videoConstraints } };

    try {
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia(constraints);
      setIsCameraOn(true);
      await listCameras();
      const backend = await pickBestBackend();
      console.log('TF.js backend:', backend);
      setStatus('камера запущена', 'ok');
      lastTsRef.current = performance.now();
      fpsSamplesRef.current = [];
      cpuSamplesRef.current = [];
      gpuSamplesRef.current = [];
      frameIdxRef.current = 0;
      startLoop();
    } catch (e) {
      console.error('Ошибка доступа к камере:', e);
      setStatus(`ошибка: ${e?.name || e?.message || e}`, 'warn');
      stopCamera();
    }
  };

  const stopCamera = () => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    setIsCameraOn(false);
    setStatus('остановлено');
    setMetrics((prev) => ({
      ...prev,
      fps: '-',
      fpsAvg: '-',
      cpu: '-',
      cpuAvg: '-',
      gpu: '-',
      gpuAvg: '-',
    }));
  };

  const gpuProbe = async (dtMs, video) => {
    if (!video || video.paused || video.ended) return null;
    const b = tf.getBackend();
    if (!['webgl'].includes(b)) return null;
    try {
      const res = await tf.time(() =>
        tf.tidy(() => {
          let t = tf.browser.fromPixels(video);
          t = tf.image.resizeBilinear(t, [160, 90], true).toFloat().mul(1 / 255);
          return t.mean();
        }),
      );
      const gpuMs = res.kernelMs ?? res.wallMs ?? 0;
      const gpuUtil = clamp01(gpuMs / Math.max(1, dtMs)) * 100;
      return { gpuUtil };
    } catch {
      return null;
    }
  };

  const startLoop = () => {
    const render = async () => {
      if (!isCameraOn) return;
      const video = document.querySelector('video');
      if (video && video.readyState >= 2) {
        const frameStart = performance.now();
        let gpuUtilNow = null;
        const now = performance.now();
        const dtMs = now - lastTsRef.current;
        if (frameIdxRef.current % 15 === 0) {
          const res = await gpuProbe(dtMs, video);
          if (res) {
            gpuUtilNow = res.gpuUtil;
            pushSample(gpuSamplesRef.current, gpuUtilNow);
          }
        }
        const afterDraw = performance.now();
        const dt = afterDraw - lastTsRef.current;
        const busy = afterDraw - frameStart;
        const fps = 1000 / Math.max(1, dt);
        const cpu = clamp01(busy / Math.max(1, dt)) * 100;
        pushSample(fpsSamplesRef.current, fps);
        pushSample(cpuSamplesRef.current, cpu);
        setMetrics({
          status: metrics.status,
          statusClass: metrics.statusClass,
          fps: fmtNum(fps),
          fpsAvg: fmtNum(avg(fpsSamplesRef.current)),
          cpu: fmtNum(cpu),
          cpuAvg: fmtNum(avg(cpuSamplesRef.current)),
          gpu: gpuUtilNow != null ? fmtNum(gpuUtilNow) : '-',
          gpuAvg: Number.isFinite(avg(gpuSamplesRef.current))
            ? fmtNum(avg(gpuSamplesRef.current))
            : '-',
        });
        lastTsRef.current = afterDraw;
        frameIdxRef.current++;
      }
      if (isCameraOn) {
        rafIdRef.current = requestAnimationFrame(render);
      }
    };
    render();
  };

  useEffect(() => {
    listCameras();
    return () => stopCamera();
  }, []);

  useEffect(() => {
    if (isCameraOn) {
      stopCamera();
      startCamera();
    }
  }, [cameraId]);

  return (
    <section className="video-conference-section">
      <div
        className="main-video-pane"
        style={{ width: '640px', height: '480px', maxWidth: '640px', maxHeight: '480px' }}
      >
        <VideoPane mediaStream={mediaStreamRef.current} useYolo={useYolo} />
        <BadgeOverlay settings={badgeSettings} />
        <VideoMetrics metrics={metrics} />
        <BottomControls
          isCameraOn={isCameraOn}
          toggleCamera={() => (isCameraOn ? stopCamera() : startCamera())}
          openSettings={() => setIsSettingsOpen(true)}
        />
        <SettingsModal
          isOpen={isSettingsOpen}
          closeModal={() => setIsSettingsOpen(false)}
          cameras={cameras}
          cameraId={cameraId}
          setCameraId={setCameraId}
          badgeSettings={badgeSettings}
          setBadgeSettings={setBadgeSettings}
        />
      </div>
    </section>
  );
};

export default VideoConference;