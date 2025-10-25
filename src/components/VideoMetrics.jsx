const VideoMetrics = ({ metrics }) => {
  return (
    <div className="video-metrics">
      <span id="status" className={metrics.statusClass}>
        Статус: {metrics.status}
      </span>
      <span className="metric" id="fps">
        FPS: {metrics.fps} fps
      </span>
      <span className="metric" id="fpsAvg">
        FPSAvg: {metrics.fpsAvg} fps
      </span>
      <span className="metric" id="cpuNow">
        CPU: {metrics.cpu}%
      </span>
      <span className="metric" id="cpuAvg">
        CPUAvg: {metrics.cpuAvg}%
      </span>
      <span className="metric" id="gpuNow">
        GPU: {metrics.gpu}%
      </span>
      <span className="metric" id="gpuAvg">
        GPUAvg: {metrics.gpuAvg}%
      </span>
    </div>
  );
};

export default VideoMetrics;