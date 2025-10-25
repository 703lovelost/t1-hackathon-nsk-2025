import { useRef } from 'react';
import BlurryCamDemo from './BlurryCamDemo';

const VideoPane = ({ mediaStream, useYolo }) => {
  const videoRef = useRef(null);

  return (
    <div
      style={{
        position: 'relative',
        width: '640px',
        height: '480px',
        overflow: 'hidden',
      }}
    >
      <BlurryCamDemo videoRef={videoRef} mediaStream={mediaStream} useYolo={useYolo} />
    </div>
  );
};

export default VideoPane;