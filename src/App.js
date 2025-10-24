// App.js
import './App.css';
import BlurryCamDemo from './BlurryCamDemo';
import Yolo11SegDemo from './Yolo11SegDemo';
import { useState } from 'react';

function App() {
  const [useYolo, setUseYolo] = useState(false);

  return (
    <div className="App">
      <header className="App-header">
        <h1>BlurryCam Demo</h1>
        <button
          onClick={() => setUseYolo(!useYolo)}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            marginBottom: '20px',
            cursor: 'pointer',
            background: useYolo ? '#61dafb' : '#282c34',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
          }}
        >
          {useYolo ? 'Переключиться на MediaPipe' : 'Использовать YOLO11m-seg'}
        </button>

        {useYolo ? <Yolo11SegDemo /> : <BlurryCamDemo />}
      </header>
    </div>
  );
}

export default App;