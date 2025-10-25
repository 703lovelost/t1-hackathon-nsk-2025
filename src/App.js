// App.js
import './App.css';
import BlurryCamDemo from './BlurryCamDemo';
import { useState } from 'react';

function App() {
  const [useYolo, setUseYolo] = useState(false);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Тест маскирования</h1>
        {/* <button
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
        </button> */}

        <BlurryCamDemo />
      </header>
    </div>
  );
}

export default App;