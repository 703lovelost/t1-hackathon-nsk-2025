import './styles/styles.css';
import VideoConference from './components/VideoConference';
import { useState } from 'react';

function App() {
  const [useYolo, setUseYolo] = useState(false);

  return (
    <div className="App">
      <header className="container">Camera test.</header>
      <main className="container">
        <VideoConference useYolo={useYolo} setUseYolo={setUseYolo} />
      </main>
    </div>
  );
}

export default App;