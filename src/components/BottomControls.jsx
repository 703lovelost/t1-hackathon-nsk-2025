const BottomControls = ({ isCameraOn, toggleCamera, openSettings }) => {
  return (
    <div className="bottom-controls">
      <div className="center-controls">
        <div className="settings-wrapper">
          <button id="settingsBtn" aria-label="Настройки" onClick={openSettings}>
            <img src="/icons/free-icon-gear-1242494.png" alt="Настройки" className="settings-icon" />
          </button>
          <span className="settings-tooltip">Настройки</span>
        </div>
        <button
          id="toggleCamBtn"
          className={`toggle-cam-btn ${isCameraOn ? 'is-active' : ''}`}
          aria-label="Включить/выключить камеру"
          onClick={toggleCamera}
        >
          <img src="/icons/free-icon-video-call-6016257.png" alt="Камера" className="cam-icon" />
          <span className="cam-cross-out"></span>
        </button>
      </div>
    </div>
  );
};

export default BottomControls;