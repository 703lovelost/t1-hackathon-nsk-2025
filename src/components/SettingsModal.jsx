import { useState, useEffect } from 'react';

const SettingsModal = ({ isOpen, closeModal, cameras, cameraId, setCameraId, badgeSettings, setBadgeSettings }) => {
  const [activeTab, setActiveTab] = useState('video-tab');
  const [logoWarning, setLogoWarning] = useState('');

  const handleBadgeChange = (e, field) => {
    const newSettings = { ...badgeSettings };
    if (field === 'show') {
      newSettings.show = e.target.checked;
    } else if (field.startsWith('show')) {
      newSettings[field] = e.target.checked;
    } else if (field === 'badgePosition' || field === 'logoType') {
      newSettings[field] = e.target.value;
    } else if (field === 'logoUpload') {
      const file = e.target.files[0];
      if (!file) return;
      if (!['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(file.type)) {
        setLogoWarning('Неверный тип файла (PNG, JPG, GIF, WEBP).');
        return;
      }
      if (file.size > 1 * 1024 * 1024) {
        setLogoWarning('Файл слишком большой (макс. 1MB).');
        return;
      }
      const reader = new FileReader();
      reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
          let warning = '';
          if (img.naturalWidth < 48 || img.naturalHeight < 48) {
            warning = 'Лого маловато (реком. 48x48+). ';
          }
          if (img.naturalWidth !== img.naturalHeight) {
            warning += 'Лого не квадратное.';
          }
          setLogoWarning(warning);
          newSettings.logoDataUrl = ev.target.result;
          setBadgeSettings(newSettings);
        };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    } else {
      newSettings[field] = e.target.value;
    }
    setBadgeSettings(newSettings);
    localStorage.setItem('badgeSettings', JSON.stringify(newSettings));
  };

  useEffect(() => {
    const storedSettings = localStorage.getItem('badgeSettings');
    if (storedSettings) {
      setBadgeSettings(JSON.parse(storedSettings));
    }
  }, []);

  return (
    <div id="settingsModal" className="modal" style={{ display: isOpen ? 'flex' : 'none' }}>
      <div className="modal-content">
        <span id="closeSettingsBtn" title="Закрыть" onClick={closeModal}>
          &times;
        </span>
        <h2>Настройки</h2>
        <div className="modal-tabs">
          <button
            className={`tab-link ${activeTab === 'video-tab' ? 'active' : ''}`}
            onClick={() => setActiveTab('video-tab')}
            data-tab="video-tab"
          >
            Видео
          </button>
          <button
            className={`tab-link ${activeTab === 'badge-tab' ? 'active' : ''}`}
            onClick={() => setActiveTab('badge-tab')}
            data-tab="badge-tab"
          >
            Бейдж
          </button>
          <button
            className={`tab-link ${activeTab === 'general-tab' ? 'active' : ''}`}
            onClick={() => setActiveTab('general-tab')}
            data-tab="general-tab"
          >
            Общие
          </button>
        </div>
        <div className="modal-tab-content">
          <div id="video-tab" className={`tab-pane ${activeTab === 'video-tab' ? 'active' : ''}`}>
            <div className="setting-row">
              <label htmlFor="cameraSelect">Камера:</label>
              <select
                id="cameraSelect"
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                disabled={cameras.length <= 1}
              >
                {cameras.map((cam, i) => (
                  <option key={cam.deviceId} value={cam.deviceId}>
                    {cam.label || `Камера ${i + 1}`}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div id="badge-tab" className={`tab-pane ${activeTab === 'badge-tab' ? 'active' : ''}`}>
            <div className="setting-row">
              <label htmlFor="badge-toggle-show">Показать бейдж</label>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  id="badge-toggle-show"
                  checked={badgeSettings.show}
                  onChange={(e) => handleBadgeChange(e, 'show')}
                />
                <span className="slider"></span>
              </label>
            </div>
            <div id="badge-settings-group" className={badgeSettings.show ? '' : 'hidden'}>
              <hr className="divider" />
              <div className="setting-group">
                <label className="group-label">Позиция на экране</label>
                <div className="radio-group">
                  {[
                    { value: 'pos-top-left', label: '↖' },
                    { value: 'pos-top-center', label: '↑' },
                    { value: 'pos-top-right', label: '↗' },
                    { value: 'pos-bottom-left', label: '↙' },
                    { value: 'pos-bottom-center', label: '↓' },
                    { value: 'pos-bottom-right', label: '↘' },
                  ].map((pos) => (
                    <div key={pos.value}>
                      <input
                        type="radio"
                        id={pos.value}
                        name="badge-position"
                        value={pos.value}
                        checked={badgeSettings.badgePosition === pos.value}
                        onChange={(e) => handleBadgeChange(e, 'badgePosition')}
                      />
                      <label htmlFor={pos.value}>{pos.label}</label>
                    </div>
                  ))}
                </div>
              </div>
              <div className="setting-group">
                <label className="group-label">Логотип</label>
                <div className="radio-group logo-type">
                  {[
                    { value: 'url', label: 'URL' },
                    { value: 'upload', label: 'Загрузить' },
                  ].map((type) => (
                    <div key={type.value}>
                      <input
                        type="radio"
                        id={`logo-type-${type.value}`}
                        name="badge-logo-type"
                        value={type.value}
                        checked={badgeSettings.logoType === type.value}
                        onChange={(e) => handleBadgeChange(e, 'logoType')}
                      />
                      <label htmlFor={`logo-type-${type.value}`}>{type.label}</label>
                    </div>
                  ))}
                </div>
                <input
                  type="text"
                  id="badge-logo-url"
                  className={`setting-input ${badgeSettings.logoType === 'url' ? '' : 'hidden'}`}
                  placeholder="https://example.com/logo.png"
                  value={badgeSettings.logoUrl}
                  onChange={(e) => handleBadgeChange(e, 'logoUrl')}
                />
                <input
                  type="file"
                  id="badge-logo-upload"
                  className={`setting-input ${badgeSettings.logoType === 'upload' ? '' : 'hidden'}`}
                  accept="image/png, image/jpeg, image/gif, image/webp"
                  onChange={(e) => handleBadgeChange(e, 'logoUpload')}
                />
                <small id="badge-logo-warning" className="warning-text">
                  {logoWarning}
                </small>
              </div>
              <div className="setting-group">
                <label className="group-label">Цвета</label>
                <div className="setting-row-toggle colors">
                  <label htmlFor="badge-field-color-primary">Фон:</label>
                  <input
                    type="text"
                    id="badge-field-color-primary"
                    className="setting-input color-text-input"
                    placeholder="#0052CC"
                    value={badgeSettings.colorPrimary}
                    onChange={(e) => handleBadgeChange(e, 'colorPrimary')}
                  />
                  <input
                    type="color"
                    id="badge-picker-color-primary"
                    className="setting-color-picker"
                    value={badgeSettings.colorPrimary}
                    onChange={(e) => handleBadgeChange(e, 'colorPrimary')}
                  />
                </div>
                <div className="setting-row-toggle colors">
                  <label htmlFor="badge-field-color-secondary">Акцент:</label>
                  <input
                    type="text"
                    id="badge-field-color-secondary"
                    className="setting-input color-text-input"
                    placeholder="#00B8D9"
                    value={badgeSettings.colorSecondary}
                    onChange={(e) => handleBadgeChange(e, 'colorSecondary')}
                  />
                  <input
                    type="color"
                    id="badge-picker-color-secondary"
                    className="setting-color-picker"
                    value={badgeSettings.colorSecondary}
                    onChange={(e) => handleBadgeChange(e, 'colorSecondary')}
                  />
                </div>
              </div>
              <div className="setting-group">
                <label className="group-label">Поля (введите данные)</label>
                {[
                  { id: 'name', placeholder: 'Иванов Сергей Петрович', toggle: 'showName' },
                  { id: 'company', placeholder: 'ООО «Рога и Копыта»', toggle: 'showCompany' },
                  { id: 'position', placeholder: 'Ведущий инженер', toggle: 'showPosition' },
                  { id: 'department', placeholder: 'Департамент КЗ', toggle: 'showDepartment' },
                  { id: 'location', placeholder: 'Новосибирск, Технопарк', toggle: 'showLocation' },
                  { id: 'telegram', placeholder: '@username', toggle: 'showTelegram' },
                  { id: 'email', placeholder: 'user@example.com', toggle: 'showEmail' },
                  { id: 'slogan', placeholder: 'Инновации в каждый кадр', toggle: 'showSlogan' },
                ].map((field) => (
                  <div className="setting-row-toggle" key={field.id}>
                    <input
                      type="text"
                      id={`badge-field-${field.id}`}
                      className="setting-input"
                      placeholder={field.placeholder}
                      value={badgeSettings[field.id]}
                      onChange={(e) => handleBadgeChange(e, field.id)}
                    />
                    <label className="toggle-switch small">
                      <input
                        type="checkbox"
                        id={`badge-toggle-${field.id}`}
                        checked={badgeSettings[field.toggle]}
                        onChange={(e) => handleBadgeChange(e, field.toggle)}
                      />
                      <span className="slider"></span>
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div id="general-tab" className={`tab-pane ${activeTab === 'general-tab' ? 'active' : ''}`}>
            <p>Другие настройки.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;