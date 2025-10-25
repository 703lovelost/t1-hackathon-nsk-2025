import { useEffect } from 'react';

const BadgeOverlay = ({ settings }) => {
  useEffect(() => {
    const overlay = document.getElementById('smartBadgeOverlay');
    const badge = document.querySelector('.smart-badge');
    const logoContainer = document.getElementById('badge-logo-container');
    const logoImg = document.getElementById('badge-logo-img');
    const mainInfo = document.querySelector('.badge-main');
    const nameText = document.getElementById('badge-name-text');
    const companyText = document.getElementById('badge-company-text');
    const details = document.querySelector('.badge-details');
    const itemPosition = document.getElementById('badge-item-position');
    const positionText = document.getElementById('badge-position-text');
    const itemDepartment = document.getElementById('badge-item-department');
    const departmentText = document.getElementById('badge-department-text');
    const itemLocation = document.getElementById('badge-item-location');
    const locationText = document.getElementById('badge-location-text');
    const itemTelegram = document.getElementById('badge-item-telegram');
    const telegramLink = document.getElementById('badge-telegram-link');
    const itemEmail = document.getElementById('badge-item-email');
    const emailLink = document.getElementById('badge-email-link');
    const itemSlogan = document.getElementById('badge-item-slogan');
    const sloganText = document.getElementById('badge-slogan-text');

    if (!overlay || !badge) return;

    overlay.style.display = settings.show ? 'block' : 'none';
    if (!settings.show) return;

    overlay.className = `badge-overlay-container ${settings.badgePosition}`;
    badge.style.backgroundColor = settings.colorPrimary;
    badge.style.outlineColor = settings.colorSecondary;

    const logoSrc = settings.logoType === 'upload' ? settings.logoDataUrl : settings.logoUrl;
    if (logoSrc) {
      logoImg.src = logoSrc;
      logoContainer.style.display = 'block';
    } else {
      logoContainer.style.display = 'none';
      logoImg.src = '';
    }

    nameText.textContent = settings.name;
    nameText.style.display = settings.showName && settings.name ? 'block' : 'none';
    companyText.textContent = settings.company;
    companyText.style.display = settings.showCompany && settings.company ? 'block' : 'none';
    mainInfo.style.display =
      (settings.showName && settings.name) || (settings.showCompany && settings.company)
        ? 'block'
        : 'none';

    positionText.textContent = settings.jobPosition;
    itemPosition.style.display = settings.showPosition && settings.jobPosition ? 'flex' : 'none';
    departmentText.textContent = settings.department;
    itemDepartment.style.display = settings.showDepartment && settings.department ? 'flex' : 'none';
    locationText.textContent = settings.location;
    itemLocation.style.display = settings.showLocation && settings.location ? 'flex' : 'none';
    sloganText.textContent = settings.slogan;
    itemSlogan.style.display = settings.showSlogan && settings.slogan ? 'flex' : 'none';

    if (settings.showTelegram && settings.telegram) {
      const username = settings.telegram.replace(/^@/, '');
      telegramLink.textContent = settings.telegram;
      telegramLink.href = `https://t.me/${username}`;
      telegramLink.style.color = settings.colorSecondary;
      itemTelegram.style.display = 'flex';
      const telegramIcon = itemTelegram.querySelector('i');
      if (telegramIcon) {
        telegramIcon.style.color = settings.colorSecondary;
        telegramIcon.style.opacity = '1';
      }
    } else {
      itemTelegram.style.display = 'none';
    }

    if (settings.showEmail && settings.email) {
      emailLink.textContent = settings.email;
      emailLink.href = `mailto:${settings.email}`;
      itemEmail.style.display = 'flex';
    } else {
      itemEmail.style.display = 'none';
    }

    const hasDetails =
      (settings.showPosition && settings.jobPosition) ||
      (settings.showDepartment && settings.department) ||
      (settings.showLocation && settings.location) ||
      (settings.showTelegram && settings.telegram) ||
      (settings.showEmail && settings.email) ||
      (settings.showSlogan && settings.slogan);
    details.style.display = hasDetails ? 'block' : 'none';
  }, [settings]);

  return (
    <div id="smartBadgeOverlay" className="badge-overlay-container">
      <div className="smart-badge">
        <div id="badge-logo-container" className="badge-logo">
          <img id="badge-logo-img" src="" alt="Logo" />
        </div>
        <div className="badge-content">
          <div className="badge-main">
            <h3 id="badge-name-text"></h3>
            <p id="badge-company-text"></p>
          </div>
          <div className="badge-details">
            <div id="badge-item-position" className="detail-item">
              <i className="fas fa-briefcase"></i>
              <span id="badge-position-text"></span>
            </div>
            <div id="badge-item-department" className="detail-item">
              <i className="fas fa-sitemap"></i>
              <span id="badge-department-text"></span>
            </div>
            <div id="badge-item-location" className="detail-item">
              <i className="fas fa-map-marker-alt"></i>
              <span id="badge-location-text"></span>
            </div>
            <div id="badge-item-telegram" className="detail-item">
              <i className="fab fa-telegram-plane"></i>
              <a id="badge-telegram-link" href="#" target="_blank" rel="noopener noreferrer"></a>
            </div>
            <div id="badge-item-email" className="detail-item">
              <i className="fas fa-at"></i>
              <a id="badge-email-link" href="#"></a>
            </div>
            <div id="badge-item-slogan" className="detail-item">
              <i className="fas fa-bullhorn"></i>
              <span id="badge-slogan-text" className="slogan-text"></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BadgeOverlay;