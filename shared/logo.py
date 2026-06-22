"""App logo and icon for PyKaboo suite.

The logo is a contemporary SVG: a stylised mouse silhouette with a
neural-network motif inside, set in a rounded-square badge.

Usage
-----
    from shared.logo import app_icon, logo_pixmap
    app.setWindowIcon(app_icon())
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# ── SVG definition ────────────────────────────────────────────────────────────

_LOGO_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e1e2e"/>
      <stop offset="100%" stop-color="#313244"/>
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#cba6f7"/>
      <stop offset="100%" stop-color="#89b4fa"/>
    </linearGradient>
  </defs>

  <!-- Badge background -->
  <rect width="256" height="256" rx="52" ry="52" fill="url(#bg)"/>

  <!-- Mouse body — smooth organic shape -->
  <ellipse cx="128" cy="148" rx="62" ry="46"
           fill="none" stroke="url(#accent)" stroke-width="5"/>

  <!-- Mouse head -->
  <ellipse cx="128" cy="100" rx="32" ry="28"
           fill="none" stroke="url(#accent)" stroke-width="5"/>

  <!-- Ears -->
  <ellipse cx="98" cy="76" rx="16" ry="12"
           fill="none" stroke="url(#accent)" stroke-width="4"/>
  <ellipse cx="158" cy="76" rx="16" ry="12"
           fill="none" stroke="url(#accent)" stroke-width="4"/>

  <!-- Eye -->
  <circle cx="120" cy="96" r="4" fill="url(#accent)"/>
  <circle cx="136" cy="96" r="4" fill="url(#accent)"/>

  <!-- Neural-net dots -->
  <circle cx="108" cy="148" r="5" fill="#cba6f7" opacity="0.8"/>
  <circle cx="128" cy="138" r="5" fill="#89b4fa" opacity="0.8"/>
  <circle cx="148" cy="148" r="5" fill="#cba6f7" opacity="0.8"/>
  <circle cx="118" cy="162" r="5" fill="#a6e3a1" opacity="0.8"/>
  <circle cx="138" cy="162" r="5" fill="#a6e3a1" opacity="0.8"/>

  <!-- Connections -->
  <line x1="108" y1="148" x2="128" y2="138" stroke="#cdd6f4" stroke-width="1.5" opacity="0.4"/>
  <line x1="128" y1="138" x2="148" y2="148" stroke="#cdd6f4" stroke-width="1.5" opacity="0.4"/>
  <line x1="108" y1="148" x2="118" y2="162" stroke="#cdd6f4" stroke-width="1.5" opacity="0.4"/>
  <line x1="148" y1="148" x2="138" y2="162" stroke="#cdd6f4" stroke-width="1.5" opacity="0.4"/>
  <line x1="128" y1="138" x2="118" y2="162" stroke="#cdd6f4" stroke-width="1.5" opacity="0.2"/>
  <line x1="128" y1="138" x2="138" y2="162" stroke="#cdd6f4" stroke-width="1.5" opacity="0.2"/>

  <!-- Tail -->
  <path d="M 190 155 Q 220 160 215 185 Q 210 205 195 200"
        fill="none" stroke="url(#accent)" stroke-width="4"
        stroke-linecap="round"/>
</svg>
"""

# ── Icon-sized (no badge, flat look) ─────────────────────────────────────────

_ICON_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
  <defs>
    <linearGradient id="a" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#cba6f7"/>
      <stop offset="100%" stop-color="#89b4fa"/>
    </linearGradient>
  </defs>
  <rect width="48" height="48" rx="10" ry="10" fill="#1e1e2e"/>
  <!-- head -->
  <ellipse cx="24" cy="19" rx="11" ry="10" fill="none" stroke="url(#a)" stroke-width="2"/>
  <!-- ears -->
  <ellipse cx="14" cy="13" rx="5" ry="4" fill="none" stroke="url(#a)" stroke-width="1.8"/>
  <ellipse cx="34" cy="13" rx="5" ry="4" fill="none" stroke="url(#a)" stroke-width="1.8"/>
  <!-- eyes -->
  <circle cx="20" cy="18" r="1.5" fill="#cba6f7"/>
  <circle cx="28" cy="18" r="1.5" fill="#89b4fa"/>
  <!-- body -->
  <ellipse cx="24" cy="34" rx="13" ry="9" fill="none" stroke="url(#a)" stroke-width="2"/>
  <!-- neural dots -->
  <circle cx="19" cy="34" r="2" fill="#cba6f7" opacity="0.9"/>
  <circle cx="24" cy="31" r="2" fill="#89b4fa" opacity="0.9"/>
  <circle cx="29" cy="34" r="2" fill="#a6e3a1" opacity="0.9"/>
  <!-- tail -->
  <path d="M37 37 Q42 39 41 43" fill="none" stroke="url(#a)" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""


def _render_svg(svg_str: str, size: int) -> QPixmap:
    data = QByteArray(svg_str.encode())
    renderer = QSvgRenderer(data)
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    renderer.render(painter)
    painter.end()
    return QPixmap.fromImage(img)


def logo_pixmap(size: int = 256) -> QPixmap:
    """Return the full logo as a QPixmap at *size* × *size*."""
    return _render_svg(_LOGO_SVG, size)


def app_icon() -> QIcon:
    """Return QIcon with multiple sizes for the taskbar / window."""
    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        svg = _ICON_SVG if sz <= 64 else _LOGO_SVG
        icon.addPixmap(_render_svg(svg, sz))
    return icon
