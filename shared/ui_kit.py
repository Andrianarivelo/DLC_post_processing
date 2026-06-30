"""Premium UI kit — design tokens, global stylesheet, and reusable widgets.

This module is the single source of visual truth for the DLC Post-Processing
studio. It centralises the colour palette, the global Qt stylesheet, and a
handful of polished, reusable widgets (cards, section headers, toggle chips,
icon buttons) so every panel shares one cohesive, modern look.

Design language
---------------
- A deep, low-glare dark canvas with layered surfaces (app < surface < card).
- One violet accent for primary actions, with teal / amber / rose accents for
  secondary states (these echo the per-animal overlay colours).
- Generous spacing, a clear type hierarchy, soft 10 px radii, and subtle
  borders instead of heavy boxes.

Usage
-----
    from shared.ui_kit import COLORS, build_app_qss, Card, section_title

    app.setStyleSheet(build_app_qss())
    card = Card("Calibration", "Map pixels to centimetres")
    card.body.addWidget(some_widget)
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ── Design tokens ─────────────────────────────────────────────────────────────
#
# A single dictionary of named colours. Reference these in code (e.g. when
# building pyqtgraph plots or OpenCV overlays) so the whole app stays in sync.

COLORS: dict[str, str] = {
    # Canvas / surfaces (darkest -> lightest)
    "app":        "#0e0e16",   # window background
    "surface":    "#16161f",   # panels, side bar, video letterbox frame
    "surface_2":  "#1c1c28",   # inputs, list backgrounds
    "card":       "#1a1a26",   # card background
    "elevated":   "#23232f",   # hover / pressed surfaces, headers
    # Borders
    "border":     "#2a2a3c",   # default hairline border
    "border_soft":"#222230",   # very subtle separators
    "border_strong": "#3a3a52",
    # Text
    "text":       "#e8e9f3",   # primary text
    "text_muted": "#9aa0bf",   # secondary text, hints
    "text_dim":   "#676d8f",   # disabled / tertiary
    # Accents
    "accent":     "#7c5cff",   # primary violet
    "accent_hi":  "#9279ff",   # hover
    "accent_lo":  "#6a48f0",   # pressed
    "accent_soft":"#2a2350",   # tinted background for selected items
    "teal":       "#3fd0c9",   # secondary accent / animal 1
    "amber":      "#ffb070",   # warning / animal 2
    "rose":       "#ff6f8c",   # danger
    "green":      "#5fd49a",   # success
}


def _c(name: str) -> str:
    """Shorthand accessor for :data:`COLORS` used inside QSS f-strings."""
    return COLORS[name]


# ── Global stylesheet ─────────────────────────────────────────────────────────

def build_app_qss() -> str:
    """Return the full application stylesheet built from :data:`COLORS`."""
    c = COLORS
    return f"""
/* ── Base ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background: {c['app']};
    color: {c['text']};
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    font-size: 13px;
}}
QToolTip {{
    background: {c['elevated']};
    color: {c['text']};
    border: 1px solid {c['border_strong']};
    padding: 5px 9px;
    border-radius: 6px;
}}

/* ── Cards & group boxes ──────────────────────────────────────────── */
QFrame#card {{
    background: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 10px;
}}
QGroupBox {{
    background: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: {c['text_muted']};
}}
/* Top-level panel: borderless, no redundant title (the side panel already
   shows the section name). Set via objectName('panelRoot'). */
QGroupBox#panelRoot, QWidget#panelRoot {{
    background: transparent;
    border: none;
    margin-top: 0;
    padding: 2px;
}}
QGroupBox#panelRoot::title {{ width: 0; height: 0; margin: 0; padding: 0; }}

/* ── Labels ───────────────────────────────────────────────────────── */
QLabel {{ color: {c['text']}; background: transparent; border: none; }}
QLabel#cardTitle {{ color: {c['text']}; font-size: 13px; font-weight: 700; }}
QLabel#cardSubtitle, QLabel#hint {{ color: {c['text_muted']}; font-size: 11px; }}
QLabel#sectionTitle {{
    color: {c['text_muted']}; font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
}}

/* ── Buttons ──────────────────────────────────────────────────────── */
QPushButton {{
    background: {c['accent']};
    color: white;
    border: none;
    padding: 7px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 12px;
    min-height: 18px;
}}
QPushButton:hover {{ background: {c['accent_hi']}; }}
QPushButton:pressed {{ background: {c['accent_lo']}; }}
QPushButton:disabled {{ background: {c['elevated']}; color: {c['text_dim']}; }}

QPushButton#secondary, QPushButton[variant="secondary"] {{
    background: {c['surface_2']};
    color: {c['text']};
    border: 1px solid {c['border_strong']};
}}
QPushButton#secondary:hover, QPushButton[variant="secondary"]:hover {{
    background: {c['elevated']}; border-color: {c['accent']};
}}
QPushButton#ghost, QPushButton[variant="ghost"] {{
    background: transparent; color: {c['text_muted']};
    border: 1px solid transparent;
}}
QPushButton#ghost:hover {{ background: {c['elevated']}; color: {c['text']}; }}
QPushButton#danger, QPushButton[variant="danger"] {{
    background: {c['rose']}; color: #1a0b10;
}}
QPushButton#danger:hover {{ background: #ff89a1; }}
QPushButton#success {{ background: {c['green']}; color: #06170f; }}

/* ── Text inputs ──────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {c['surface_2']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 6px 10px;
    border-radius: 8px;
    min-height: 20px;
    selection-background-color: {c['accent']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {c['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c['text_dim']}; background: {c['surface']};
}}
QLineEdit[readOnly="true"] {{ background: {c['surface']}; color: {c['text_muted']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {c['elevated']}; border: none; width: 16px; border-radius: 3px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {c['accent']};
}}

/* ── Combo boxes ──────────────────────────────────────────────────── */
QComboBox {{
    background: {c['surface_2']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 6px 12px;
    border-radius: 8px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: {c['border_strong']}; }}
QComboBox:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {c['elevated']};
    color: {c['text']};
    border: 1px solid {c['border_strong']};
    border-radius: 8px;
    selection-background-color: {c['accent']};
    outline: none;
    padding: 4px;
}}

/* ── Checkboxes & radio ───────────────────────────────────────────── */
QCheckBox, QRadioButton {{ color: {c['text']}; spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {c['border_strong']};
    background: {c['surface_2']};
}}
QCheckBox::indicator {{ border-radius: 5px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {c['accent']}; }}
QCheckBox::indicator:checked {{
    background: {c['accent']}; border-color: {c['accent']};
    image: none;
}}
QRadioButton::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}

/* ── Toggle chips (checkable QToolButton with objectName 'chip') ──── */
QToolButton#chip {{
    background: {c['surface_2']};
    color: {c['text_muted']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 600;
}}
QToolButton#chip:hover {{ border-color: {c['border_strong']}; color: {c['text']}; }}
QToolButton#chip:checked {{
    background: {c['accent_soft']};
    color: {c['text']};
    border-color: {c['accent']};
}}
QToolButton#iconButton {{
    background: {c['surface_2']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 6px;
}}
QToolButton#iconButton:hover {{ background: {c['elevated']}; border-color: {c['accent']}; }}

/* ── Sliders ──────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{ background: {c['surface_2']}; height: 6px; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: white; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px; border: 2px solid {c['accent']};
}}
QSlider::handle:horizontal:hover {{ background: {c['accent_hi']}; }}

/* ── Progress ─────────────────────────────────────────────────────── */
QProgressBar {{
    background: {c['surface_2']}; border: none; border-radius: 6px;
    text-align: center; color: {c['text']}; height: 18px; font-size: 11px;
}}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 6px; }}

/* ── Lists & tables ───────────────────────────────────────────────── */
QListWidget, QTreeWidget, QTableWidget {{
    background: {c['surface_2']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{ padding: 5px 7px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {c['accent']}; color: white; }}
QListWidget::item:hover {{ background: {c['elevated']}; }}
QTableWidget {{ gridline-color: {c['border']}; }}
QTableWidget::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QHeaderView::section {{
    background: {c['elevated']}; color: {c['text_muted']};
    padding: 6px 8px; border: none; border-right: 1px solid {c['border']};
    font-weight: 600; font-size: 11px;
}}

/* ── Tabs ─────────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {c['text_muted']};
    padding: 7px 14px; border: none; margin-right: 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ background: {c['card']}; color: {c['text']}; }}
QTabBar::tab:hover {{ color: {c['text']}; }}

/* ── Scrollbars ───────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['border_strong']}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {c['accent']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['border_strong']}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {c['accent']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── Splitter & menus ─────────────────────────────────────────────── */
QSplitter::handle {{ background: {c['border']}; }}
QSplitter::handle:hover {{ background: {c['accent']}; }}
QMenuBar {{ background: {c['surface']}; color: {c['text']}; border-bottom: 1px solid {c['border']}; }}
QMenuBar::item {{ padding: 5px 11px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {c['elevated']}; }}
QMenu {{ background: {c['elevated']}; color: {c['text']}; border: 1px solid {c['border_strong']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['accent']}; color: white; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 4px 8px; }}
"""


# ── Reusable widgets ──────────────────────────────────────────────────────────

class Card(QFrame):
    """A titled content container with a clean header and an accent stripe.

    Add child widgets/layouts to :attr:`body` (a ``QVBoxLayout``). The card has
    no nested ``QGroupBox`` border noise, giving panels a flat, modern feel.
    """

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        *,
        accent: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Build the card frame.

        Args:
            title: Bold heading shown in the card header. Pass an empty string
                to create a header-less card (body only).
            subtitle: Smaller muted description line below the title.
            accent: CSS hex colour for the left accent stripe. Defaults to the
                primary violet (``COLORS['accent']``). Pass a teal or amber
                value to colour-code cards by animal or category.
            parent: Optional Qt parent widget.

        After construction, add child widgets or layouts via :attr:`body`.
        The optional :attr:`header_row` attribute (``QHBoxLayout``) is set
        only when *title* is non-empty.
        """
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)

        if title:
            head = QHBoxLayout()
            head.setSpacing(8)
            # A 3 px wide coloured stripe provides a quick visual accent
            # without the visual weight of a full-colour header band.
            stripe = QFrame()
            stripe.setFixedWidth(3)
            stripe.setMinimumHeight(16)
            stripe.setStyleSheet(
                f"background: {accent or COLORS['accent']}; border-radius: 2px;"
            )
            head.addWidget(stripe)
            titles = QVBoxLayout()
            titles.setSpacing(1)
            lbl = QLabel(title)
            lbl.setObjectName("cardTitle")
            titles.addWidget(lbl)
            if subtitle:
                sub = QLabel(subtitle)
                sub.setObjectName("cardSubtitle")
                sub.setWordWrap(True)
                titles.addWidget(sub)
            head.addLayout(titles, 1)
            self.header_row = head
            outer.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)


def section_title(text: str) -> QLabel:
    """An uppercase, letter-spaced section heading."""
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


def hint(text: str = "") -> QLabel:
    """A muted, word-wrapped helper label."""
    lbl = QLabel(text)
    lbl.setObjectName("hint")
    lbl.setWordWrap(True)
    return lbl


def divider() -> QFrame:
    """A thin horizontal separator line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {COLORS['border']}; border: none;")
    return line


def style_button(btn: QPushButton, variant: str = "primary") -> QPushButton:
    """Tag a button with a visual variant: primary | secondary | ghost | danger | success."""
    if variant != "primary":
        btn.setObjectName(variant)
    return btn


class ToggleChip(QToolButton):
    """A compact, checkable chip with an optional Feather icon and label.

    Used for overlay toggles (Skeleton / Labels / Behavior ...). Far clearer
    than a bare ``QCheckBox`` row because the whole chip lights up when active.
    """

    def __init__(
        self,
        label: str,
        icon_name: Optional[str] = None,
        *,
        checked: bool = False,
        tooltip: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Create a toggle chip.

        Args:
            label: Text displayed on the chip (e.g. "Skeleton", "Labels").
            icon_name: Optional Feather icon name (e.g. ``"eye"``). When
                provided the icon appears to the left of the label and its
                colour shifts between ``text_muted`` (off) and ``text``
                (on) to reinforce the toggle state visually.
            checked: Initial checked state of the chip.
            tooltip: Optional tooltip string shown on hover.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setObjectName("chip")
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            if icon_name
            else Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.setText(label)
        if tooltip:
            self.setToolTip(tooltip)
        self._icon_name = icon_name
        if icon_name:
            self._refresh_icon()
            # Re-colour the icon whenever the toggle state changes so the
            # icon brightness tracks the chip's checked/unchecked state.
            self.toggled.connect(lambda _checked: self._refresh_icon())

    def _refresh_icon(self) -> None:
        """Rerender the Feather icon in the colour matching the current toggle state.

        Called once at construction and again on every ``toggled`` signal so
        the icon always reflects whether the chip is on or off.
        """
        from shared.icons import icon_qicon
        color = COLORS["text"] if self.isChecked() else COLORS["text_muted"]
        self.setIcon(icon_qicon(self._icon_name, size=15, color=color))
        self.setIconSize(QSize(15, 15))
