"""Reusable activity-bar layout supporting both right-side vertical
and bottom horizontal bar placement.

Two modes (set via ``bar_position``):

``"right"`` (MouseTracker, Ethoscore)::

    ┌───────────────────────────┬────────────┬────┐
    │                           │            │    │
    │   central (stretch)       │ slide      │ AB │  ← vertical bar
    │                           │ panel      │    │
    ├───────────────────────────┤            │    │
    │   pinned bottom (opt.)    │            │    │
    └───────────────────────────┴────────────┴────┘

``"bottom"`` (DLC Processor)::

    ┌──────────────────────────────────────────────┐
    │   [pinned left (opt.)]  │  central (stretch) │
    ├──────────────────────────────────────────────┤
    │   slide panel                                │
    ├──────────────────────────────────────────────┤
    │   activity bar (horizontal)                  │
    └──────────────────────────────────────────────┘

Toggle behaviour
----------------
Uses ``pressed`` signal (not ``clicked``) so Qt never auto-toggles
button checked state.  State is managed entirely in Python.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from shared.icons import icon_qicon

# ── Constants ─────────────────────────────────────────────────────────────────

PANEL_H = 340   # slide-up panel height
PANEL_W = 560   # slide-left panel width
BAR_H   = 44    # horizontal bar height
BAR_W   = 48    # vertical bar width

_HBAR_QSS = """
QWidget#activityBar {
    background: #181825;
    border-top: 1px solid #313244;
}
"""

_VBAR_QSS = """
QWidget#activityBar {
    background: #181825;
    border-left: 1px solid #313244;
}
"""

_BTN_QSS = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px;
    margin: 2px;
}
QToolButton:hover {
    background: #313244;
}
QToolButton[active="true"] {
    background: #313244;
}
"""

_PANEL_QSS = """
QWidget#slidePanel {
    background: #1e1e2e;
}
QLabel#panelTitle {
    color: #a6adc8;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 6px 12px 4px;
}
"""

_PINNED_QSS = "background: #252535; border-right: 1px solid #3d3d5c;"


# ── Activity bar (horizontal or vertical) ─────────────────────────────────────

class ActivityBar(QWidget):
    """Strip of icon buttons — horizontal (bottom) or vertical (right)."""

    def __init__(self, orientation: str = "horizontal", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("activityBar")
        self._orientation = orientation

        if orientation == "horizontal":
            self.setFixedHeight(BAR_H)
            self.setStyleSheet(_HBAR_QSS + _BTN_QSS)
            self._layout = QHBoxLayout(self)
            self._layout.setContentsMargins(6, 2, 6, 2)
        else:
            self.setFixedWidth(BAR_W)
            self.setStyleSheet(_VBAR_QSS + _BTN_QSS)
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(2, 6, 2, 6)

        self._layout.setSpacing(2)
        self._layout.addStretch()
        self._buttons: list[QToolButton] = []

    def add_button(self, key: str, icon_name: str, label: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName(key)
        btn.setToolTip(label)
        btn.setIcon(icon_qicon(icon_name, size=18, color="#6c7086"))
        btn.setIconSize(QSize(18, 18))
        btn.setCheckable(False)
        btn.setProperty("active", "false")
        btn.setProperty("icon_name", icon_name)

        if self._orientation == "horizontal":
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setText(label)
            btn.setFixedHeight(BAR_H - 4)
            # Explicit font to avoid QFont::setPointSize(-1) on Windows
            font = btn.font()
            font.setPixelSize(10)
            btn.setFont(font)
        else:
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            btn.setFixedSize(BAR_W - 8, BAR_W - 8)

        # Insert before trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, btn)
        self._buttons.append(btn)
        return btn

    def add_separator(self) -> None:
        sep = QFrame()
        if self._orientation == "horizontal":
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet("color: #3d3d5c; margin: 6px 4px;")
        else:
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color: #3d3d5c; margin: 4px 6px;")
        self._layout.insertWidget(self._layout.count() - 1, sep)

    def set_active(self, key: Optional[str]) -> None:
        for btn in self._buttons:
            is_active = btn.objectName() == key
            btn.setProperty("active", "true" if is_active else "false")
            icon_name = btn.property("icon_name")
            if icon_name:
                color = "#cba6f7" if is_active else "#6c7086"
                btn.setIcon(icon_qicon(icon_name, size=18, color=color))
            btn.style().unpolish(btn)
            btn.style().polish(btn)


# ── Sliding panel ─────────────────────────────────────────────────────────────

class SlidePanel(QWidget):
    """Panel that slides open in a given direction."""

    def __init__(self, direction: str = "up", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("slidePanel")
        self.setStyleSheet(_PANEL_QSS)
        self._direction = direction

        if direction == "up":
            self.setMaximumHeight(0)
            self.setMinimumHeight(0)
            self.setStyleSheet(_PANEL_QSS + "QWidget#slidePanel { border-top: 1px solid #313244; }")
            self._prop = b"maximumHeight"
            self._target = PANEL_H
        else:   # "left"
            self.setMaximumWidth(0)
            self.setMinimumWidth(0)
            self.setStyleSheet(_PANEL_QSS + "QWidget#slidePanel { border-left: 1px solid #313244; }")
            self._prop = b"maximumWidth"
            self._target = PANEL_W

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title = QLabel()
        self._title.setObjectName("panelTitle")
        font = self._title.font()
        font.setPixelSize(10)
        self._title.setFont(font)
        outer.addWidget(self._title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244; margin: 0;")
        outer.addWidget(sep)

        self._stack = QStackedWidget()
        outer.addWidget(self._stack, 1)

        self._anim = QPropertyAnimation(self, self._prop)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._open = False
        self._page_idx: dict[str, int] = {}

    def add_page(self, key: str, widget: QWidget) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        self._page_idx[key] = self._stack.addWidget(scroll)

    def show_page(self, key: str, title: str) -> None:
        self._stack.setCurrentIndex(self._page_idx.get(key, 0))
        self._title.setText(title.upper())
        if not self._open:
            self._open = True
            if self._direction == "left":
                self.setMinimumWidth(self._target)
            else:
                self.setMinimumHeight(self._target)
            self._animate(0, self._target)

    def close_panel(self) -> None:
        if self._open:
            self._open = False
            if self._direction == "left":
                self.setMinimumWidth(0)
            else:
                self.setMinimumHeight(0)
            cur = self.height() if self._direction == "up" else self.width()
            self._animate(cur, 0)

    def is_open(self) -> bool:
        return self._open

    def _animate(self, start: int, end: int) -> None:
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()


# ── Composite layout ──────────────────────────────────────────────────────────

class SidebarLayout(QWidget):
    """
    Activity-bar layout with two orientations.

    Parameters
    ----------
    bar_position : "bottom" or "right"
    """

    def __init__(self, bar_position: str = "bottom", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bar_position = bar_position

        if bar_position == "right":
            self._init_right()
        else:
            self._init_bottom()

        self._activities: dict[str, tuple[QToolButton, str]] = {}
        self._active_key: Optional[str] = None

    # ── Right-bar init ────────────────────────────────────────────────────────

    def _init_right(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left column: central + optional pinned bottom
        self._left_col = QVBoxLayout()
        self._left_col.setContentsMargins(0, 0, 0, 0)
        self._left_col.setSpacing(0)

        self._central_holder = QWidget()
        self._central_layout = QVBoxLayout(self._central_holder)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._left_col.addWidget(self._central_holder, 1)

        root.addLayout(self._left_col, 1)

        # Slide panel (slides left from bar)
        self._panel = SlidePanel(direction="left")
        root.addWidget(self._panel)

        # Vertical activity bar
        self._bar = ActivityBar(orientation="vertical")
        root.addWidget(self._bar)

    # ── Bottom-bar init ───────────────────────────────────────────────────────

    def _init_bottom(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top row: optional pinned left + central
        self._top_row = QHBoxLayout()
        self._top_row.setContentsMargins(0, 0, 0, 0)
        self._top_row.setSpacing(0)

        self._central_holder = QWidget()
        self._central_layout = QVBoxLayout(self._central_holder)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._top_row.addWidget(self._central_holder, 1)

        root.addLayout(self._top_row, 1)

        # Slide panel (slides up)
        self._panel = SlidePanel(direction="up")
        root.addWidget(self._panel)

        # Horizontal activity bar
        self._bar = ActivityBar(orientation="horizontal")
        root.addWidget(self._bar)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_pinned_sidebar(self, widget: QWidget, width: int = 270) -> None:
        """Left pinned column (bottom-bar mode only)."""
        container = QWidget()
        container.setFixedWidth(width)
        container.setStyleSheet(_PINNED_QSS)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(widget)
        self._top_row.insertWidget(0, container)

    def set_pinned_bottom(self, widget: QWidget) -> None:
        """Always-visible section below the central area (right-bar mode)."""
        widget.setStyleSheet(
            widget.styleSheet()
            + "; border-top: 1px solid #313244;"
        )
        self._left_col.addWidget(widget)

    def add_activity(
        self,
        key: str,
        icon_name: str,
        label: str,
        title: str,
        panel_widget: QWidget,
    ) -> None:
        btn = self._bar.add_button(key, icon_name, label)
        self._panel.add_page(key, panel_widget)
        self._activities[key] = (btn, title)
        btn.pressed.connect(lambda k=key, t=title: self._on_activity(k, t))

    def add_bar_separator(self) -> None:
        self._bar.add_separator()

    def set_central(self, widget: QWidget) -> None:
        for i in reversed(range(self._central_layout.count())):
            item = self._central_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self._central_layout.addWidget(widget)

    def show_activity(self, key: str) -> None:
        """Programmatically open a panel (e.g. to pre-select Identity)."""
        if key in self._activities:
            _, title = self._activities[key]
            self._active_key = key
            self._bar.set_active(key)
            self._panel.show_page(key, title)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_activity(self, key: str, title: str) -> None:
        if self._active_key == key:
            self._panel.close_panel()
            self._active_key = None
            self._bar.set_active(None)
        else:
            self._active_key = key
            self._bar.set_active(key)
            self._panel.show_page(key, title)
