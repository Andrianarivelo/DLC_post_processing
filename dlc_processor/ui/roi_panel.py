"""ROI (Region of Interest) management panel.

Users can define ROIs as circles, rectangles, or polygons by specifying
coordinates and dimensions. ROIs are drawn on the video overlay and used
to compute occupancy metrics (time spent, entry frequency, behaviour-in-zone).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)

# ── ROI data model ───────────────────────────────────────────────────────────

_DEFAULT_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


class ROIDef:
    """In-memory ROI definition."""

    def __init__(
        self,
        name: str,
        shape: str,  # "circle", "rectangle", "polygon"
        cx: float = 0,
        cy: float = 0,
        radius: float = 50,
        width: float = 100,
        height: float = 100,
        points: Optional[list[tuple[float, float]]] = None,
        color: str = "#e74c3c",
    ) -> None:
        self.name = name
        self.shape = shape
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.width = width
        self.height = height
        self.points = points or []
        self.color = color

    def as_polygon(self) -> list[tuple[float, float]]:
        """Convert ROI to a polygon (list of (x, y) tuples)."""
        if self.shape == "polygon":
            return list(self.points)
        elif self.shape == "circle":
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            return [(self.cx + self.radius * np.cos(a),
                     self.cy + self.radius * np.sin(a)) for a in angles]
        elif self.shape == "rectangle":
            x0 = self.cx - self.width / 2
            y0 = self.cy - self.height / 2
            x1 = self.cx + self.width / 2
            y1 = self.cy + self.height / 2
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return []

    def to_dict(self) -> dict:
        return {
            "name": self.name, "shape": self.shape,
            "cx": self.cx, "cy": self.cy,
            "radius": self.radius, "width": self.width, "height": self.height,
            "points": self.points, "color": self.color,
        }

    @staticmethod
    def from_dict(d: dict) -> "ROIDef":
        return ROIDef(**d)


# ── Panel ────────────────────────────────────────────────────────────────────

class ROIPanel(QGroupBox):
    """ROI definition, editing, and metrics panel."""

    rois_changed = Signal(list)  # list[ROIDef] — emitted when ROIs are added/removed/edited

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Regions of Interest", parent)
        self._rois: list[ROIDef] = []
        self._animal_dfs: dict[str, pd.DataFrame] = {}
        self._fps = 25.0
        self._frame_w = 0
        self._frame_h = 0
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Shape selection
        add_group = QGroupBox("Add ROI")
        add_lay = QVBoxLayout(add_group)
        add_lay.setSpacing(5)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Shape:"))
        self._combo_shape = QComboBox()
        self._combo_shape.addItems(["circle", "rectangle", "polygon"])
        self._combo_shape.currentTextChanged.connect(self._on_shape_changed)
        row1.addWidget(self._combo_shape)
        row1.addWidget(QLabel("Name:"))
        self._edit_name = QLineEdit()
        self._edit_name.setPlaceholderText("ROI name")
        row1.addWidget(self._edit_name)
        add_lay.addLayout(row1)

        # Coordinate form
        form = QFormLayout()
        form.setSpacing(4)

        self._spin_cx = QDoubleSpinBox()
        self._spin_cx.setRange(0, 10000)
        self._spin_cx.setDecimals(0)
        self._spin_cx.setSuffix(" px")
        form.addRow("Centre X:", self._spin_cx)

        self._spin_cy = QDoubleSpinBox()
        self._spin_cy.setRange(0, 10000)
        self._spin_cy.setDecimals(0)
        self._spin_cy.setSuffix(" px")
        form.addRow("Centre Y:", self._spin_cy)

        self._spin_radius = QDoubleSpinBox()
        self._spin_radius.setRange(1, 5000)
        self._spin_radius.setValue(50)
        self._spin_radius.setDecimals(0)
        self._spin_radius.setSuffix(" px")
        self._lbl_radius = QLabel("Radius:")
        form.addRow(self._lbl_radius, self._spin_radius)

        self._spin_width = QDoubleSpinBox()
        self._spin_width.setRange(1, 10000)
        self._spin_width.setValue(100)
        self._spin_width.setDecimals(0)
        self._spin_width.setSuffix(" px")
        self._lbl_width = QLabel("Width:")
        form.addRow(self._lbl_width, self._spin_width)

        self._spin_height = QDoubleSpinBox()
        self._spin_height.setRange(1, 10000)
        self._spin_height.setValue(100)
        self._spin_height.setDecimals(0)
        self._spin_height.setSuffix(" px")
        self._lbl_height = QLabel("Height:")
        form.addRow(self._lbl_height, self._spin_height)

        self._edit_points = QLineEdit()
        self._edit_points.setPlaceholderText("x1,y1;x2,y2;x3,y3;...")
        self._edit_points.setToolTip("Semicolon-separated x,y pairs for polygon vertices")
        self._lbl_points = QLabel("Points:")
        form.addRow(self._lbl_points, self._edit_points)

        add_lay.addLayout(form)

        btn_add = QPushButton("Add ROI")
        btn_add.clicked.connect(self._add_roi)
        add_lay.addWidget(btn_add)
        layout.addWidget(add_group)

        # Show/hide based on shape
        self._on_shape_changed("circle")

        # ROI list
        list_group = QGroupBox("Defined ROIs")
        list_lay = QVBoxLayout(list_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(140)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._roi_list_container = QWidget()
        self._roi_list_layout = QVBoxLayout(self._roi_list_container)
        self._roi_list_layout.setContentsMargins(0, 0, 0, 0)
        self._roi_list_layout.setSpacing(2)
        self._roi_list_layout.addStretch()
        scroll.setWidget(self._roi_list_container)
        list_lay.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        list_lay.addLayout(btn_row)
        layout.addWidget(list_group)

        # Analysis
        analysis_group = QGroupBox("ROI Analysis")
        analysis_lay = QVBoxLayout(analysis_group)

        kp_row = QHBoxLayout()
        kp_row.addWidget(QLabel("Keypoint:"))
        self._combo_keypoint = QComboBox()
        self._combo_keypoint.addItem("body_centre")
        kp_row.addWidget(self._combo_keypoint)

        self._spin_proximity = QDoubleSpinBox()
        self._spin_proximity.setRange(0, 500)
        self._spin_proximity.setValue(30)
        self._spin_proximity.setSuffix(" px")
        self._spin_proximity.setToolTip("Distance threshold for 'close to ROI' (0 = in ROI only)")
        kp_row.addWidget(QLabel("Proximity:"))
        kp_row.addWidget(self._spin_proximity)
        analysis_lay.addLayout(kp_row)

        btn_analyze = QPushButton("Analyze ROI Occupancy")
        btn_analyze.clicked.connect(self._analyze)
        analysis_lay.addWidget(btn_analyze)

        self._results_table = QTableWidget(0, 6)
        self._results_table.setHorizontalHeaderLabels([
            "ROI", "Animal", "Frames In", "Time %", "Entries", "Close %",
        ])
        self._results_table.setMaximumHeight(200)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        analysis_lay.addWidget(self._results_table)

        layout.addWidget(analysis_group)

    # ── Shape visibility ─────────────────────────────────────────────────

    def _on_shape_changed(self, shape: str) -> None:
        is_circle = shape == "circle"
        is_rect = shape == "rectangle"
        is_poly = shape == "polygon"

        self._lbl_radius.setVisible(is_circle)
        self._spin_radius.setVisible(is_circle)
        self._lbl_width.setVisible(is_rect)
        self._spin_width.setVisible(is_rect)
        self._lbl_height.setVisible(is_rect)
        self._spin_height.setVisible(is_rect)
        self._lbl_points.setVisible(is_poly)
        self._edit_points.setVisible(is_poly)

        # Centre is used for circle and rectangle
        cx_visible = not is_poly
        self._spin_cx.setVisible(cx_visible)
        self._spin_cy.setVisible(cx_visible)

    # ── Public API ───────────────────────────────────────────────────────

    def set_animal_dfs(self, dfs: dict, fps: float = 25.0) -> None:
        self._animal_dfs = dfs
        self._fps = fps
        # Update keypoint combo
        self._combo_keypoint.clear()
        self._combo_keypoint.addItem("body_centre")
        if dfs:
            first_df = next(iter(dfs.values()))
            for bp in get_bodyparts(first_df):
                if bp not in ("body_centre",):
                    self._combo_keypoint.addItem(bp)

    def set_video_info(self, w: int, h: int) -> None:
        self._frame_w = w
        self._frame_h = h
        self._spin_cx.setRange(0, max(w, 1))
        self._spin_cy.setRange(0, max(h, 1))
        # Default centre to frame centre
        if self._spin_cx.value() == 0:
            self._spin_cx.setValue(w // 2)
            self._spin_cy.setValue(h // 2)

    def get_rois(self) -> list[ROIDef]:
        return list(self._rois)

    # ── Add / Remove ROIs ────────────────────────────────────────────────

    def _add_roi(self) -> None:
        shape = self._combo_shape.currentText()
        name = self._edit_name.text().strip()
        if not name:
            name = f"ROI_{len(self._rois) + 1}"
        color = _DEFAULT_COLORS[len(self._rois) % len(_DEFAULT_COLORS)]

        if shape == "polygon":
            pts = self._parse_points(self._edit_points.text())
            if len(pts) < 3:
                return
            roi = ROIDef(name=name, shape="polygon", points=pts, color=color)
        elif shape == "circle":
            roi = ROIDef(
                name=name, shape="circle",
                cx=self._spin_cx.value(), cy=self._spin_cy.value(),
                radius=self._spin_radius.value(), color=color,
            )
        else:  # rectangle
            roi = ROIDef(
                name=name, shape="rectangle",
                cx=self._spin_cx.value(), cy=self._spin_cy.value(),
                width=self._spin_width.value(), height=self._spin_height.value(),
                color=color,
            )

        self._rois.append(roi)
        self._rebuild_roi_list()
        self.rois_changed.emit(self._rois)
        self._edit_name.clear()

    def _remove_roi(self, name: str) -> None:
        self._rois = [r for r in self._rois if r.name != name]
        self._rebuild_roi_list()
        self.rois_changed.emit(self._rois)

    def _clear_all(self) -> None:
        self._rois.clear()
        self._rebuild_roi_list()
        self.rois_changed.emit(self._rois)
        self._results_table.setRowCount(0)

    def _rebuild_roi_list(self) -> None:
        # Clear existing widgets (except final stretch)
        while self._roi_list_layout.count() > 1:
            item = self._roi_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for roi in self._rois:
            row_w = QWidget()
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(2, 1, 2, 1)
            row_lay.setSpacing(4)

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: {roi.color}; border: 1px solid #555; border-radius: 2px;"
            )
            row_lay.addWidget(swatch)

            info = f"{roi.name} ({roi.shape})"
            if roi.shape == "circle":
                info += f" r={roi.radius:.0f}"
            elif roi.shape == "rectangle":
                info += f" {roi.width:.0f}x{roi.height:.0f}"
            elif roi.shape == "polygon":
                info += f" {len(roi.points)} pts"
            lbl = QLabel(info)
            lbl.setStyleSheet("color: #cdd6f4; font-size: 11px;")
            row_lay.addWidget(lbl, 1)

            btn_del = QPushButton("x")
            btn_del.setFixedSize(20, 20)
            btn_del.setStyleSheet("font-size: 10px; padding: 0px;")
            btn_del.clicked.connect(lambda checked=False, n=roi.name: self._remove_roi(n))
            row_lay.addWidget(btn_del)

            self._roi_list_layout.insertWidget(self._roi_list_layout.count() - 1, row_w)

    @staticmethod
    def _parse_points(text: str) -> list[tuple[float, float]]:
        """Parse 'x1,y1;x2,y2;...' into list of (x, y) tuples."""
        pts: list[tuple[float, float]] = []
        for part in text.split(";"):
            part = part.strip()
            if "," not in part:
                continue
            try:
                xs, ys = part.split(",", 1)
                pts.append((float(xs.strip()), float(ys.strip())))
            except ValueError:
                continue
        return pts

    # ── Analysis ─────────────────────────────────────────────────────────

    @Slot()
    def _analyze(self) -> None:
        if not self._rois or not self._animal_dfs:
            return

        from dlc_processor.core.roi_analyzer import ROIAnalyzer, ROI

        analyzer = ROIAnalyzer(fps=self._fps)
        for roi_def in self._rois:
            poly = roi_def.as_polygon()
            if len(poly) >= 3:
                analyzer.add_roi(roi_def.name, poly)

        keypoint = self._combo_keypoint.currentText()
        proximity_px = self._spin_proximity.value()

        rows: list[tuple[str, ...]] = []
        for animal_id, df in self._animal_dfs.items():
            results = analyzer.analyze(df, animal_id, keypoint)
            for zr in results:
                # Entry count: number of False→True transitions
                occ = zr.occupancy
                entries = int(np.sum(np.diff(occ.astype(np.int8)) == 1))

                # Close-to-ROI: expand polygon by proximity_px and re-test
                close_pct = zr.pct_in  # default = same as in-ROI
                if proximity_px > 0:
                    close_frames = _compute_proximity(
                        df, keypoint, self._rois, zr.roi_name, proximity_px,
                    )
                    close_pct = 100.0 * close_frames / max(len(df), 1)

                rows.append((
                    zr.roi_name, animal_id,
                    str(zr.frames_in), f"{zr.pct_in:.1f}%",
                    str(entries), f"{close_pct:.1f}%",
                ))

        self._results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self._results_table.setItem(r, c, QTableWidgetItem(val))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_proximity(
    df: pd.DataFrame,
    keypoint: str,
    rois: list[ROIDef],
    roi_name: str,
    proximity_px: float,
) -> int:
    """Count frames where keypoint is within proximity_px of the ROI boundary."""
    import cv2

    bps = get_bodyparts(df)
    kp = _resolve_kp(keypoint, bps)
    if kp is None:
        return 0

    x = df.get(f"{kp}_x", pd.Series(dtype=float)).to_numpy(np.float64)
    y = df.get(f"{kp}_y", pd.Series(dtype=float)).to_numpy(np.float64)

    roi_def = next((r for r in rois if r.name == roi_name), None)
    if roi_def is None:
        return 0

    poly = roi_def.as_polygon()
    if len(poly) < 3:
        return 0

    contour = np.array([[int(px), int(py)] for px, py in poly], dtype=np.int32)
    contour = contour.reshape((-1, 1, 2))

    count = 0
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        dist = cv2.pointPolygonTest(contour, (float(x[i]), float(y[i])), True)
        # dist > 0 = inside, dist == 0 = on boundary, dist < 0 = outside
        # Close = inside OR within proximity_px of boundary
        if dist >= -proximity_px:
            count += 1
    return count


def _resolve_kp(name: str, bps: list[str]) -> Optional[str]:
    """Resolve keypoint name to available bodypart."""
    if name == "body_centre":
        candidates = ["center", "Centre", "body_centre", "neck", "Neck", "mid"]
        for c in candidates:
            if c in bps:
                return c
        return bps[0] if bps else None
    nl = name.lower()
    for bp in bps:
        if bp.lower() == nl:
            return bp
    return bps[0] if bps else None
