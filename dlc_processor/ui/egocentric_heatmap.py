"""Egocentric position heatmap dialog.

Shows where animal B spends time relative to animal A in A's body-frame
(A centered, body axis pointing up). Annotated with concentric distance
rings and angular labels like a polar plot, with the mouse SVG silhouette
overlaid at the centre.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dlc_processor.core.dlc_loader import get_bodyparts

logger = logging.getLogger(__name__)

_SVG_PATH = Path(__file__).resolve().parent.parent / "assets" / "mouse_top.svg"

# Heatmap grid: ±radius pixels in egocentric space
_DEFAULT_RADIUS = 300   # px
_DEFAULT_BINS = 80

# Annotation style
_GRID_PEN = pg.mkPen(color=(200, 200, 220, 60), width=1, style=Qt.PenStyle.DashLine)
_GRID_TEXT_COLOR = "#b0b8d0"
_ANGLE_TEXT_COLOR = "#d0d4e0"


class EgocentricHeatmapDialog(QDialog):
    """Dialog showing egocentric position heatmap with polar annotations."""

    def __init__(
        self,
        animal_dfs: dict,
        fps: float = 25.0,
        current_frame: int = 0,
        px_per_cm: float = 0.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Egocentric Position Heatmap")
        self.setMinimumSize(580, 680)
        self.resize(640, 740)

        self._animal_dfs = animal_dfs
        self._fps = fps
        self._current_frame = current_frame
        self._px_per_cm = float(max(px_per_cm, 0.0))
        self._animals = list(animal_dfs.keys())

        # Persistent overlay items
        self._svg_item: Optional[QGraphicsPixmapItem] = None
        self._grid_items: list = []  # distance rings + angle lines + labels

        # Cached data for export
        self._last_ego_x: Optional[np.ndarray] = None
        self._last_ego_y: Optional[np.ndarray] = None
        self._last_H: Optional[np.ndarray] = None
        self._last_edges: Optional[np.ndarray] = None
        self._last_meta: dict = {}

        self._setup_ui()
        self._update_heatmap()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Controls
        ctrl = QGroupBox("Settings")
        ctrl_lay = QFormLayout(ctrl)
        ctrl_lay.setSpacing(5)

        self._combo_focal = QComboBox()
        self._combo_other = QComboBox()
        for a in self._animals:
            self._combo_focal.addItem(a)
            self._combo_other.addItem(a)
        if len(self._animals) >= 2:
            self._combo_other.setCurrentIndex(1)
        ctrl_lay.addRow("Focal animal (centre):", self._combo_focal)
        ctrl_lay.addRow("Other animal:", self._combo_other)

        self._spin_radius = QSpinBox()
        self._spin_radius.setToolTip("Radius of the egocentric map around the focal animal")
        self._spin_radius.setRange(50, 2000)
        if self._px_per_cm > 0:
            self._spin_radius.setValue(max(1, int(round(_DEFAULT_RADIUS / self._px_per_cm))))
            self._spin_radius.setSuffix(" cm")
        else:
            self._spin_radius.setValue(_DEFAULT_RADIUS)
            self._spin_radius.setSuffix(" px")
        ctrl_lay.addRow("Radius:", self._spin_radius)

        self._spin_bins = QSpinBox()
        self._spin_bins.setRange(20, 200)
        self._spin_bins.setValue(_DEFAULT_BINS)
        self._spin_bins.setToolTip("Resolution of the heatmap grid (higher = finer detail, slower)")
        ctrl_lay.addRow("Grid bins:", self._spin_bins)

        self._spin_rings = QSpinBox()
        self._spin_rings.setRange(2, 10)
        self._spin_rings.setValue(4)
        self._spin_rings.setToolTip("Number of concentric distance rings")
        ctrl_lay.addRow("Distance rings:", self._spin_rings)

        # Time window
        n_total = max((len(df) for df in self._animal_dfs.values()), default=1)
        self._spin_start = QSpinBox()
        self._spin_start.setRange(0, max(n_total - 1, 0))
        self._spin_start.setValue(0)
        ctrl_lay.addRow("Start frame:", self._spin_start)

        self._spin_end = QSpinBox()
        self._spin_end.setRange(1, n_total)
        self._spin_end.setValue(n_total)
        ctrl_lay.addRow("End frame:", self._spin_end)

        self._spin_window = QSpinBox()
        self._spin_window.setRange(0, n_total)
        self._spin_window.setValue(0)
        self._spin_window.setSpecialValueText("Full range")
        self._spin_window.setSuffix(" fr")
        self._spin_window.setToolTip(
            "Sliding window around current frame (0 = use start/end range)"
        )
        ctrl_lay.addRow("Window (around cursor):", self._spin_window)

        layout.addWidget(ctrl)

        btn_row = QHBoxLayout()
        btn_update = QPushButton("Update")
        btn_update.clicked.connect(self._update_heatmap)
        btn_row.addWidget(btn_update)
        btn_export = QPushButton("Export Data")
        btn_export.setToolTip("Export raw egocentric coordinates and histogram to CSV/NPZ")
        btn_export.clicked.connect(self._export_data)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        self._lbl_info = QLabel("")
        btn_row.addWidget(self._lbl_info)
        layout.addLayout(btn_row)

        # Plot
        self._pw = pg.PlotWidget()
        self._pw.setAspectLocked(True)
        self._pw.hideAxis("left")
        self._pw.hideAxis("bottom")
        self._pw.setBackground("#1e1e2e")
        layout.addWidget(self._pw, 1)

        # ImageItem for heatmap
        self._img_item = pg.ImageItem()
        self._pw.addItem(self._img_item)

        # Pre-render SVG data
        self._svg_pixmap: Optional[QPixmap] = None
        self._svg_w = 0
        self._svg_h = 0
        self._prerender_svg()

        # Colorbar
        self._cbar_widget = pg.GraphicsLayoutWidget()
        self._cbar_widget.setFixedHeight(30)
        self._cbar_widget.setBackground("#1e1e2e")
        layout.addWidget(self._cbar_widget)
        self._cbar_item = None

    def _prerender_svg(self) -> None:
        """Render the mouse SVG to a pixmap for later overlay."""
        if not _SVG_PATH.exists():
            logger.warning("Mouse SVG not found: %s", _SVG_PATH)
            return
        renderer = QSvgRenderer(str(_SVG_PATH))
        if not renderer.isValid():
            return
        svg_h = 200
        vb = renderer.viewBoxF()
        aspect = vb.width() / vb.height() if vb.height() else 0.5
        self._svg_w = int(svg_h * aspect)
        self._svg_h = svg_h

    # ── Computation ──────────────────────────────────────────────────────────

    def set_current_frame(self, frame_idx: int) -> None:
        """Called when the video cursor moves — update if windowed."""
        self._current_frame = frame_idx
        if self._spin_window.value() > 0:
            self._update_heatmap()

    @Slot()
    def _update_heatmap(self) -> None:
        self._clear_export_cache()
        focal_id = self._combo_focal.currentText()
        other_id = self._combo_other.currentText()
        if focal_id == other_id or focal_id not in self._animal_dfs or other_id not in self._animal_dfs:
            self._lbl_info.setText("Select two different animals")
            return

        df_focal = self._animal_dfs[focal_id]
        df_other = self._animal_dfs[other_id]

        # Determine frame range
        window = self._spin_window.value()
        if window > 0:
            half = window // 2
            start = max(0, self._current_frame - half)
            end = min(len(df_focal), self._current_frame + half)
        else:
            start = self._spin_start.value()
            end = self._spin_end.value()

        n = min(len(df_focal), len(df_other), end) - start
        if n < 2:
            self._lbl_info.setText("Not enough frames")
            return

        # Get body coordinates
        bps_focal = get_bodyparts(df_focal)
        nose_col, tail_col = _find_nose_tail(bps_focal)
        if nose_col is None or tail_col is None:
            self._lbl_info.setText("Cannot find nose/tail bodyparts")
            return

        bps_other = get_bodyparts(df_other)
        other_nose, other_tail = _find_nose_tail(bps_other)

        # Extract coordinates for the slice
        sl = slice(start, start + n)
        focal_nose_x = df_focal[f"{nose_col}_x"].to_numpy(dtype=np.float64)[sl]
        focal_nose_y = df_focal[f"{nose_col}_y"].to_numpy(dtype=np.float64)[sl]
        focal_tail_x = df_focal[f"{tail_col}_x"].to_numpy(dtype=np.float64)[sl]
        focal_tail_y = df_focal[f"{tail_col}_y"].to_numpy(dtype=np.float64)[sl]

        # Focal centre = midpoint nose-tail
        focal_cx = (focal_nose_x + focal_tail_x) / 2
        focal_cy = (focal_nose_y + focal_tail_y) / 2

        # Body axis: tail → nose (pointing "forward")
        ax_x = focal_nose_x - focal_tail_x
        ax_y = focal_nose_y - focal_tail_y
        ax_len = np.sqrt(ax_x**2 + ax_y**2)
        ax_len = np.where(ax_len < 1e-6, 1.0, ax_len)
        ax_x /= ax_len
        ax_y /= ax_len

        # Other animal centre
        if other_nose is not None and other_tail is not None:
            on_x = df_other[f"{other_nose}_x"].to_numpy(dtype=np.float64)[sl]
            on_y = df_other[f"{other_nose}_y"].to_numpy(dtype=np.float64)[sl]
            ot_x = df_other[f"{other_tail}_x"].to_numpy(dtype=np.float64)[sl]
            ot_y = df_other[f"{other_tail}_y"].to_numpy(dtype=np.float64)[sl]
            other_x = (on_x + ot_x) / 2
            other_y = (on_y + ot_y) / 2
        elif other_nose is not None:
            other_x = df_other[f"{other_nose}_x"].to_numpy(dtype=np.float64)[sl]
            other_y = df_other[f"{other_nose}_y"].to_numpy(dtype=np.float64)[sl]
        else:
            bp0 = bps_other[0]
            other_x = df_other[f"{bp0}_x"].to_numpy(dtype=np.float64)[sl]
            other_y = df_other[f"{bp0}_y"].to_numpy(dtype=np.float64)[sl]

        # Transform to egocentric coordinates
        # DLC uses image coords (Y down). The forward axis is (ax_x, ax_y).
        # In image-Y-down space:
        #   rightward perpendicular = (-ax_y, ax_x)
        #   forward (dot with axis)  = dx*ax_x + dy*ax_y
        dx = other_x - focal_cx
        dy = other_y - focal_cy

        ego_x = dx * (-ax_y) + dy * ax_x   # rightward of body axis
        ego_y = dx * ax_x + dy * ax_y       # forward along body axis

        # Filter NaN
        valid = np.isfinite(ego_x) & np.isfinite(ego_y)
        ego_x = ego_x[valid]
        ego_y = ego_y[valid]

        if len(ego_x) < 2:
            self._lbl_info.setText("Not enough valid data points")
            return

        # Compute 2D histogram
        radius = (
            self._spin_radius.value() * self._px_per_cm
            if self._px_per_cm > 0
            else self._spin_radius.value()
        )
        bins = self._spin_bins.value()
        edges = np.linspace(-radius, radius, bins + 1)

        H, _, _ = np.histogram2d(ego_x, ego_y, bins=[edges, edges])
        self._last_ego_x = ego_x.copy()
        self._last_ego_y = ego_y.copy()
        self._last_H = H.copy()
        self._last_edges = edges.copy()
        self._last_meta = {
            "focal_id": focal_id,
            "other_id": other_id,
            "start_frame": int(start),
            "end_frame": int(start + n),
            "window_frames": int(window),
            "radius_px": int(radius),
            "radius_cm": float(radius / self._px_per_cm) if self._px_per_cm > 0 else np.nan,
            "bins": int(bins),
            "n_points": int(len(ego_x)),
            "fps": float(self._fps),
        }

        # Normalise to probability
        total = H.sum()
        if total > 0:
            H = H / total

        # Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=1.8)

        # Mask outside the circular boundary for a clean polar look
        cx_grid = np.linspace(-radius, radius, bins)
        cy_grid = np.linspace(-radius, radius, bins)
        xx, yy = np.meshgrid(cx_grid, cy_grid, indexing="ij")
        dist_grid = np.sqrt(xx**2 + yy**2)
        H[dist_grid > radius] = np.nan
        self._last_H = H.copy()

        # Colormap — jet-like for similarity to reference
        cmap = pg.colormap.get("CET-L8")  # perceptually uniform warm/cool
        lut = cmap.getLookupTable(nPts=256)

        h_max = np.nanmax(H)
        self._last_meta["max_density"] = float(h_max)
        if h_max > 0:
            H_display = H / h_max
        else:
            H_display = np.zeros_like(H)

        # Replace NaN with -1 so LUT maps them to transparent
        H_u8 = np.where(np.isnan(H_display), 0, (H_display * 255).astype(np.uint8))

        # Make outside-circle pixels transparent by setting alpha
        rgba = lut[H_u8]  # (bins, bins, 4) or (bins, bins, 3)
        if rgba.ndim == 3 and rgba.shape[2] == 3:
            alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate([rgba, alpha], axis=2)
        rgba[dist_grid > radius] = [30, 30, 46, 0]  # transparent outside

        self._img_item.setImage(rgba, autoLevels=False)
        self._img_item.setRect(-radius, -radius, 2 * radius, 2 * radius)

        # Set view
        pad = radius * 1.15
        self._pw.setRange(xRange=(-pad, pad), yRange=(-pad, pad), padding=0)

        # Draw polar grid + SVG
        self._draw_polar_grid(radius)
        self._update_svg_position(radius)
        self._update_colorbar(h_max)

        self._lbl_info.setText(
            f"{len(ego_x)} pts | frames {start}\u2013{start + n} "
            f"| max density {h_max:.5f}"
        )

    # ── Polar grid annotations ───────────────────────────────────────────────

    # Export
    def _clear_export_cache(self) -> None:
        """Drop cached export data so failed recomputes cannot export stale results."""
        self._last_ego_x = None
        self._last_ego_y = None
        self._last_H = None
        self._last_edges = None
        self._last_meta = {}

    @Slot()
    def _export_data(self) -> None:
        """Export the latest egocentric coordinates and heatmap data."""
        if (
            self._last_ego_x is None
            or self._last_ego_y is None
            or self._last_H is None
            or self._last_edges is None
        ):
            self._lbl_info.setText("Nothing to export. Update the heatmap first.")
            return

        default_name = (
            f"egocentric_{self._last_meta.get('focal_id', 'A')}"
            f"_vs_{self._last_meta.get('other_id', 'B')}.npz"
        )
        out_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Egocentric Heatmap Data",
            str(Path.home() / default_name),
            "NumPy archive (*.npz);;CSV bundle (*.csv)",
        )
        if not out_path:
            return

        out_file = Path(out_path)
        try:
            if selected_filter.startswith("CSV") or out_file.suffix.lower() == ".csv":
                self._export_csv_bundle(out_file)
                self._lbl_info.setText(f"Exported CSV bundle: {out_file.stem}_*.csv")
            else:
                if out_file.suffix.lower() != ".npz":
                    out_file = out_file.with_suffix(".npz")
                self._export_npz(out_file)
                self._lbl_info.setText(f"Exported {out_file.name}")
        except Exception as exc:
            logger.exception("Egocentric heatmap export error: %s", exc)
            self._lbl_info.setText(f"Export failed: {exc}")

    def _export_npz(self, out_file: Path) -> None:
        """Export all cached arrays and metadata into a compressed NPZ archive."""
        payload = {
            "ego_x_px": self._last_ego_x,
            "ego_y_px": self._last_ego_y,
            "hist_density": self._last_H,
            "bin_edges_px": self._last_edges,
        }
        if self._px_per_cm > 0:
            payload["ego_x_cm"] = self._last_ego_x / self._px_per_cm
            payload["ego_y_cm"] = self._last_ego_y / self._px_per_cm
            payload["bin_edges_cm"] = self._last_edges / self._px_per_cm
        for key, value in self._last_meta.items():
            payload[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(out_file, **payload)

    def _export_csv_bundle(self, out_file: Path) -> None:
        """Export raw points, heatmap bins, and metadata as companion CSV files."""
        base = out_file.with_suffix("")
        points_path = base.with_name(f"{base.stem}_points.csv")
        heatmap_path = base.with_name(f"{base.stem}_heatmap.csv")
        meta_path = base.with_name(f"{base.stem}_meta.csv")

        headers = ["ego_x_px", "ego_y_px"]
        arrays = [self._last_ego_x, self._last_ego_y]
        if self._px_per_cm > 0:
            headers.extend(["ego_x_cm", "ego_y_cm"])
            arrays.extend([self._last_ego_x / self._px_per_cm, self._last_ego_y / self._px_per_cm])
        point_rows = np.column_stack(arrays)
        np.savetxt(
            points_path,
            point_rows,
            delimiter=",",
            header=",".join(headers),
            comments="",
        )

        centers = (self._last_edges[:-1] + self._last_edges[1:]) / 2
        grid_x, grid_y = np.meshgrid(centers, centers, indexing="ij")
        valid = np.isfinite(self._last_H)
        heat_headers = ["ego_x_center_px", "ego_y_center_px", "density"]
        heat_arrays = [grid_x[valid], grid_y[valid], self._last_H[valid]]
        if self._px_per_cm > 0:
            heat_headers = [
                "ego_x_center_px",
                "ego_y_center_px",
                "ego_x_center_cm",
                "ego_y_center_cm",
                "density",
            ]
            heat_arrays = [
                grid_x[valid],
                grid_y[valid],
                grid_x[valid] / self._px_per_cm,
                grid_y[valid] / self._px_per_cm,
                self._last_H[valid],
            ]
        heat_rows = np.column_stack(heat_arrays)
        np.savetxt(
            heatmap_path,
            heat_rows,
            delimiter=",",
            header=",".join(heat_headers),
            comments="",
        )

        with meta_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["key", "value"])
            for key, value in self._last_meta.items():
                writer.writerow([key, value])

    # Polar grid annotations
    def _draw_polar_grid(self, radius: float) -> None:
        """Draw concentric distance rings, radial lines, and angle labels."""
        vb = self._pw.getViewBox()

        # Remove old grid items
        for item in self._grid_items:
            try:
                vb.removeItem(item)
            except Exception:
                pass
        self._grid_items.clear()

        n_rings = self._spin_rings.value()
        ring_step = radius / n_rings

        # Concentric distance rings
        for i in range(1, n_rings + 1):
            r = ring_step * i
            theta = np.linspace(0, 2 * np.pi, 120)
            cx = r * np.cos(theta)
            cy = r * np.sin(theta)
            ring = pg.PlotCurveItem(cx, cy, pen=_GRID_PEN)
            vb.addItem(ring)
            self._grid_items.append(ring)

            # Distance label — place at ~45° (upper-right)
            lx = r * np.cos(np.radians(45))
            ly = r * np.sin(np.radians(45))
            if self._px_per_cm > 0:
                label_text = f"{r / self._px_per_cm:.1f} cm"
            else:
                label_text = f"{int(r)} px"
            label = pg.TextItem(label_text, color=_GRID_TEXT_COLOR, anchor=(0, 1))
            label.setFont(_grid_font())
            label.setPos(lx, ly)
            vb.addItem(label)
            self._grid_items.append(label)

        # Radial lines + angle labels at 0°, 90°, 180°, 270°
        # Convention: 0° = ahead (top), 90° = right, 180° = behind, 270° = left
        angle_labels = {
            0: "0° (ahead)",
            90: "90°",
            180: "180° (behind)",
            270: "270°",
        }
        for angle_deg, label_text in angle_labels.items():
            # In our coordinate system: 0° is +Y (up), 90° is +X (right)
            theta = np.radians(90 - angle_deg)  # convert to math convention
            ex = radius * np.cos(theta)
            ey = radius * np.sin(theta)
            line = pg.PlotCurveItem(
                [0, ex], [0, ey],
                pen=pg.mkPen(color=(200, 200, 220, 40), width=1, style=Qt.PenStyle.DotLine),
            )
            vb.addItem(line)
            self._grid_items.append(line)

            # Angle label at edge
            margin = 1.08
            lx = ex * margin
            ly = ey * margin
            anchor = _angle_anchor(angle_deg)
            txt = pg.TextItem(label_text, color=_ANGLE_TEXT_COLOR, anchor=anchor)
            txt.setFont(_grid_font(bold=True))
            txt.setPos(lx, ly)
            vb.addItem(txt)
            self._grid_items.append(txt)

        # Additional radial ticks at 45°, 135°, 225°, 315°
        for angle_deg in (45, 135, 225, 315):
            theta = np.radians(90 - angle_deg)
            ex = radius * np.cos(theta)
            ey = radius * np.sin(theta)
            line = pg.PlotCurveItem(
                [0, ex], [0, ey],
                pen=pg.mkPen(color=(200, 200, 220, 25), width=1, style=Qt.PenStyle.DotLine),
            )
            vb.addItem(line)
            self._grid_items.append(line)

            lx = ex * 1.06
            ly = ey * 1.06
            txt = pg.TextItem(f"{angle_deg}°", color=_GRID_TEXT_COLOR, anchor=(0.5, 0.5))
            txt.setFont(_grid_font())
            txt.setPos(lx, ly)
            vb.addItem(txt)
            self._grid_items.append(txt)

    # ── SVG overlay ──────────────────────────────────────────────────────────

    def _update_svg_position(self, radius: float) -> None:
        """Place the SVG pixmap at the centre of the plot."""
        if self._svg_w == 0:
            return

        vb = self._pw.getViewBox()

        # Remove old overlay
        if self._svg_item is not None:
            try:
                vb.removeItem(self._svg_item)
            except Exception:
                pass
            self._svg_item = None

        # Render SVG at high-res
        target_h = radius * 0.4
        aspect = self._svg_w / max(self._svg_h, 1)
        target_w = target_h * aspect

        render_h = max(int(target_h * 2.5), 10)
        render_w = max(int(render_h * aspect), 10)

        from PySide6.QtGui import QPainter, QImage
        from PySide6.QtCore import QRectF

        renderer = QSvgRenderer(str(_SVG_PATH))
        qimg = QImage(render_w, render_h, QImage.Format.Format_ARGB32_Premultiplied)
        qimg.fill(Qt.GlobalColor.transparent)
        painter = QPainter(qimg)
        renderer.render(painter, QRectF(0, 0, render_w, render_h))
        painter.end()

        qimg = _apply_alpha_cap(qimg, 130)
        pix = QPixmap.fromImage(qimg)

        self._svg_item = QGraphicsPixmapItem(pix)

        from PySide6.QtGui import QTransform
        scale_x = target_w / render_w
        scale_y = target_h / render_h
        t = QTransform()
        t.translate(-target_w / 2, -target_h / 2)
        t.scale(scale_x, -scale_y)  # flip Y for plot coords
        self._svg_item.setTransform(t)
        vb.addItem(self._svg_item)

    # ── Colorbar ─────────────────────────────────────────────────────────────

    def _update_colorbar(self, max_val: float) -> None:
        """Draw a horizontal gradient colorbar below the plot."""
        self._cbar_widget.clear()
        plot = self._cbar_widget.addPlot()
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMouseEnabled(False, False)
        plot.hideButtons()

        # Create gradient bar as an image
        n_px = 256
        bar = np.arange(n_px, dtype=np.uint8).reshape(1, n_px)
        cmap = pg.colormap.get("CET-L8")
        lut = cmap.getLookupTable(nPts=256)

        img = pg.ImageItem(bar)
        img.setLookupTable(lut)
        img.setLevels([0, 255])
        img.setRect(0, 0, 1, 0.3)
        plot.addItem(img)
        plot.setRange(xRange=(-0.05, 1.05), yRange=(-0.3, 0.6), padding=0)

        # Labels
        lbl_lo = pg.TextItem("0", color=_GRID_TEXT_COLOR, anchor=(0.5, 0))
        lbl_lo.setFont(_grid_font())
        lbl_lo.setPos(0, -0.15)
        plot.addItem(lbl_lo)

        lbl_hi = pg.TextItem(f"{max_val:.4f}", color=_GRID_TEXT_COLOR, anchor=(0.5, 0))
        lbl_hi.setFont(_grid_font())
        lbl_hi.setPos(1, -0.15)
        plot.addItem(lbl_hi)

        lbl_mid = pg.TextItem("Position probability", color=_ANGLE_TEXT_COLOR, anchor=(0.5, 1))
        lbl_mid.setFont(_grid_font(bold=True))
        lbl_mid.setPos(0.5, 0.55)
        plot.addItem(lbl_mid)


# ── Helpers ──────────────────────────────────────────────────────────────────

_NOSE_NAMES = {"nose", "Nose", "snout", "Snout"}
_TAIL_NAMES = {
    "tail", "Tail", "tailbase", "Tailbase", "tail_base", "Tail_base",
    "tail_tip",
}


def _find_nose_tail(bodyparts: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Find nose and tail bodypart names from available bodyparts."""
    nose = None
    tail = None
    bp_lower = {bp.lower(): bp for bp in bodyparts}

    for name in _NOSE_NAMES:
        if name in bodyparts:
            nose = name
            break
    if nose is None:
        actual = bp_lower.get("nose")
        if actual:
            nose = actual

    for name in _TAIL_NAMES:
        if name in bodyparts:
            tail = name
            break
    if tail is None:
        for candidate in ("tailbase", "tail_base", "tail"):
            actual = bp_lower.get(candidate)
            if actual:
                tail = actual
                break

    return nose, tail


def _apply_alpha_cap(qimg: "QImage", max_alpha: int) -> "QImage":
    """Cap the alpha channel of a QImage using numpy (fast, no pixel loop)."""
    from PySide6.QtGui import QImage

    w, h = qimg.width(), qimg.height()
    bits = qimg.bits()
    arr = np.frombuffer(bits, dtype=np.uint8).reshape((h, w, 4)).copy()
    alpha = arr[:, :, 3]
    arr[:, :, 3] = np.minimum(alpha, max_alpha)
    result = QImage(arr.data, w, h, w * 4, QImage.Format.Format_ARGB32_Premultiplied)
    return result.copy()


def _grid_font(bold: bool = False):
    """Return a small font for grid annotations."""
    from PySide6.QtGui import QFont
    f = QFont("Segoe UI", 8)
    f.setBold(bold)
    return f


def _angle_anchor(angle_deg: int) -> tuple[float, float]:
    """Return TextItem anchor for angle labels placed outside the ring."""
    if angle_deg == 0:
        return (0.5, 1.0)   # top centre — anchor bottom
    elif angle_deg == 90:
        return (0.0, 0.5)   # right — anchor left
    elif angle_deg == 180:
        return (0.5, 0.0)   # bottom — anchor top
    elif angle_deg == 270:
        return (1.0, 0.5)   # left — anchor right
    return (0.5, 0.5)
