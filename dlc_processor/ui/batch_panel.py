"""Batch computation, metadata, and grouped summary panel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.ui_kit import COLORS, Card, hint

# Form alignment + calm field widths so combos / numeric inputs sit in a tidy
# right-aligned column instead of stretching across the whole card.
_FORM_LABEL_ALIGN = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
_FORM_ALIGN = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
_FIELD_WIDTH = 150
_NUM_WIDTH = 120


def _make_form() -> QFormLayout:
    """Build a consistently styled, right-aligned form layout."""
    form = QFormLayout()
    form.setSpacing(8)
    form.setHorizontalSpacing(12)
    form.setLabelAlignment(_FORM_LABEL_ALIGN)
    form.setFormAlignment(_FORM_ALIGN)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    return form


def _fixed(widget: QWidget, width: int = _FIELD_WIDTH) -> QWidget:
    """Constrain a control to a fixed, premium-looking column width."""
    widget.setFixedWidth(width)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


_META_COLUMNS = [
    "recording",
    "animal",
    "mouseId",
    "condition",
    "genotype",
    "group",
    "sex",
    "cohort",
    "notes",
]


class BatchPanel(QGroupBox):
    """Visible batch + metadata workflow for loaded DLC Processor projects."""

    batch_compute_requested = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Batch / Metadata", parent)
        self._records: list[dict] = []
        self._metadata_by_key: dict[str, dict] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Metadata table + import/export ─────────────────────────────────
        meta_card = Card(
            "Recording metadata",
            "Edit subject / group fields used in exported tables and grouped summaries",
            accent=COLORS["teal"],
        )

        self._table = QTableWidget(0, len(_META_COLUMNS))
        self._table.setHorizontalHeaderLabels([_metadata_header(c) for c in _META_COLUMNS])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(180)
        self._table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        meta_card.body.addWidget(self._table)

        meta_btns = QHBoxLayout()
        meta_btns.setSpacing(8)
        btn_refresh = QPushButton("Refresh Loaded Rows")
        btn_refresh.setObjectName("secondary")
        btn_refresh.clicked.connect(lambda: self.set_records(self._records))
        meta_btns.addWidget(btn_refresh)
        btn_import = QPushButton("Import Metadata CSV")
        btn_import.setObjectName("secondary")
        btn_import.clicked.connect(self._import_metadata)
        meta_btns.addWidget(btn_import)
        btn_export = QPushButton("Export Metadata CSV")
        btn_export.setObjectName("secondary")
        btn_export.clicked.connect(self._export_metadata)
        meta_btns.addWidget(btn_export)
        meta_btns.addStretch()
        meta_card.body.addLayout(meta_btns)
        meta_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(meta_card, 1)

        # ── Processing options ─────────────────────────────────────────────
        opts_card = Card("Processing options", "What to compute for every loaded recording")

        self._chk_social = QCheckBox("Compute social behaviours")
        self._chk_social.setChecked(True)
        opts_card.body.addWidget(self._chk_social)
        self._chk_social_masks = QCheckBox("Use masks for social contact")
        self._chk_social_masks.setChecked(False)
        self._chk_social_masks.setToolTip("Use segmentation masks during social behaviour export; slower on long videos")
        opts_card.body.addWidget(self._chk_social_masks)
        self._chk_fix_mask_identity = QCheckBox("Fix identities from masks first")
        self._chk_fix_mask_identity.setChecked(False)
        self._chk_fix_mask_identity.setToolTip(
            "Before computing metrics, use paired COCO mask track IDs as identity ground truth for every loaded recording"
        )
        opts_card.body.addWidget(self._chk_fix_mask_identity)
        self._chk_summary = QCheckBox("Generate grouped summary tables + figures")
        self._chk_summary.setChecked(True)
        opts_card.body.addWidget(self._chk_summary)
        self._chk_position_maps = QCheckBox("Export averaged position maps")
        self._chk_position_maps.setChecked(True)
        opts_card.body.addWidget(self._chk_position_maps)

        jobs_form = _make_form()
        self._spin_jobs = QSpinBox()
        self._spin_jobs.setRange(0, max(1, int(os.cpu_count() or 1)))
        self._spin_jobs.setValue(0)
        self._spin_jobs.setSpecialValueText("Auto")
        self._spin_jobs.setToolTip(
            "Parallel CPU workers for batch metric/social export. Auto uses up to CPU count minus one."
        )
        _fixed(self._spin_jobs, _NUM_WIDTH)
        jobs_form.addRow("CPU jobs", self._spin_jobs)
        opts_card.body.addLayout(jobs_form)
        layout.addWidget(opts_card)

        # ── Frame window ───────────────────────────────────────────────────
        window_card = Card("Frame window", "Optionally restrict analysis to a range of source frames")

        self._chk_time_window = QCheckBox("Analyze frame window")
        self._chk_time_window.setToolTip("Restrict batch metrics, social behavior, transitions, and position maps to source frame numbers.")
        self._chk_time_window.toggled.connect(lambda _checked: self._update_ui_state())
        window_card.body.addWidget(self._chk_time_window)

        window_form = _make_form()
        self._spin_start_frame = QSpinBox()
        self._spin_start_frame.setRange(0, 2_000_000_000)
        self._spin_start_frame.setSingleStep(100)
        self._spin_start_frame.setToolTip("First source frame number to include.")
        _fixed(self._spin_start_frame, _NUM_WIDTH)
        window_form.addRow("Start frame", self._spin_start_frame)
        self._spin_end_frame = QSpinBox()
        self._spin_end_frame.setRange(0, 2_000_000_000)
        self._spin_end_frame.setSpecialValueText("To end")
        self._spin_end_frame.setSingleStep(100)
        self._spin_end_frame.setToolTip("Last source frame number to include. Leave at 0 to continue to the last tracked frame.")
        _fixed(self._spin_end_frame, _NUM_WIDTH)
        window_form.addRow("End frame", self._spin_end_frame)
        window_card.body.addLayout(window_form)
        layout.addWidget(window_card)

        # ── Grouped summary ────────────────────────────────────────────────
        summary_card = Card("Grouped summary", "How recordings are compared and plotted")

        summary_form = _make_form()
        self._combo_plot_style = QComboBox()
        self._combo_plot_style.addItem("Boxplot + stripplot", "box_strip")
        self._combo_plot_style.addItem("Bar + stripplot", "bar_strip")
        _fixed(self._combo_plot_style)
        summary_form.addRow("Summary plot", self._combo_plot_style)
        self._combo_comparison = QComboBox()
        self._combo_comparison.addItem("Holm multiple t-tests", "holm_ttest")
        self._combo_comparison.addItem("Tukey HSD", "tukey")
        _fixed(self._combo_comparison)
        summary_form.addRow("Stats", self._combo_comparison)
        self._combo_group = QComboBox()
        self._combo_group.addItem("Animal", "animal")
        self._combo_group.addItem("Mouse ID", "mouseId")
        self._combo_group.addItem("Condition", "condition")
        self._combo_group.addItem("Genotype", "genotype")
        self._combo_group.addItem("Condition + Genotype", "condition,genotype")
        self._combo_group.addItem("Mouse ID + Condition", "mouseId,condition")
        self._combo_group.addItem("Mouse ID + Genotype", "mouseId,genotype")
        self._combo_group.addItem("Group", "group")
        self._combo_group.addItem("Sex", "sex")
        self._combo_group.addItem("Cohort", "cohort")
        _fixed(self._combo_group, _FIELD_WIDTH + 30)
        summary_form.addRow("Compare by", self._combo_group)
        self._color_map = QLineEdit()
        self._color_map.setPlaceholderText("e.g. WT=0, Het=1, control=2")
        self._color_map.setToolTip("Map group labels to tab10 colour indices")
        summary_form.addRow("Tab10 colors", self._color_map)
        self._animal_filter = QLineEdit()
        self._animal_filter.setPlaceholderText("All labels, e.g. mouse1")
        summary_form.addRow("Focus animal label", self._animal_filter)
        self._mouse_id_filter = QLineEdit()
        self._mouse_id_filter.setPlaceholderText("All IDs, e.g. 31098")
        summary_form.addRow("Focus mouseId", self._mouse_id_filter)
        summary_card.body.addLayout(summary_form)
        layout.addWidget(summary_card)

        # ── Output destination ─────────────────────────────────────────────
        out_card = Card("Output destination", "Where per-video tables and summaries are written")

        out_form = _make_form()
        self._combo_output = QComboBox()
        self._combo_output.addItem("Frame-time folders", "time_file_folder")
        self._combo_output.addItem("DLC folders", "dlc_file_folder")
        self._combo_output.addItem("Video folders", "video_file_folder")
        self._combo_output.addItem("Custom folder", "custom")
        self._combo_output.currentIndexChanged.connect(lambda _idx: self._update_ui_state())
        _fixed(self._combo_output, _FIELD_WIDTH + 30)
        out_form.addRow("Output", self._combo_output)
        out_card.body.addLayout(out_form)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self._custom_dir = QLineEdit()
        self._custom_dir.setPlaceholderText("Custom output folder...")
        self._custom_dir.textChanged.connect(lambda _text: self._update_ui_state())
        out_row.addWidget(self._custom_dir, 1)
        btn_dir = QPushButton("Browse")
        btn_dir.setObjectName("secondary")
        btn_dir.clicked.connect(self._pick_output_dir)
        out_row.addWidget(btn_dir)
        self._btn_open_output = QPushButton("Open")
        self._btn_open_output.setObjectName("secondary")
        self._btn_open_output.setToolTip("Open the selected batch output folder")
        self._btn_open_output.clicked.connect(self._open_output_dir)
        out_row.addWidget(self._btn_open_output)
        out_card.body.addLayout(out_row)
        layout.addWidget(out_card)

        # ── Run + progress ─────────────────────────────────────────────────
        run_card = Card("Run batch", accent=COLORS["accent"])

        self._btn_compute = QPushButton("Process All Loaded Videos")
        self._btn_compute.setToolTip(
            "Compute behavior metrics and social behaviours for every loaded recording row, then export per-video tables"
        )
        self._btn_compute.clicked.connect(self._request_batch)
        run_card.body.addWidget(self._btn_compute)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        run_card.body.addWidget(self._progress)

        self._status = hint("Load recordings to enable batch compute.")
        run_card.body.addWidget(self._status)
        layout.addWidget(run_card)

        self._update_ui_state()

    def set_records(self, records: list[dict], *, capture_existing: bool = True) -> None:
        if capture_existing:
            self._capture_table()
        self._records = list(records or [])
        rows: list[dict] = []
        for rec in self._records:
            animals = _animals_for_record(rec) or self._metadata_animals_for_record(rec)
            if not animals:
                animals = [""]
            recording = _recording_name(rec)
            for animal in animals:
                row = {
                    "recording": recording,
                    "animal": animal,
                    "mouseId": _default_mouse_id(recording, animal),
                    "condition": "",
                    "genotype": "",
                    "group": "",
                    "sex": "",
                    "cohort": "",
                    "notes": "",
                }
                row.update(_normalize_metadata_row(self._metadata_for_record(rec, animal), recording, animal))
                rows.append(row)

        self._table.blockSignals(True)
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, name in enumerate(_META_COLUMNS):
                item = QTableWidgetItem(str(row.get(name, "") or ""))
                if name in {"recording", "animal"}:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)
        self._table.blockSignals(False)
        self._update_ui_state()

    def _metadata_for_record(self, record: dict, animal: str) -> dict:
        recording = _recording_name(record)
        exact = self._metadata_by_key.get(f"{recording}|{animal}")
        if exact:
            return exact
        rec_aliases = _recording_aliases(recording)
        for path_key in ("dlc_path", "video_path", "time_path", "mask_path"):
            path = str(record.get(path_key, "") or "")
            if path:
                rec_aliases.update(_recording_aliases(Path(path).stem))
        animal_key = str(animal or "").strip().casefold()
        for row in self._metadata_by_key.values():
            if str(row.get("animal", "") or "").strip().casefold() != animal_key:
                continue
            row_recording = str(row.get("recording", "") or "")
            if rec_aliases & _recording_aliases(row_recording):
                return row
        return {}

    def _metadata_animals_for_record(self, record: dict) -> list[str]:
        recording = _recording_name(record)
        rec_aliases = _recording_aliases(recording)
        for path_key in ("dlc_path", "video_path", "time_path", "mask_path"):
            path = str(record.get(path_key, "") or "")
            if path:
                rec_aliases.update(_recording_aliases(Path(path).stem))
        animals: list[str] = []
        seen: set[str] = set()
        for row in self._metadata_by_key.values():
            if not (rec_aliases & _recording_aliases(str(row.get("recording", "") or ""))):
                continue
            animal = str(row.get("animal", "") or "").strip()
            if not animal:
                continue
            key = animal.casefold()
            if key in seen:
                continue
            seen.add(key)
            animals.append(animal)
        return animals

    def restore_metadata_rows(self, rows: list[dict]) -> None:
        for row in rows or []:
            coerced = _coerce_metadata_row(row)
            recording = str(coerced.get("recording", "") or "")
            animal = str(coerced.get("animal", "") or "")
            key = f"{recording}|{animal}"
            self._metadata_by_key[key] = _normalize_metadata_row(coerced, recording, animal)
        self.set_records(self._records, capture_existing=False)

    def metadata_rows(self) -> list[dict]:
        self._capture_table()
        rows = list(self._metadata_by_key.values())
        return sorted(
            rows,
            key=lambda row: (
                _recording_sort_key(str(row.get("recording", "") or "")),
                str(row.get("animal", "") or "").casefold(),
            ),
        )

    def attach_metadata_to_records(self, records: list[dict]) -> list[dict]:
        self._capture_table()
        out_records: list[dict] = []
        for rec in records:
            copied = dict(rec)
            animals = _animals_for_record(rec) or self._metadata_animals_for_record(rec)
            animal_metadata: dict[str, dict] = {}
            for animal in animals:
                row = self._metadata_for_record(rec, animal)
                if row:
                    animal_metadata[animal] = {
                        key: value for key, value in row.items()
                        if key not in {"recording", "animal"} and str(value)
                    }
            copied["animal_metadata"] = animal_metadata
            copied["metadata"] = _common_record_metadata(animal_metadata)
            out_records.append(copied)
        return out_records

    def import_metadata_csv(self, path: str | Path) -> int:
        df = pd.read_csv(path, dtype=str).fillna("")
        rows = [_coerce_metadata_row(row) for row in df.to_dict("records")]
        self.restore_metadata_rows(rows)
        return len(rows)

    def export_state(self) -> dict:
        return {
            "batch_export_output_mode": self._combo_output.currentData() or "time_file_folder",
            "batch_export_custom_dir": self._custom_dir.text().strip(),
            "batch_summary_group_by": self._combo_group.currentData() or "animal",
            "batch_summary_animal_filter": self._animal_filter.text().strip(),
            "batch_summary_mouse_id_filter": self._mouse_id_filter.text().strip(),
            "batch_summary_plot_style": self._combo_plot_style.currentData() or "box_strip",
            "batch_summary_comparison": self._combo_comparison.currentData() or "holm_ttest",
            "batch_summary_color_map": self._color_map.text().strip(),
            "batch_export_position_maps": self._chk_position_maps.isChecked(),
            "batch_compute_social": self._chk_social.isChecked(),
            "batch_social_use_masks": self._chk_social_masks.isChecked(),
            "batch_fix_identity_from_masks": self._chk_fix_mask_identity.isChecked(),
            "batch_generate_summary": self._chk_summary.isChecked(),
            "batch_cpu_jobs": int(self._spin_jobs.value()),
            "batch_analysis_frame_window_enabled": self._chk_time_window.isChecked(),
            "batch_analysis_start_frame": int(self._spin_start_frame.value()),
            "batch_analysis_end_frame": int(self._spin_end_frame.value()),
        }

    def restore_state(self, settings: dict) -> None:
        mode = settings.get("batch_export_output_mode", "time_file_folder")
        idx = self._combo_output.findData(mode)
        if idx >= 0:
            self._combo_output.setCurrentIndex(idx)
        self._custom_dir.setText(str(settings.get("batch_export_custom_dir", "") or ""))
        group_by = settings.get("batch_summary_group_by", "animal")
        idx = self._combo_group.findData(group_by)
        if idx >= 0:
            self._combo_group.setCurrentIndex(idx)
        self._animal_filter.setText(str(settings.get("batch_summary_animal_filter", "") or ""))
        self._mouse_id_filter.setText(str(settings.get("batch_summary_mouse_id_filter", "") or ""))
        style = settings.get("batch_summary_plot_style", "box_strip")
        if style == "bar_strip":
            style = "box_strip"
        idx = self._combo_plot_style.findData(style)
        if idx >= 0:
            self._combo_plot_style.setCurrentIndex(idx)
        comparison = settings.get("batch_summary_comparison", "holm_ttest")
        idx = self._combo_comparison.findData(comparison)
        if idx >= 0:
            self._combo_comparison.setCurrentIndex(idx)
        self._color_map.setText(str(settings.get("batch_summary_color_map", "") or ""))
        self._chk_position_maps.setChecked(bool(settings.get("batch_export_position_maps", True)))
        self._chk_social.setChecked(bool(settings.get("batch_compute_social", True)))
        self._chk_social_masks.setChecked(bool(settings.get("batch_social_use_masks", False)))
        self._chk_fix_mask_identity.setChecked(bool(settings.get("batch_fix_identity_from_masks", False)))
        self._chk_summary.setChecked(bool(settings.get("batch_generate_summary", True)))
        self._spin_jobs.setValue(int(settings.get("batch_cpu_jobs", 0) or 0))
        self._chk_time_window.setChecked(bool(settings.get("batch_analysis_frame_window_enabled", False)))
        self._spin_start_frame.setValue(int(settings.get("batch_analysis_start_frame", 0) or 0))
        self._spin_end_frame.setValue(int(settings.get("batch_analysis_end_frame", 0) or 0))
        self._update_ui_state()

    def set_running(self, running: bool) -> None:
        self._progress.setVisible(running)
        if running:
            self._progress.setValue(0)
        self._spin_jobs.setEnabled(not running)
        self._chk_fix_mask_identity.setEnabled(not running)
        self._chk_time_window.setEnabled(not running)
        self._spin_start_frame.setEnabled(not running and self._chk_time_window.isChecked())
        self._spin_end_frame.setEnabled(not running and self._chk_time_window.isChecked())
        self._btn_compute.setEnabled(not running and self._has_records_to_compute())

    def set_progress(self, pct: int, text: str) -> None:
        self._progress.setVisible(True)
        self._progress.setValue(max(0, min(100, int(pct))))
        self._status.setText(text)

    def set_result(self, text: str) -> None:
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._status.setText(text)
        self._update_ui_state()

    def _request_batch(self) -> None:
        self._capture_table()
        self.batch_compute_requested.emit({
            "compute_social": self._chk_social.isChecked(),
            "use_masks_for_social": self._chk_social_masks.isChecked(),
            "fix_identity_from_masks": self._chk_fix_mask_identity.isChecked(),
            "generate_summary": self._chk_summary.isChecked(),
            "output_mode": self._combo_output.currentData() or "time_file_folder",
            "custom_output_dir": self._custom_dir.text().strip(),
            "group_by": str(self._combo_group.currentData() or "animal").split(","),
            "animal_filter": self._animal_filter.text().strip(),
            "mouse_id_filter": self._mouse_id_filter.text().strip(),
            "plot_style": self._combo_plot_style.currentData() or "box_strip",
            "comparison_method": self._combo_comparison.currentData() or "holm_ttest",
            "tab10_color_map": self._color_map.text().strip(),
            "export_position_maps": self._chk_position_maps.isChecked(),
            "n_jobs": int(self._spin_jobs.value()),
            "analysis_frame_window_enabled": self._chk_time_window.isChecked(),
            "analysis_start_frame": int(self._spin_start_frame.value()),
            "analysis_end_frame": int(self._spin_end_frame.value()),
        })

    def _capture_table(self) -> None:
        for r in range(self._table.rowCount()):
            row = {}
            for c, name in enumerate(_META_COLUMNS):
                item = self._table.item(r, c)
                row[name] = item.text().strip() if item else ""
            key = f"{row.get('recording', '')}|{row.get('animal', '')}"
            if key.strip("|"):
                self._metadata_by_key[key] = row

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Batch Output Folder",
            self._custom_dir.text().strip() or str(Path.home()),
        )
        if path:
            self._custom_dir.setText(path)
            idx = self._combo_output.findData("custom")
            if idx >= 0:
                self._combo_output.setCurrentIndex(idx)
        self._update_ui_state()

    def _open_output_dir(self) -> None:
        folder = self._current_output_dir()
        if folder is None:
            self._status.setText("Choose or generate an output folder first.")
            return
        if not folder.exists() or not folder.is_dir():
            self._status.setText(f"Output folder does not exist yet: {folder}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._status.setText(f"Could not open output folder: {folder}")

    def _current_output_dir(self) -> Path | None:
        mode = self._combo_output.currentData() or "time_file_folder"
        custom = self._custom_dir.text().strip()
        if mode == "custom":
            return Path(custom) if custom else None
        for rec in self._records:
            if mode == "dlc_file_folder" and rec.get("dlc_path"):
                return Path(str(rec["dlc_path"])).parent
            if mode == "video_file_folder" and rec.get("video_path"):
                return Path(str(rec["video_path"])).parent
            if rec.get("time_path"):
                return Path(str(rec["time_path"])).parent
            if rec.get("dlc_path"):
                return Path(str(rec["dlc_path"])).parent
            if rec.get("video_path"):
                return Path(str(rec["video_path"])).parent
        return Path(custom) if custom else None

    def _import_metadata(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Metadata CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            count = self.import_metadata_csv(path)
        except Exception as exc:
            self._status.setText(f"Metadata import failed: {exc}")
            return
        self._status.setText(f"Imported metadata: {Path(path).name} ({count} rows)")

    def _export_metadata(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metadata CSV",
            str(Path.home() / "dlc_metadata.csv"),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        rows = self.metadata_rows()
        pd.DataFrame(rows, columns=_META_COLUMNS).to_csv(path, index=False)
        self._status.setText(f"Metadata exported: {Path(path).name}")

    def _has_records_to_compute(self) -> bool:
        return any(rec.get("dlc_path") or rec.get("animal_dfs") for rec in self._records)

    def _update_ui_state(self) -> None:
        has_records = self._has_records_to_compute()
        custom_needed = self._combo_output.currentData() == "custom"
        self._custom_dir.setEnabled(custom_needed)
        time_window = self._chk_time_window.isChecked()
        self._spin_start_frame.setEnabled(time_window)
        self._spin_end_frame.setEnabled(time_window)
        self._btn_compute.setEnabled(has_records and (not custom_needed or bool(self._custom_dir.text().strip())))
        out_dir = self._current_output_dir()
        self._btn_open_output.setEnabled(out_dir is not None and out_dir.exists() and out_dir.is_dir())
        if not has_records:
            self._status.setText("Load recordings to enable batch compute.")
        elif not self._progress.isVisible():
            self._status.setText(
                f"{len(self._records)} loaded row(s). Edit metadata, then process all loaded videos."
            )


def _recording_name(record: dict) -> str:
    for key in ("dlc_path", "video_path", "time_path"):
        path = record.get(key, "")
        if path:
            return Path(path).stem
    return f"recording_{int(record.get('index', 0)) + 1:03d}"


def _animals_for_record(record: dict) -> list[str]:
    dfs = record.get("animal_dfs") or {}
    if dfs:
        return [str(aid) for aid in dfs.keys()]
    return [str(aid) for aid in (record.get("animal_ids") or []) if str(aid)]


def _metadata_key(record: dict, animal: str) -> str:
    return f"{_recording_name(record)}|{animal}"


def _metadata_header(column: str) -> str:
    return "Mouse ID" if column == "mouseId" else column.replace("_", " ").title()


def _default_mouse_id(recording: str, animal: str) -> str:
    """Infer the biological subject ID from the recording name when possible.

    In this lab convention the filename starts with the experimental animal ID,
    and that subject is tracked as the local ``mouse1`` identity in each file.
    ``mouse2`` remains editable because its biological ID is not encoded in the
    recording name.
    """
    animal_text = str(animal or "").strip().lower()
    if animal_text not in {"mouse1", "mouse_1", "1"}:
        return ""
    first = str(recording or "").strip().split("_", 1)[0]
    return first if first else ""


def _coerce_metadata_row(row: dict | None) -> dict:
    source = {str(k).strip(): v for k, v in dict(row or {}).items()}
    aliases = {
        "recording": ("recording", "recording_name", "file", "file_name", "filename", "session", "trial"),
        "animal": ("animal", "animal_id", "track", "identity", "individual", "individuals"),
        "mouseId": ("mouseId", "mouseid", "mouse_id", "mouse", "subject", "subject_id", "animalid"),
        "condition": ("condition", "treatment", "group_condition"),
        "genotype": ("genotype", "geno"),
        "group": ("group", "experimental_group"),
        "sex": ("sex",),
        "cohort": ("cohort",),
        "notes": ("notes", "note", "comment", "comments"),
    }
    lower = {key.lower(): key for key in source}
    out: dict[str, str] = {}
    for target, candidates in aliases.items():
        value = ""
        for candidate in candidates:
            source_key = lower.get(candidate.lower())
            if source_key is not None:
                value = str(source.get(source_key, "") or "")
                break
        out[target] = value
    return out


def _recording_aliases(recording: str) -> set[str]:
    text = str(recording or "").strip()
    if not text:
        return {""}
    stems = {text, Path(text).stem}
    aliases: set[str] = set()
    for stem in stems:
        value = _normalize_recording_token(stem)
        if not value:
            continue
        aliases.add(value)
        current = value
        changed = True
        while changed:
            changed = False
            for suffix in (
                "_metrics_behavior",
                "_tracking",
                "_filtered",
                "_analysis",
                "_results",
                "_data",
            ):
                if current.endswith(suffix):
                    current = current[: -len(suffix)]
                    aliases.add(current)
                    changed = True
        if "_start_" in current:
            aliases.add(current.split("_start_", 1)[0])
    return aliases or {text.casefold()}


def _normalize_recording_token(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = Path(text).stem if "/" in text else text
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _recording_sort_key(recording: str) -> tuple[str, str]:
    aliases = sorted(_recording_aliases(recording))
    return (aliases[0] if aliases else "", str(recording or "").casefold())


def _normalize_metadata_row(row: dict | None, recording: str, animal: str) -> dict:
    source = _coerce_metadata_row(row)
    normalized = {col: str(source.get(col, "") or "") for col in _META_COLUMNS}
    normalized["recording"] = str(source.get("recording", recording) or recording)
    normalized["animal"] = str(source.get("animal", animal) or animal)
    if not normalized.get("mouseId"):
        normalized["mouseId"] = _default_mouse_id(recording, animal)
    return normalized


def _common_record_metadata(animal_metadata: dict[str, dict]) -> dict:
    if not animal_metadata:
        return {}
    keys = set().union(*(meta.keys() for meta in animal_metadata.values()))
    keys.discard("mouseId")
    common: dict[str, str] = {}
    for key in keys:
        values = sorted({str(meta.get(key, "") or "") for meta in animal_metadata.values() if str(meta.get(key, "") or "")})
        if len(values) == 1:
            common[key] = values[0]
        elif len(values) > 1:
            common[key] = " | ".join(values)
    return common
