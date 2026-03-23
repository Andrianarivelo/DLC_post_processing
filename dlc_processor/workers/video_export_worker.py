"""Background worker for overlay video export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class VideoExportWorker(QThread):
    """Render and write an overlay video while reporting progress."""

    progress = Signal(int)
    status = Signal(str)
    completed = Signal(str, str)  # output_path, error

    def __init__(
        self,
        video_path: str,
        output_path: str,
        animal_dfs: dict,
        behavior_arrays: Optional[dict],
        fps: int,
        draw_skeleton: bool,
        draw_labels: bool,
        draw_behaviors: bool,
        skeleton_edges: Optional[list[tuple[str, str]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.output_path = output_path
        self.animal_dfs = animal_dfs
        self.behavior_arrays = behavior_arrays or {}
        self.fps = fps
        self.draw_skeleton = draw_skeleton
        self.draw_labels = draw_labels
        self.draw_behaviors = draw_behaviors
        self.skeleton_edges = skeleton_edges
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        from dlc_processor.workers.overlay_worker import OverlayWorker, _SKELETON_EDGES

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.completed.emit("", "Cannot open source video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            self.completed.emit("", "Cannot create output video writer.")
            return

        overlay = OverlayWorker.__new__(OverlayWorker)
        overlay.animal_dfs = self.animal_dfs
        overlay.behavior_arrays = self.behavior_arrays if self.draw_behaviors else {}
        overlay.draw_skeleton = self.draw_skeleton
        overlay.draw_labels = self.draw_labels
        overlay.draw_behaviors = self.draw_behaviors
        overlay.skeleton_edges = self.skeleton_edges if self.skeleton_edges else list(_SKELETON_EDGES)
        overlay.fill_body = False

        animals = list(self.animal_dfs.keys())
        last_pct = -1

        try:
            for fi in range(total_frames):
                if self._abort:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    rendered = overlay._render(frame, fi, animals)
                except Exception:
                    rendered = frame

                writer.write(rendered)

                pct = int(round(100.0 * (fi + 1) / max(total_frames, 1)))
                if pct != last_pct:
                    last_pct = pct
                    self.progress.emit(pct)
                    self.status.emit(
                        f"Exporting… {fi + 1}/{total_frames} frames ({pct}%)"
                    )
        finally:
            cap.release()
            writer.release()

        if self._abort:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.completed.emit("", "Video export cancelled.")
            return

        self.progress.emit(100)
        self.completed.emit(str(out_path), "")
