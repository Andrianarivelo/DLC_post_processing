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
        start_frame: int = 0,
        end_frame: Optional[int] = None,
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
        self.start_frame = max(0, int(start_frame or 0))
        self.end_frame = None if end_frame is None or int(end_frame or 0) <= 0 else int(end_frame)
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
        export_frames = _export_source_frames(
            self.animal_dfs,
            total_frames,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )
        if not export_frames:
            cap.release()
            self.completed.emit("", "No video frames fall inside the requested export window.")
            return

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
        overlay.frame_index_mode = "video"

        animals = list(self.animal_dfs.keys())
        last_pct = -1

        try:
            for out_idx, source_frame in enumerate(export_frames):
                if self._abort:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame))
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    rendered = overlay._render(frame, int(source_frame), animals)
                except Exception:
                    rendered = frame

                writer.write(rendered)

                pct = int(round(100.0 * (out_idx + 1) / max(len(export_frames), 1)))
                if pct != last_pct:
                    last_pct = pct
                    self.progress.emit(pct)
                    self.status.emit(
                        f"Exporting... {out_idx + 1}/{len(export_frames)} frames ({pct}%)"
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


def _export_source_frames(
    animal_dfs: dict,
    total_frames: int,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> list[int]:
    start = max(0, int(start_frame or 0))
    end = int(end_frame) if end_frame is not None and int(end_frame) > 0 else max(int(total_frames) - 1, -1)
    frame_numbers = _frame_numbers_from_dfs(animal_dfs)
    if frame_numbers is None or len(frame_numbers) == 0:
        return [idx for idx in range(max(int(total_frames), 0)) if start <= idx <= end]
    return [int(v) for v in frame_numbers if 0 <= int(v) < total_frames and start <= int(v) <= end]


def _frame_numbers_from_dfs(animal_dfs: dict) -> Optional[np.ndarray]:
    if not animal_dfs:
        return None
    first_df = next(iter(animal_dfs.values()), None)
    if first_df is None:
        return None
    frames = getattr(first_df, "attrs", {}).get("frame_numbers")
    if frames is None:
        return None
    arr = np.asarray(frames, dtype=np.int64).reshape(-1)
    if len(arr) != len(first_df):
        return None
    return arr
