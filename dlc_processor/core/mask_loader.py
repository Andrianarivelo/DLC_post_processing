"""Mask loading helpers for DLC Processor.

Supports the COCO RLE JSON written by MouseTracker/LISBET inference exports.
The store indexes annotations by source-video frame number and decodes masks
only for the frame being displayed.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskAnnotation:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    segmentation: Any
    size: tuple[int, int]  # height, width


class CocoMaskStore:
    """Lazy COCO mask index keyed by absolute source-video frame number."""

    def __init__(
        self,
        path: Path,
        frames: dict[int, list[MaskAnnotation]],
        info: Optional[dict] = None,
        cache_size: int = 12,
        payload: Optional[dict] = None,
        image_id_to_frame: Optional[dict[int, int]] = None,
    ) -> None:
        self.path = Path(path)
        self.frames = frames
        self.info = info or {}
        self.cache_size = max(1, int(cache_size))
        self._payload = deepcopy(payload) if isinstance(payload, dict) else None
        self._image_id_to_frame = dict(image_id_to_frame or {})
        self._cache: OrderedDict[int, list[tuple[int, np.ndarray, tuple[float, float, float, float], float]]] = OrderedDict()
        self._annotation_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    @classmethod
    def from_file(cls, path: str | Path) -> "CocoMaskStore":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "annotations" not in payload:
            raise ValueError(f"Not a COCO mask JSON: {p}")

        image_id_to_frame: dict[int, int] = {}
        image_id_to_size: dict[int, tuple[int, int]] = {}
        for image in payload.get("images", []) or []:
            if not isinstance(image, dict):
                continue
            image_id = int(image.get("id", image.get("frame_index", 0)))
            frame_idx = int(image.get("frame_index", image_id))
            height = int(image.get("height", 0) or 0)
            width = int(image.get("width", 0) or 0)
            image_id_to_frame[image_id] = frame_idx
            if height > 0 and width > 0:
                image_id_to_size[image_id] = (height, width)

        frames: dict[int, list[MaskAnnotation]] = {}
        for ann in payload.get("annotations", []) or []:
            if not isinstance(ann, dict):
                continue
            image_id = int(ann.get("image_id", 0))
            frame_idx = image_id_to_frame.get(image_id, image_id)
            seg = ann.get("segmentation")
            size = image_id_to_size.get(image_id)
            if size is None and isinstance(seg, dict):
                raw_size = seg.get("size")
                if isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
                    size = (int(raw_size[0]), int(raw_size[1]))
            if size is None:
                continue
            bbox_raw = ann.get("bbox", [0, 0, 0, 0])
            bbox = tuple(float(v) for v in list(bbox_raw)[:4])
            if len(bbox) != 4:
                bbox = (0.0, 0.0, 0.0, 0.0)
            frames.setdefault(frame_idx, []).append(
                MaskAnnotation(
                    track_id=int(ann.get("track_id", ann.get("category_id", 0)) or 0),
                    bbox=bbox,
                    score=float(ann.get("score", 1.0) or 0.0),
                    segmentation=seg,
                    size=size,
                )
            )

        return cls(
            path=p,
            frames=frames,
            info=payload.get("info") or {},
            payload=payload,
            image_id_to_frame=image_id_to_frame,
        )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def annotation_count(self) -> int:
        return sum(len(items) for items in self.frames.values())

    @property
    def frame_range(self) -> tuple[int, int] | None:
        if not self.frames:
            return None
        keys = sorted(self.frames)
        return keys[0], keys[-1]

    def masks_for_frame(
        self,
        frame_index: int,
    ) -> list[tuple[int, np.ndarray, tuple[float, float, float, float], float]]:
        frame_index = int(frame_index)
        cached = self._cache.get(frame_index)
        if cached is not None:
            self._cache.move_to_end(frame_index)
            return cached

        decoded: list[tuple[int, np.ndarray, tuple[float, float, float, float], float]] = []
        for ann in self.frames.get(frame_index, []):
            try:
                mask = decode_coco_segmentation(ann.segmentation, ann.size)
            except Exception:
                logger.debug("Failed to decode mask for frame %d", frame_index, exc_info=True)
                continue
            if mask is not None and mask.any():
                decoded.append((ann.track_id, mask, ann.bbox, ann.score))

        self._cache[frame_index] = decoded
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return decoded

    def mask_for_annotation(self, frame_index: int, annotation_index: int) -> Optional[np.ndarray]:
        """Decode one annotation mask without decoding the whole frame."""
        frame_index = int(frame_index)
        annotation_index = int(annotation_index)
        key = (frame_index, annotation_index)
        cached = self._annotation_cache.get(key)
        if cached is not None:
            self._annotation_cache.move_to_end(key)
            return cached

        annotations = self.frames.get(frame_index, [])
        if annotation_index < 0 or annotation_index >= len(annotations):
            return None
        ann = annotations[annotation_index]
        try:
            mask = decode_coco_segmentation(ann.segmentation, ann.size)
        except Exception:
            logger.debug("Failed to decode mask for frame %d annotation %d", frame_index, annotation_index, exc_info=True)
            return None
        if mask is None or not mask.any():
            return None

        self._annotation_cache[key] = mask
        while len(self._annotation_cache) > self.cache_size * 4:
            self._annotation_cache.popitem(last=False)
        return mask

    def with_swapped_track_ids(
        self,
        track_a: int,
        track_b: int,
        frame_indices: Optional[Iterable[int]] = None,
    ) -> "CocoMaskStore":
        """Return a copy with two mask track identities swapped.

        ``frame_indices`` are source-video frame numbers. When omitted, the
        swap is applied to every frame in the store.
        """
        track_a = int(track_a)
        track_b = int(track_b)
        if track_a == track_b:
            return self

        frame_set = None
        if frame_indices is not None:
            frame_set = {int(frame) for frame in frame_indices}
            if not frame_set:
                return self

        swapped_frames: dict[int, list[MaskAnnotation]] = {}
        for frame_index, annotations in self.frames.items():
            if frame_set is not None and int(frame_index) not in frame_set:
                swapped_frames[int(frame_index)] = list(annotations)
                continue

            swapped_annotations: list[MaskAnnotation] = []
            for ann in annotations:
                if ann.track_id == track_a:
                    swapped_annotations.append(replace(ann, track_id=track_b))
                elif ann.track_id == track_b:
                    swapped_annotations.append(replace(ann, track_id=track_a))
                else:
                    swapped_annotations.append(ann)
            swapped_frames[int(frame_index)] = swapped_annotations

        swapped_payload = self._swapped_payload(track_a, track_b, frame_set)
        return CocoMaskStore(
            path=self.path,
            frames=swapped_frames,
            info=dict(self.info),
            cache_size=self.cache_size,
            payload=swapped_payload,
            image_id_to_frame=self._image_id_to_frame,
        )

    def with_reassigned_track_ids(
        self,
        track_map: dict[int, int],
        frame_indices: Optional[Iterable[int]] = None,
    ) -> "CocoMaskStore":
        """Return a copy with mask track IDs reassigned by ``track_map``."""
        clean_map = {
            int(src): int(dst)
            for src, dst in (track_map or {}).items()
            if int(src) != int(dst)
        }
        if not clean_map:
            return self

        frame_set = None
        if frame_indices is not None:
            frame_set = {int(frame) for frame in frame_indices}
            if not frame_set:
                return self

        reassigned_frames: dict[int, list[MaskAnnotation]] = {}
        for frame_index, annotations in self.frames.items():
            if frame_set is not None and int(frame_index) not in frame_set:
                reassigned_frames[int(frame_index)] = list(annotations)
                continue

            reassigned_annotations: list[MaskAnnotation] = []
            for ann in annotations:
                new_track = clean_map.get(int(ann.track_id))
                if new_track is None:
                    reassigned_annotations.append(ann)
                else:
                    reassigned_annotations.append(replace(ann, track_id=new_track))
            reassigned_frames[int(frame_index)] = reassigned_annotations

        reassigned_payload = self._reassigned_payload(clean_map, frame_set)
        return CocoMaskStore(
            path=self.path,
            frames=reassigned_frames,
            info=dict(self.info),
            cache_size=self.cache_size,
            payload=reassigned_payload,
            image_id_to_frame=self._image_id_to_frame,
        )

    def save_json(self, output_path: str | Path) -> str:
        """Write the current COCO payload to *output_path*."""
        payload = deepcopy(self._payload)
        if not isinstance(payload, dict):
            payload = self._minimal_payload()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(output)

    def _swapped_payload(
        self,
        track_a: int,
        track_b: int,
        frame_set: Optional[set[int]],
    ) -> Optional[dict]:
        if not isinstance(self._payload, dict):
            return None
        payload = deepcopy(self._payload)
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            return payload
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            image_id = int(ann.get("image_id", 0) or 0)
            frame_idx = int(self._image_id_to_frame.get(image_id, image_id))
            if frame_set is not None and frame_idx not in frame_set:
                continue
            if "track_id" in ann:
                try:
                    current = int(ann.get("track_id", 0) or 0)
                except Exception:
                    continue
                if current == track_a:
                    ann["track_id"] = track_b
                elif current == track_b:
                    ann["track_id"] = track_a
            elif "category_id" in ann:
                try:
                    current = int(ann.get("category_id", 0) or 0)
                except Exception:
                    continue
                if current == track_a:
                    ann["category_id"] = track_b
                elif current == track_b:
                    ann["category_id"] = track_a
        info = payload.get("info")
        if isinstance(info, dict):
            info["dlc_processor_cleaned"] = True
        else:
            payload["info"] = {"dlc_processor_cleaned": True}
        return payload

    def _reassigned_payload(
        self,
        track_map: dict[int, int],
        frame_set: Optional[set[int]],
    ) -> Optional[dict]:
        if not isinstance(self._payload, dict):
            return None
        payload = deepcopy(self._payload)
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            return payload
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            image_id = int(ann.get("image_id", 0) or 0)
            frame_idx = int(self._image_id_to_frame.get(image_id, image_id))
            if frame_set is not None and frame_idx not in frame_set:
                continue
            key = "track_id" if "track_id" in ann else "category_id"
            try:
                current = int(ann.get(key, 0) or 0)
            except Exception:
                continue
            if current not in track_map:
                continue
            ann["track_id"] = int(track_map[current])
        info = payload.get("info")
        if isinstance(info, dict):
            info["dlc_processor_cleaned"] = True
        else:
            payload["info"] = {"dlc_processor_cleaned": True}
        return payload

    def _minimal_payload(self) -> dict:
        images = []
        annotations = []
        ann_id = 1
        for frame_idx, items in sorted(self.frames.items()):
            images.append({"id": int(frame_idx), "frame_index": int(frame_idx)})
            for ann in items:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": int(frame_idx),
                        "track_id": int(ann.track_id),
                        "bbox": [float(v) for v in ann.bbox],
                        "score": float(ann.score),
                        "segmentation": ann.segmentation,
                    }
                )
                ann_id += 1
        return {
            "info": dict(self.info),
            "images": images,
            "annotations": annotations,
        }


def is_coco_mask_json(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix.lower() != ".json" or not p.exists():
        return False
    name = p.name.lower()
    if "mask" in name and ("coco" in name or "polygon" in name):
        return True
    try:
        with p.open("r", encoding="utf-8") as fh:
            head = fh.read(4096)
    except Exception:
        return False
    return '"annotations"' in head and '"segmentation"' in head


def decode_coco_segmentation(
    segmentation: Any,
    size: tuple[int, int],
) -> Optional[np.ndarray]:
    """Decode COCO polygon, uncompressed RLE, or compressed RLE to bool mask."""
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        return None

    if isinstance(segmentation, dict):
        raw_size = segmentation.get("size")
        if isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
            height, width = int(raw_size[0]), int(raw_size[1])
        counts = segmentation.get("counts")
        if isinstance(counts, str):
            return _decode_rle_counts(_decode_compressed_counts(counts), height, width)
        if isinstance(counts, list):
            return _decode_rle_counts([int(v) for v in counts], height, width)
        return None

    if isinstance(segmentation, list):
        return _decode_polygons(segmentation, height, width)

    return None


def _decode_polygons(polygons: Iterable[Iterable[float]], height: int, width: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    contours: list[np.ndarray] = []
    for polygon in polygons:
        coords = np.asarray(list(polygon), dtype=np.float32).reshape(-1, 2)
        if len(coords) >= 3:
            contours.append(np.round(coords).astype(np.int32).reshape(-1, 1, 2))
    if contours:
        cv2.fillPoly(mask, contours, 1)
    return mask.astype(bool)


def _decode_rle_counts(counts: list[int], height: int, width: int) -> np.ndarray:
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    value = 0
    for count in counts:
        count = max(0, int(count))
        end = min(idx + count, flat.size)
        if value == 1 and end > idx:
            flat[idx:end] = 1
        idx = end
        value = 1 - value
        if idx >= flat.size:
            break
    return flat.reshape((height, width), order="F").astype(bool)


def _decode_compressed_counts(text: str) -> list[int]:
    """Decode pycocotools' compact COCO RLE count string."""
    counts: list[int] = []
    p = 0
    m = 0
    while p < len(text):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(text[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            if not more and (c & 0x10):
                x |= -1 << (5 * (k + 1))
            k += 1
        if m > 2:
            x += counts[m - 2]
        counts.append(int(x))
        m += 1
    return counts
