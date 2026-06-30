"""Tests for the LISBET mask+pose JSONL(.gz) loader and folder discovery."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from dlc_processor.core.dlc_loader import detect_animals, get_bodyparts, load_dlc_file
from dlc_processor.core.mask_loader import CocoMaskStore
from dlc_processor.core.maskpose_loader import (
    MASKPOSE_BODYPARTS,
    build_maskpose_mask_store,
    detect_keypoint_animals,
    is_maskpose_combined_file,
    is_maskpose_keypoint_file,
    is_maskpose_mask_file,
    load_maskpose_keypoints,
)
from dlc_processor.core.project_manager import (
    discover_maskpose_export,
    scan_folder_with_sidecars,
)

# A square mask polygon so contact/overlap is easy to reason about.
_SQUARE_A = [[10, 10], [40, 10], [40, 40], [10, 40]]
_SQUARE_B = [[60, 60], [90, 60], [90, 90], [60, 90]]
_SQUARE_B_TOUCHING = [[40, 10], [70, 10], [70, 40], [40, 40]]  # shares edge with A


def _kps(cx: float, cy: float) -> list[list[float]]:
    """Eight keypoints scattered around (cx, cy), each [x, y, conf]."""
    return [[cx + i, cy + i, 0.9] for i in range(len(MASKPOSE_BODYPARTS))]


def _write_jsonl_gz(path: Path, frames: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame) + "\n")
    return path


def _combined_frames() -> list[dict]:
    """Two leading empty frames, then two animals; mouse2 appears late."""
    return [
        {"f": 0, "animals": []},
        {"f": 1, "animals": []},
        {"f": 2, "animals": [
            {"id": 1, "keypoints": _kps(20, 20), "mask": _SQUARE_A, "mask_conf": 0.95},
        ]},
        {"f": 3, "animals": [
            {"id": 1, "keypoints": _kps(20, 20), "mask": _SQUARE_A, "mask_conf": 0.95},
            {"id": 2, "keypoints": _kps(75, 75), "mask": _SQUARE_B, "mask_conf": 0.80},
        ]},
        {"f": 4, "animals": [
            {"id": 1, "keypoints": _kps(20, 20), "mask": _SQUARE_A, "mask_conf": 0.95},
            {"id": 2, "keypoints": _kps(55, 25), "mask": _SQUARE_B_TOUCHING, "mask_conf": 0.80},
        ]},
    ]


def test_load_maskpose_keypoints_builds_trimmed_dataframes(tmp_path: Path) -> None:
    path = _write_jsonl_gz(tmp_path / "rec_keypoints.jsonl.gz", _combined_frames())
    dfs = load_maskpose_keypoints(path)

    assert set(dfs) == {"mouse1", "mouse2"}
    mouse1 = dfs["mouse1"]
    # Leading empty frames (0, 1) are trimmed; tracking starts at frame 2.
    assert get_bodyparts(mouse1) == list(MASKPOSE_BODYPARTS)
    assert mouse1.attrs["frame_numbers"].tolist() == [2, 3, 4]
    assert mouse1["nose_x"].tolist() == [20.0, 20.0, 20.0]
    # mouse2 only present from frame 3 -> NaN on the first retained row.
    mouse2 = dfs["mouse2"]
    assert np.isnan(mouse2["nose_x"].iloc[0])
    assert mouse2["nose_x"].iloc[1] == 75.0


def test_load_dlc_file_routes_jsonl_to_keypoints(tmp_path: Path) -> None:
    path = _write_jsonl_gz(tmp_path / "rec_keypoints.jsonl.gz", _combined_frames())
    assert set(load_dlc_file(path)) == {"mouse1", "mouse2"}
    assert detect_animals(path) == ["mouse1", "mouse2"]
    assert detect_keypoint_animals(path) == ["mouse1", "mouse2"]


def test_build_mask_store_decodes_polygons_at_frame_size(tmp_path: Path) -> None:
    path = _write_jsonl_gz(tmp_path / "rec_mask.jsonl.gz", _combined_frames())
    store = build_maskpose_mask_store(path, frame_size=(120, 120))

    assert store.frame_range == (2, 4)
    masks = store.masks_for_frame(3)
    assert len(masks) == 2
    track_ids = sorted(tid for tid, _m, _b, _s in masks)
    assert track_ids == [1, 2]
    for _tid, mask, bbox, score in masks:
        assert mask.shape == (120, 120)
        assert mask.any()
        assert score > 0
    # Polygon A spans x,y in [10, 40] -> bbox (10, 10, 30, 30).
    bbox_a = next(b for t, _m, b, _s in masks if t == 1)
    assert bbox_a == (10.0, 10.0, 30.0, 30.0)


def test_mask_store_infers_size_without_video(tmp_path: Path) -> None:
    path = _write_jsonl_gz(tmp_path / "rec_mask.jsonl.gz", _combined_frames())
    store = build_maskpose_mask_store(path)  # no frame_size, no video
    masks = store.masks_for_frame(3)
    assert masks and all(m.any() for _t, m, _b, _s in masks)


def test_mask_store_swap_and_save_roundtrip(tmp_path: Path) -> None:
    path = _write_jsonl_gz(tmp_path / "rec_mask.jsonl.gz", _combined_frames())
    store = build_maskpose_mask_store(path, frame_size=(120, 120))
    swapped = store.with_swapped_track_ids(1, 2, frame_indices=[3])

    out = swapped.save_json(tmp_path / "cleaned.json")
    reloaded = CocoMaskStore.from_file(out)
    reloaded_tracks = sorted(t for t, _m, _b, _s in reloaded.masks_for_frame(3))
    assert reloaded_tracks == [1, 2]
    # Masks still decode after the round-trip (polygon size persisted).
    assert all(m.any() for _t, m, _b, _s in reloaded.masks_for_frame(3))


def test_maskpose_contact_detection(tmp_path: Path) -> None:
    from dlc_processor.core.mask_social import pair_mask_contact

    kp_path = _write_jsonl_gz(tmp_path / "rec_keypoints.jsonl.gz", _combined_frames())
    mask_path = _write_jsonl_gz(tmp_path / "rec_mask.jsonl.gz", _combined_frames())
    dfs = load_maskpose_keypoints(kp_path)
    store = build_maskpose_mask_store(mask_path, frame_size=(120, 120))

    contact = pair_mask_contact(
        store, dfs["mouse1"], dfs["mouse2"], max_edge_gap_px=2
    )
    # Retained rows map to frames [2, 3, 4]; squares touch only at frame 4.
    assert contact.tolist() == [False, False, True]


def test_detectors_recognise_suffixes_and_content(tmp_path: Path) -> None:
    kp = _write_jsonl_gz(tmp_path / "rec_keypoints.jsonl.gz", _combined_frames())
    mask = _write_jsonl_gz(tmp_path / "rec_mask.jsonl.gz", _combined_frames())
    combined = _write_jsonl_gz(tmp_path / "rec_maskpose.jsonl.gz", _combined_frames())

    assert is_maskpose_keypoint_file(kp) and not is_maskpose_mask_file(kp)
    assert is_maskpose_mask_file(mask) and not is_maskpose_keypoint_file(mask)
    assert is_maskpose_combined_file(combined)
    assert is_maskpose_keypoint_file(combined) and is_maskpose_mask_file(combined)

    # A generically named JSONL is classified by content sniffing.
    generic = _write_jsonl_gz(tmp_path / "something.jsonl.gz", _combined_frames())
    assert is_maskpose_combined_file(generic)


def test_discover_maskpose_export_pairs_by_base(tmp_path: Path) -> None:
    root = tmp_path / "mask_pose"
    (root / "keypoint").mkdir(parents=True)
    (root / "mask").mkdir()
    (root / "overlays").mkdir()
    (tmp_path / "videos").mkdir()

    bases = ["A_001_preprocessed", "A_002_preprocessed"]
    for base in bases:
        _write_jsonl_gz(root / "keypoint" / f"{base}_keypoints.jsonl.gz", _combined_frames())
        _write_jsonl_gz(root / "mask" / f"{base}_mask.jsonl.gz", _combined_frames())
        (tmp_path / "videos" / f"{base}.mp4").write_bytes(b"fake")

    found = discover_maskpose_export(root)
    assert [Path(p).name for p in found["keypoint_files"]] == [
        f"{b}_keypoints.jsonl.gz" for b in bases
    ]
    assert [Path(p).name for p in found["mask_files"]] == [f"{b}_mask.jsonl.gz" for b in bases]
    assert [Path(p).stem for p in found["videos"]] == bases

    # Pointing at a subfolder resolves the same export root.
    via_subdir = discover_maskpose_export(root / "keypoint")
    assert via_subdir["keypoint_files"] == found["keypoint_files"]

    # And the generic sidecar scanner surfaces the same files.
    videos, dlc, masks, _times = scan_folder_with_sidecars(root)
    assert len(dlc) == 2 and len(masks) == 2 and len(videos) == 2
