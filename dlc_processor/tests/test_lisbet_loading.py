from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dlc_processor.core.dlc_loader import detect_animals, load_dlc_file
from dlc_processor.core.mask_loader import CocoMaskStore
from dlc_processor.core.project_manager import scan_folder_with_masks, scan_folder_with_sidecars
from dlc_processor.core.time_loader import align_times_to_dfs, load_frame_times


def test_dlc_loader_trims_to_coordinate_window_and_keeps_source_frames(tmp_path: Path) -> None:
    columns = pd.MultiIndex.from_product(
        [["scorer"], ["mouse1"], ["nose"], ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(np.nan, index=[100, 101, 102, 103, 104], columns=columns)
    df.loc[101:103, ("scorer", "mouse1", "nose", "x")] = [1.0, 2.0, 3.0]
    df.loc[101:103, ("scorer", "mouse1", "nose", "y")] = [4.0, 5.0, 6.0]
    df.loc[101:103, ("scorer", "mouse1", "nose", "likelihood")] = 0.9

    csv_path = tmp_path / "window_tracking.csv"
    df.to_csv(csv_path)

    loaded = load_dlc_file(csv_path)
    mouse = loaded["mouse1"]

    assert len(mouse) == 3
    assert mouse["nose_x"].tolist() == [1.0, 2.0, 3.0]
    assert mouse.attrs["frame_numbers"].tolist() == [101, 102, 103]


def test_detect_animals_reads_csv_and_h5_headers(tmp_path: Path) -> None:
    columns = pd.MultiIndex.from_product(
        [["scorer"], ["mouse1", "mouse2"], ["nose"], ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(np.arange(12, dtype=float).reshape(2, 6), columns=columns)
    csv_path = tmp_path / "two_mice_tracking.csv"
    h5_path = tmp_path / "two_mice_tracking.h5"
    df.to_csv(csv_path)
    df.to_hdf(h5_path, key="df_with_missing", mode="w")

    assert detect_animals(csv_path) == ["mouse1", "mouse2"]
    assert detect_animals(h5_path) == ["mouse1", "mouse2"]


def test_scan_folder_with_masks_reads_lisbet_manifest(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    csv_path = tmp_path / "trial_tracking.csv"
    mask_path = tmp_path / "trial_masks_coco.json"
    video.write_bytes(b"")
    csv_path.write_text("", encoding="utf-8")
    mask_path.write_text('{"images": [], "annotations": []}', encoding="utf-8")
    (tmp_path / "tracking_manifest.json").write_text(
        json.dumps(
            {
                "csv_outputs": [str(csv_path)],
                "coco_outputs": [str(mask_path)],
                "config": {"input_paths": [str(video)]},
            }
        ),
        encoding="utf-8",
    )

    videos, dlc_files, mask_files = scan_folder_with_masks(tmp_path)

    assert videos == [str(video)]
    assert dlc_files == [str(csv_path)]
    assert mask_files == [str(mask_path)]


def test_scan_folder_with_sidecars_separates_frame_times_from_tracking(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    csv_path = tmp_path / "trial_tracking.csv"
    time_path = tmp_path / "trial_adj_aligned_frame_times.csv"
    video.write_bytes(b"")
    csv_path.write_text("", encoding="utf-8")
    time_path.write_text("frame,time_s\n0,1.0\n1,1.1\n", encoding="utf-8")

    videos, dlc_files, _mask_files, time_files = scan_folder_with_sidecars(tmp_path)

    assert videos == [str(video)]
    assert dlc_files == [str(csv_path)]
    assert time_files == [str(time_path)]


def test_frame_time_loader_ignores_timesy_metadata_footer(tmp_path: Path) -> None:
    time_path = tmp_path / "barcode_adj_aligned_frame_times.csv"
    time_path.write_text(
        "frame,time_s\n"
        "0,10.0\n"
        "1,10.033\n"
        "2,10.067\n"
        "\n"
        "# method,interpolation\n"
        "# matched_pulses,3\n",
        encoding="utf-8",
    )

    loaded = load_frame_times(time_path)

    assert loaded.frames.tolist() == [0, 1, 2]
    assert np.allclose(loaded.times, [10.0, 10.033, 10.067])


def test_frame_time_loader_accepts_one_column_sync_txt(tmp_path: Path) -> None:
    time_path = tmp_path / "frame_time_sync.txt"
    time_path.write_text(
        "-3.031614313135833072e-02\n"
        "3.016857743957747362e-03\n"
        "3.634988561876181801e-02\n",
        encoding="utf-8",
    )

    loaded = load_frame_times(time_path)

    assert loaded.frames is None
    assert np.allclose(loaded.times, [-0.03031614313135833, 0.0030168577439577474, 0.03634988561876182])


def test_frame_time_alignment_uses_tracking_source_frames() -> None:
    df = pd.DataFrame({"nose_x": [1.0, 2.0], "nose_y": [3.0, 4.0]})
    df.attrs["frame_numbers"] = np.array([101, 103])
    time_path = "dummy_adj_aligned_frame_times.csv"
    from dlc_processor.core.time_loader import FrameTimeData

    times = FrameTimeData(
        path=time_path,
        frames=np.array([100, 101, 102, 103]),
        times=np.array([9.9, 10.0, 10.1, 10.2]),
    )

    aligned = align_times_to_dfs(times, {"mouse1": df})

    assert np.allclose(aligned, [10.0, 10.2])


def test_coco_mask_store_decodes_uncompressed_rle(tmp_path: Path) -> None:
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 1] = True
    mask[2, 3] = True
    counts = _encode_uncompressed_rle(mask)
    path = tmp_path / "trial_masks_coco.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 10, "frame_index": 123, "height": 3, "width": 4}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "track_id": 2,
                        "bbox": [1, 0, 3, 3],
                        "score": 0.8,
                        "segmentation": {"size": [3, 4], "counts": counts},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = CocoMaskStore.from_file(path)
    items = store.masks_for_frame(123)

    assert store.frame_range == (123, 123)
    assert len(items) == 1
    track_id, decoded, _bbox, score = items[0]
    assert track_id == 2
    assert score == 0.8
    assert np.array_equal(decoded, mask)


def test_coco_mask_store_swaps_track_ids_for_selected_frames(tmp_path: Path) -> None:
    mask_a = np.zeros((3, 4), dtype=bool)
    mask_a[0, 1] = True
    mask_b = np.zeros((3, 4), dtype=bool)
    mask_b[2, 3] = True
    path = tmp_path / "trial_masks_coco.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "frame_index": 100, "height": 3, "width": 4},
                    {"id": 11, "frame_index": 101, "height": 3, "width": 4},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "track_id": 1,
                        "bbox": [1, 0, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_a)},
                    },
                    {
                        "id": 2,
                        "image_id": 10,
                        "track_id": 2,
                        "bbox": [3, 2, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_b)},
                    },
                    {
                        "id": 3,
                        "image_id": 11,
                        "track_id": 1,
                        "bbox": [1, 0, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_a)},
                    },
                    {
                        "id": 4,
                        "image_id": 11,
                        "track_id": 2,
                        "bbox": [3, 2, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_b)},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    store = CocoMaskStore.from_file(path)
    swapped = store.with_swapped_track_ids(1, 2, frame_indices=[100])

    assert [item[0] for item in store.masks_for_frame(100)] == [1, 2]
    assert [item[0] for item in swapped.masks_for_frame(100)] == [2, 1]
    assert [item[0] for item in swapped.masks_for_frame(101)] == [1, 2]
    assert np.array_equal(swapped.masks_for_frame(100)[0][1], mask_a)


def test_coco_mask_store_saves_swapped_cleaned_json(tmp_path: Path) -> None:
    mask_a = np.zeros((3, 4), dtype=bool)
    mask_a[0, 1] = True
    mask_b = np.zeros((3, 4), dtype=bool)
    mask_b[2, 3] = True
    path = tmp_path / "trial_masks_coco.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "frame_index": 100, "height": 3, "width": 4},
                    {"id": 11, "frame_index": 101, "height": 3, "width": 4},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "track_id": 1,
                        "category_id": 1,
                        "bbox": [1, 0, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_a)},
                    },
                    {
                        "id": 2,
                        "image_id": 10,
                        "track_id": 2,
                        "category_id": 1,
                        "bbox": [3, 2, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_b)},
                    },
                    {
                        "id": 3,
                        "image_id": 11,
                        "track_id": 1,
                        "category_id": 1,
                        "bbox": [1, 0, 1, 1],
                        "segmentation": {"size": [3, 4], "counts": _encode_uncompressed_rle(mask_a)},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    cleaned_path = tmp_path / "trial_masks_coco_cleaned.json"
    store = CocoMaskStore.from_file(path)
    swapped = store.with_swapped_track_ids(1, 2, frame_indices=[100])
    swapped.save_json(cleaned_path)

    payload = json.loads(cleaned_path.read_text(encoding="utf-8"))
    by_id = {ann["id"]: ann for ann in payload["annotations"]}
    assert by_id[1]["track_id"] == 2
    assert by_id[2]["track_id"] == 1
    assert by_id[3]["track_id"] == 1
    assert by_id[1]["category_id"] == 1
    assert payload["info"]["dlc_processor_cleaned"] is True


def _encode_uncompressed_rle(mask: np.ndarray) -> list[int]:
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        if int(value) == current:
            run += 1
        else:
            counts.append(run)
            current = int(value)
            run = 1
    counts.append(run)
    return counts
