"""Quick smoke tests for new modules."""

from pathlib import Path
import json
import os
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest


def test_batch_count_bouts():
    from dlc_processor.core.batch_processor import _count_bouts

    assert _count_bouts(np.array([True, True, False, True, False, True, True])) == 3
    assert _count_bouts(np.array([])) == 0
    assert _count_bouts(np.array([False, False])) == 0
    assert _count_bouts(np.array([True])) == 1


def test_framewise_export_uses_underscore_behavior_labels():
    from dlc_processor.core.batch_processor import _build_framewise_table
    from dlc_processor.core.social_behaviors import export_behavior_label

    assert export_behavior_label("a_following_b") == "A_follows_B"

    table = _build_framewise_table(
        {"mouse1": pd.DataFrame({"nose_x": [1.0, 2.0]})},
        {
            "a_following_b": np.array([True, False]),
            "social_mouse1_vs_mouse2__b_following_a": np.array([False, True]),
        },
        frame_numbers=np.array([10, 11]),
        time_s=np.array([0.1, 0.2]),
    )

    assert "A_follows_B" in table.columns
    assert "social_mouse1_vs_mouse2__B_follows_A" in table.columns
    assert "a_following_b" not in table.columns


def test_video_overlay_export_source_frames_respect_window():
    from dlc_processor.workers.video_export_worker import _export_source_frames

    assert _export_source_frames({}, 8, start_frame=3, end_frame=5) == [3, 4, 5]
    assert _export_source_frames({}, 8, start_frame=5, end_frame=0) == [5, 6, 7]

    df = pd.DataFrame({"x": np.arange(4)})
    df.attrs["frame_numbers"] = np.array([10, 11, 13, 20])
    assert _export_source_frames({"mouse1": df}, 30, start_frame=11, end_frame=13) == [11, 13]
    assert _export_source_frames({"mouse1": df}, 30, start_frame=13, end_frame=0) == [13, 20]
    assert _export_source_frames({"mouse1": df}, 30, start_frame=21, end_frame=0) == []


def test_batch_metadata_uses_preloaded_animal_ids():
    from dlc_processor.ui.batch_panel import _animals_for_record

    assert _animals_for_record({"animal_ids": ["mouse1", "mouse2"]}) == ["mouse1", "mouse2"]


def test_social_panel_process_all_loaded_videos_emits_current_thresholds():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from dlc_processor.ui.social_panel import SocialPanel

    app = QApplication.instance() or QApplication([])
    _ = app
    panel = SocialPanel()
    panel.set_project_batch_count(3)
    panel._spin_close.setValue(17.0)
    panel._spin_side.setValue(41.0)
    panel._spin_follow.setValue(23.0)
    panel._spin_fwin.setValue(9)
    panel._spin_median.setValue(3)
    panel._spin_likelihood.setValue(0.35)

    emitted = []
    panel.batch_process_requested.connect(lambda options: emitted.append(options))

    assert panel._btn_process_all.isEnabled()
    panel._btn_process_all.click()

    assert emitted
    options = emitted[0]
    params = options["social_params"]
    assert options["compute_social"] is True
    assert params["close_tol_px"] == pytest.approx(17.0)
    assert params["side_tol_px"] == pytest.approx(41.0)
    assert params["follow_tol_px"] == pytest.approx(23.0)
    assert params["follow_window"] == 9
    assert params["median_filter"] == 3
    assert params["likelihood_threshold"] == pytest.approx(0.35)


def test_cleaning_panel_timeline_flags_drive_identity_swap_range():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from dlc_processor.ui.cleaning_panel import CleaningPanel

    app = QApplication.instance() or QApplication([])
    _ = app
    panel = CleaningPanel()
    panel.set_n_frames(100)
    panel.set_animal_dfs({"mouse1": pd.DataFrame(), "mouse2": pd.DataFrame()})

    ranges = []
    shown = []
    emitted = []
    panel.range_changed.connect(lambda start, end: ranges.append((start, end)))
    panel.range_selection_requested.connect(lambda: shown.append(True))
    panel.identity_swap_requested.connect(lambda params: emitted.append(params))

    panel.flag_range_start(10)
    panel.flag_range_end(20)

    assert panel._chk_use_range.isChecked()
    assert panel._spin_range_start.value() == 10
    assert panel._spin_range_end.value() == 21
    assert ranges[-1] == (10, 21)
    assert shown
    assert panel._btn_fix_mask_ids.isEnabled()

    panel.request_identity_swap()

    assert emitted
    assert emitted[0]["animal_a"] == "mouse1"
    assert emitted[0]["animal_b"] == "mouse2"
    assert emitted[0]["start_frame"] == 10
    assert emitted[0]["end_frame"] == 21

    panel.clear_range()

    assert not panel._chk_use_range.isChecked()
    assert ranges[-1] == (0, 0)

    single_panel = CleaningPanel()
    single_panel.set_animal_dfs({"mouse1": pd.DataFrame()})
    assert single_panel._btn_fix_mask_ids.isEnabled()


def test_single_animal_mask_identity_normalizes_to_mouse1(tmp_path):
    from dlc_processor.core.mask_loader import CocoMaskStore, MaskAnnotation
    from dlc_processor.tab_widget import _normalize_single_animal_mask_identity

    df = pd.DataFrame({"nose_x": [1.0], "nose_y": [2.0]})
    payload = {
        "images": [{"id": 10, "frame_index": 5, "height": 8, "width": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 0,
                "track_id": 7,
                "bbox": [1, 1, 3, 3],
                "score": 1.0,
                "segmentation": {"size": [8, 8], "counts": [64]},
            }
        ],
    }
    store = CocoMaskStore(
        path=tmp_path / "masks.json",
        frames={
            5: [
                MaskAnnotation(
                    track_id=7,
                    bbox=(1.0, 1.0, 3.0, 3.0),
                    score=1.0,
                    segmentation={"size": [8, 8], "counts": [64]},
                    size=(8, 8),
                )
            ]
        },
        payload=payload,
        image_id_to_frame={10: 5},
    )

    fixed, fixed_store, summary = _normalize_single_animal_mask_identity(
        {"animal_0": df},
        store,
    )

    assert list(fixed.keys()) == ["mouse1"]
    assert fixed_store.frames[5][0].track_id == 1
    assert summary["tracking_renamed"] is True
    assert summary["mask_reassigned"] is True
    assert summary["old_track_id"] == 7

    saved_path = fixed_store.save_json(tmp_path / "masks_cleaned.json")
    saved = json.loads(Path(saved_path).read_text(encoding="utf-8"))

    assert saved["annotations"][0]["track_id"] == 1
    assert saved["annotations"][0]["category_id"] == 0
    assert saved["info"]["dlc_processor_cleaned"] is True


def test_batch_metadata_import_matches_vta_recording_alias_and_attaches(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from dlc_processor.ui.batch_panel import BatchPanel

    app = QApplication.instance() or QApplication([])
    _ = app
    panel = BatchPanel()
    rec = {
        "dlc_path": str(tmp_path / "31098_2_NPX_object_social_food.csv"),
        "animal_ids": [],
    }
    panel.set_records([rec])
    metadata_path = tmp_path / "dlc_metadata.csv"
    metadata_path.write_text(
        "recording,animal,mouseId,condition,genotype,group,sex,cohort,notes\n"
        "31098_2_NPX_object_social_food_start_310s_end_1500s_tracking,mouse1,31098,,Het,,,,\n"
        "31098_2_NPX_object_social_food_start_310s_end_1500s_tracking,mouse2,31098-stim,,Het,,,,\n",
        encoding="utf-8",
    )

    assert panel.import_metadata_csv(metadata_path) == 2
    rows = panel.metadata_rows()
    assert {row["animal"] for row in rows} >= {"mouse1", "mouse2"}
    visible = {
        panel._table.item(row, 1).text(): panel._table.item(row, 4).text()
        for row in range(panel._table.rowCount())
    }
    assert visible["mouse1"] == "Het"
    assert visible["mouse2"] == "Het"

    attached = panel.attach_metadata_to_records([rec])[0]
    assert attached["animal_metadata"]["mouse1"]["mouseId"] == "31098"
    assert attached["animal_metadata"]["mouse1"]["genotype"] == "Het"
    assert attached["metadata"]["genotype"] == "Het"


def test_batch_metadata_import_overwrites_existing_blank_visible_rows(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from dlc_processor.ui.batch_panel import BatchPanel

    app = QApplication.instance() or QApplication([])
    _ = app
    panel = BatchPanel()
    rec = {
        "dlc_path": str(tmp_path / "31102_2_NPX_object_social_food_start_310s_end_1500s_tracking.csv"),
        "animal_ids": ["mouse1", "mouse2"],
    }
    panel.set_records([rec])
    assert panel._table.item(0, 4).text() == ""
    metadata_path = tmp_path / "dlc_metadata.csv"
    metadata_path.write_text(
        "recording,animal,mouseId,condition,genotype,group,sex,cohort,notes\n"
        "31102_2_NPX_object_social_food_start_310s_end_1500s_tracking,mouse1,31102,,WT,,,,\n"
        "31102_2_NPX_object_social_food_start_310s_end_1500s_tracking,mouse2,31102-stim,,WT,,,,\n",
        encoding="utf-8",
    )

    panel.import_metadata_csv(metadata_path)

    visible = {
        panel._table.item(row, 1).text(): (
            panel._table.item(row, 2).text(),
            panel._table.item(row, 4).text(),
        )
        for row in range(panel._table.rowCount())
    }
    assert visible == {
        "mouse1": ("31102", "WT"),
        "mouse2": ("31102-stim", "WT"),
    }


def test_scan_folder_skips_dlc_metadata_table(tmp_path):
    from dlc_processor.core.project_manager import scan_folder_with_sidecars

    (tmp_path / "dlc_metadata.csv").write_text("recording,animal,mouseId\n", encoding="utf-8")
    tracking = tmp_path / "session_tracking.csv"
    tracking.write_text("frame,x,y\n", encoding="utf-8")

    _videos, dlc_files, _mask_files, _time_files = scan_folder_with_sidecars(tmp_path)

    assert dlc_files == [str(tracking)]


def test_project_manager_persists_dlc_animal_ids(tmp_path):
    from dlc_processor.core.project_manager import load_project, save_project

    path = tmp_path / "project.dlcproj"
    save_project(
        path,
        {
            "dlc_files": ["a.csv"],
            "dlc_animal_ids_by_path": {"a.csv": ["mouse1", "mouse2"]},
            "settings": {"batch_summary_group_by": "genotype"},
        },
    )

    loaded = load_project(path)
    assert loaded["dlc_animal_ids_by_path"] == {"a.csv": ["mouse1", "mouse2"]}
    assert loaded["settings"]["batch_summary_group_by"] == "genotype"


def test_short_name_mappings():
    from dlc_processor.ui.plot_panel import _short_name

    assert _short_name("body_speed_px_s") == "speed (px/s)"
    assert _short_name("a_nose2anogenital_b") == "A to B anogenital"
    assert _short_name("body_elongation_px") == "elongation (px)"
    assert _short_name("heading_body_angle_diff_deg") == "crab-walk angle"
    assert _short_name("trajectory_curvature_1_px") == "curvature (1/px)"
    assert _short_name("distance_traveled_cm") == "cum. distance (cm)"
    assert _short_name("partner_distance_px") == "partner dist (px)"
    assert _short_name("rearing") == "rearing"
    assert _short_name("mouse1__mobile") == "mouse1 mobile"
    assert _short_name("mouse1__immobile") == "mouse1 immobile"
    assert _short_name("fighting") == "fighting"
    assert _short_name("a_approaches_b") == "A approaches B"
    assert _short_name("a_escapes_b") == "A escapes B"
    assert _short_name("a_withdrawal_after_contact_b") == "A withdraws after contact B"


def test_egocentric_transform_direction():
    """Verify ego_x/ego_y signs match body-frame expectations."""
    # Mouse facing right in image coords: nose=(120,100), tail=(80,100)
    ax_x = np.array([1.0])  # forward = right
    ax_y = np.array([0.0])

    # Partner ahead (right): (200, 100), centre=(100,100)
    dx, dy = np.array([100.0]), np.array([0.0])
    ego_x = dx * (-ax_y) + dy * ax_x
    ego_y = dx * ax_x + dy * ax_y
    assert ego_y[0] > 0, "Partner ahead should be positive ego_y"
    assert abs(ego_x[0]) < 1e-6, "Partner directly ahead should have ~0 ego_x"

    # Partner behind (left): (-100, 0)
    dx2, dy2 = np.array([-100.0]), np.array([0.0])
    ego_y2 = dx2 * ax_x + dy2 * ax_y
    assert ego_y2[0] < 0, "Partner behind should be negative ego_y"

    # Partner to the right (below in screen for rightward-facing mouse)
    # Mouse faces right -> body-right is downward -> (0, +50)
    dx3, dy3 = np.array([0.0]), np.array([50.0])
    ego_x3 = dx3 * (-ax_y) + dy3 * ax_x
    assert ego_x3[0] > 0, "Partner to body-right should be positive ego_x"


def test_adaptive_tolerance_uses_median():
    """Verify adaptive tolerance uses median, not mean, with upper cap."""
    from dlc_processor.core.social_behaviors import SocialBehaviors

    n = 100
    df_a = pd.DataFrame({
        "nose_x": np.full(n, 100.0), "nose_y": np.full(n, 100.0),
        "nose_likelihood": np.ones(n),
        "tailbase_x": np.full(n, 50.0), "tailbase_y": np.full(n, 100.0),
        "tailbase_likelihood": np.ones(n),
    })
    df_b = pd.DataFrame({
        "nose_x": np.full(n, 200.0), "nose_y": np.full(n, 100.0),
        "nose_likelihood": np.ones(n),
        "tailbase_x": np.full(n, 250.0), "tailbase_y": np.full(n, 100.0),
        "tailbase_likelihood": np.ones(n),
    })

    sb = SocialBehaviors(df_a, df_b, close_tol=0)  # trigger adaptive
    # Body length = 50 px, so 30% = 15 px
    assert 14.0 < sb.close_tol < 16.0, f"Expected ~15, got {sb.close_tol}"


def test_framewise_batch_export_uses_loaded_frame_times():
    from dlc_processor.core.batch_processor import compute_framewise_metrics_and_behaviors
    from dlc_processor.core.time_loader import FrameTimeData

    n = 4
    df_a = pd.DataFrame({
        "nose_x": [10.0, 11.0, 12.0, 13.0],
        "nose_y": [0.0, 0.0, 0.0, 0.0],
        "tailbase_x": [0.0, 1.0, 2.0, 3.0],
        "tailbase_y": [0.0, 0.0, 0.0, 0.0],
        "body_center_x": [5.0, 6.0, 7.0, 8.0],
        "body_center_y": [0.0, 0.0, 0.0, 0.0],
    })
    df_b = pd.DataFrame({
        "nose_x": [30.0, 31.0, 32.0, 33.0],
        "nose_y": [0.0, 0.0, 0.0, 0.0],
        "tailbase_x": [40.0, 41.0, 42.0, 43.0],
        "tailbase_y": [0.0, 0.0, 0.0, 0.0],
        "body_center_x": [35.0, 36.0, 37.0, 38.0],
        "body_center_y": [0.0, 0.0, 0.0, 0.0],
    })
    df_a.attrs["frame_numbers"] = np.array([10, 11, 12, 13])
    df_b.attrs["frame_numbers"] = np.array([10, 11, 12, 13])
    times = FrameTimeData(
        path="times.csv",
        frames=np.arange(20),
        times=np.arange(20, dtype=float) * 0.5,
    )

    result = compute_framewise_metrics_and_behaviors(
        {"mouse1": df_a, "mouse2": df_b},
        fps=30.0,
        time_data=times,
        compute_social=True,
        social_kwargs={"median_filter": 1, "likelihood_threshold": 0.0},
    )

    table = result["table"]
    assert table["frame_number"].tolist() == [10, 11, 12, 13]
    np.testing.assert_allclose(table["time_s"].to_numpy(), [5.0, 5.5, 6.0, 6.5])
    assert "mouse1__body_speed_px_s" in table.columns
    assert any(col.startswith("social_mouse1_vs_mouse2__") for col in table.columns)


def test_framewise_batch_frame_window_preserves_source_timestamps():
    from dlc_processor.core.batch_processor import compute_framewise_metrics_and_behaviors
    from dlc_processor.core.time_loader import FrameTimeData

    df_a = pd.DataFrame({
        "nose_x": [10.0, 11.0, 12.0, 13.0],
        "nose_y": [0.0, 0.0, 0.0, 0.0],
        "tailbase_x": [0.0, 1.0, 2.0, 3.0],
        "tailbase_y": [0.0, 0.0, 0.0, 0.0],
        "body_center_x": [5.0, 6.0, 7.0, 8.0],
        "body_center_y": [0.0, 0.0, 0.0, 0.0],
    })
    df_b = pd.DataFrame({
        "nose_x": [30.0, 31.0, 32.0, 33.0],
        "nose_y": [0.0, 0.0, 0.0, 0.0],
        "tailbase_x": [40.0, 41.0, 42.0, 43.0],
        "tailbase_y": [0.0, 0.0, 0.0, 0.0],
        "body_center_x": [35.0, 36.0, 37.0, 38.0],
        "body_center_y": [0.0, 0.0, 0.0, 0.0],
    })
    df_a.attrs["frame_numbers"] = np.array([100, 101, 102, 103])
    df_b.attrs["frame_numbers"] = np.array([100, 101, 102, 103])
    times = FrameTimeData(
        path="times.csv",
        frames=np.array([100, 101, 102, 103]),
        times=np.array([310.0, 310.033, 310.067, 310.100]),
    )

    result = compute_framewise_metrics_and_behaviors(
        {"mouse1": df_a, "mouse2": df_b},
        fps=30.0,
        time_data=times,
        compute_social=True,
        social_kwargs={"median_filter": 1, "likelihood_threshold": 0.0},
        analysis_frame_range=(101, 102),
    )

    table = result["table"]
    assert table["frame_number"].tolist() == [101, 102]
    np.testing.assert_allclose(table["time_s"].to_numpy(), [310.033, 310.067])
    assert any(col.startswith("social_mouse1_vs_mouse2__") for col in table.columns)


def test_framewise_batch_frame_window_end_zero_means_track_end():
    from dlc_processor.core.batch_processor import compute_framewise_metrics_and_behaviors
    from dlc_processor.core.time_loader import FrameTimeData

    df_a = pd.DataFrame({
        "nose_x": [10.0, 11.0, 12.0, 13.0],
        "nose_y": [0.0, 0.0, 0.0, 0.0],
        "tailbase_x": [0.0, 1.0, 2.0, 3.0],
        "tailbase_y": [0.0, 0.0, 0.0, 0.0],
    })
    df_b = pd.DataFrame({
        "nose_x": [30.0, 31.0, 32.0, 33.0],
        "nose_y": [0.0, 0.0, 0.0, 0.0],
        "tailbase_x": [40.0, 41.0, 42.0, 43.0],
        "tailbase_y": [0.0, 0.0, 0.0, 0.0],
    })
    df_a.attrs["frame_numbers"] = np.array([100, 101, 102, 103])
    df_b.attrs["frame_numbers"] = np.array([100, 101, 102, 103])
    times = FrameTimeData(
        path="times.csv",
        frames=np.array([100, 101, 102, 103]),
        times=np.array([310.0, 310.033, 310.067, 310.100]),
    )

    result = compute_framewise_metrics_and_behaviors(
        {"mouse1": df_a, "mouse2": df_b},
        fps=30.0,
        time_data=times,
        compute_social=False,
        analysis_frame_range=(102, 0),
    )

    table = result["table"]
    assert table["frame_number"].tolist() == [102, 103]
    np.testing.assert_allclose(table["time_s"].to_numpy(), [310.067, 310.100])


def test_parallel_batch_export_writes_glm_tables(tmp_path):
    from dlc_processor.core.batch_processor import batch_export_recordings

    def _df(offset: float) -> pd.DataFrame:
        n = 8
        return pd.DataFrame({
            "nose_x": np.linspace(offset + 10.0, offset + 17.0, n),
            "nose_y": np.zeros(n),
            "tailbase_x": np.linspace(offset, offset + 7.0, n),
            "tailbase_y": np.zeros(n),
            "body_center_x": np.linspace(offset + 5.0, offset + 12.0, n),
            "body_center_y": np.zeros(n),
        })

    records = [
        {"animal_dfs": {"mouse1": _df(0.0), "mouse2": _df(40.0)}, "metadata": {"condition": "a"}},
        {"animal_dfs": {"mouse1": _df(5.0), "mouse2": _df(50.0)}, "metadata": {"condition": "b"}},
    ]

    result = batch_export_recordings(
        records,
        custom_output_dir=str(tmp_path),
        output_mode="custom",
        compute_social=True,
        social_kwargs={"median_filter": 1, "likelihood_threshold": 0.0},
        n_jobs=2,
    )

    assert result["n_files"] == 2
    for exported in result["exported"]:
        assert Path(exported["table_path"]).exists()
        assert Path(exported["glm_csv_path"]).exists()
        assert Path(exported["glm_h5_path"]).exists()
    assert (tmp_path / "batch_export_index.csv").exists()
    assert any(col.startswith("social_mouse1_vs_mouse2__") for col in pd.read_csv(result["exported"][0]["glm_csv_path"]).columns)


def test_batch_export_can_fix_identities_from_masks_before_metrics(tmp_path):
    from dlc_processor.core.batch_processor import batch_export_recordings

    mouse1 = pd.DataFrame(
        {
            "nose_x": [101.0],
            "nose_y": [10.0],
            "tailbase_x": [100.0],
            "tailbase_y": [10.0],
        }
    )
    mouse2 = pd.DataFrame(
        {
            "nose_x": [11.0],
            "nose_y": [10.0],
            "tailbase_x": [10.0],
            "tailbase_y": [10.0],
        }
    )
    store = SimpleNamespace(
        frames={
            0: [
                SimpleNamespace(track_id=1, bbox=(0.0, 0.0, 20.0, 20.0)),
                SimpleNamespace(track_id=2, bbox=(90.0, 0.0, 20.0, 20.0)),
            ]
        }
    )

    result = batch_export_recordings(
        [{"animal_dfs": {"mouse1": mouse1, "mouse2": mouse2}, "mask_store": store}],
        custom_output_dir=str(tmp_path),
        output_mode="custom",
        compute_social=False,
        fix_identity_from_masks=True,
        n_jobs=1,
    )

    exported = result["exported"][0]
    glm = pd.read_csv(exported["glm_csv_path"])
    assert exported["mask_identity_frames_checked"] == 1
    assert exported["mask_identity_frames_corrected"] == 1
    assert Path(exported["mask_identity_cleaned_path"]).exists()
    assert glm.loc[0, "mouse1__nose_x"] == 11.0
    assert glm.loc[0, "mouse2__nose_x"] == 101.0


def test_glm_export_frame_window_keeps_external_time_alignment(tmp_path):
    from dlc_processor.core.batch_processor import batch_export_recordings
    from dlc_processor.core.time_loader import FrameTimeData

    def _df(offset: float) -> pd.DataFrame:
        df = pd.DataFrame({
            "nose_x": np.linspace(offset + 10.0, offset + 15.0, 6),
            "nose_y": np.zeros(6),
            "tailbase_x": np.linspace(offset, offset + 5.0, 6),
            "tailbase_y": np.zeros(6),
            "body_center_x": np.linspace(offset + 5.0, offset + 10.0, 6),
            "body_center_y": np.zeros(6),
        })
        df.attrs["frame_numbers"] = np.array([50, 51, 52, 53, 54, 55])
        return df

    time_data = FrameTimeData(
        path="frames.csv",
        frames=np.array([50, 51, 52, 53, 54, 55]),
        times=np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5]),
    )
    result = batch_export_recordings(
        [{
            "animal_dfs": {"mouse1": _df(0.0), "mouse2": _df(40.0)},
            "time_data": time_data,
            "metadata": {"condition": "a"},
        }],
        custom_output_dir=str(tmp_path),
        output_mode="custom",
        compute_social=True,
        social_kwargs={"median_filter": 1, "likelihood_threshold": 0.0},
        analysis_frame_range=(52, 54),
    )

    glm = pd.read_csv(result["exported"][0]["glm_csv_path"])
    assert glm["frame_number"].tolist() == [52, 53, 54]
    np.testing.assert_allclose(glm["time_s"].to_numpy(), [101.0, 101.5, 102.0])
    assert set(glm["timestamp_source"]) == {"loaded_frame_times"}
    assert result["exported"][0]["timestamp_source"] == "loaded_frame_times"
    assert result["exported"][0]["timestamp_count"] == 6
    assert any(col.startswith("social_mouse1_vs_mouse2__") for col in glm.columns)


def test_batch_export_h5_falls_back_without_pytables(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    from dlc_processor.core.batch_processor import batch_export_recordings

    def _raise_missing_tables(self, *args, **kwargs):
        raise ImportError("Missing optional dependency 'pytables'.")

    monkeypatch.setattr(pd.DataFrame, "to_hdf", _raise_missing_tables)
    df = pd.DataFrame({
        "nose_x": [1.0, 2.0],
        "nose_y": [1.0, 2.0],
        "tailbase_x": [0.0, 1.0],
        "tailbase_y": [1.0, 2.0],
    })

    result = batch_export_recordings(
        [{"animal_dfs": {"mouse1": df}}],
        custom_output_dir=str(tmp_path),
        output_mode="custom",
        compute_social=False,
        n_jobs=1,
    )

    assert result["n_files"] == 1
    exported = result["exported"][0]
    assert Path(exported["glm_csv_path"]).exists()
    assert Path(exported["glm_h5_path"]).exists()
    assert "h5py fallback" in exported["glm_h5_error"]
    with h5py.File(exported["glm_h5_path"], "r") as h5:
        assert "glm_ready" in h5
        assert h5["glm_ready"].attrs["n_rows"] == 2


def test_framewise_table_build_avoids_fragmentation_warning():
    from pandas.errors import PerformanceWarning

    from dlc_processor.core.batch_processor import _build_framewise_table

    n = 4
    animal_df = pd.DataFrame({
        "nose_x": np.arange(n, dtype=float),
        "nose_y": np.arange(n, dtype=float),
    })
    behaviors = {f"behavior_{idx}": np.array([True, False, True, False]) for idx in range(180)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerformanceWarning)
        table = _build_framewise_table(
            {"mouse1": animal_df},
            behaviors,
            frame_numbers=np.arange(n),
            time_s=np.arange(n, dtype=float),
        )

    assert len(table.columns) == 184
    assert not any(isinstance(item.message, PerformanceWarning) for item in caught)


def test_metadata_group_summary_exports_tables_and_figures(tmp_path):
    from dlc_processor.core.metadata_summary import export_group_summaries

    rows = []
    for recording, condition, speed, social_mean, frequency, bout_s, cumulative_s in [
        ("rec_a1", "control", 1.0, 0.10, 2.0, 0.50, 4.0),
        ("rec_a2", "control", 1.2, 0.12, 3.0, 0.55, 5.0),
        ("rec_b1", "treated", 2.0, 0.20, 5.0, 0.80, 8.0),
        ("rec_b2", "treated", 2.2, 0.23, 6.0, 0.90, 9.0),
    ]:
        rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse1",
            "mouseId": recording.split("_", 1)[0],
            "metric": "mouse1__body_speed_cm_s",
            "metric_base": "body_speed_cm_s",
            "mean": speed,
            "condition": condition,
            "is_behavior": False,
        })
        rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse2",
            "mouseId": "intruder",
            "metric": "mouse2__body_speed_cm_s",
            "metric_base": "body_speed_cm_s",
            "mean": 100.0,
            "condition": condition,
            "is_behavior": False,
        })
        rows.append({
            "recording": recording,
            "scope": "social_pair",
            "animal": "mouse1",
            "partner": "mouse2",
            "mouseId": recording.split("_", 1)[0],
            "partner_mouseId": "intruder",
            "metric": "social_mouse1_vs_mouse2__nose_to_nose",
            "metric_base": "nose_to_nose",
            "mean": social_mean,
            "fraction": social_mean,
            "bouts": frequency,
            "frequency_per_min": frequency,
            "mean_bout_duration_s": bout_s,
            "cumulative_duration_s": cumulative_s,
            "condition": condition,
            "is_behavior": True,
        })

    result = export_group_summaries(
        pd.DataFrame(rows),
        tmp_path,
        group_by=("condition",),
        animal_filter="mouse1",
    )

    expected_tables = {
        "batch_metrics_by_group.csv",
        "batch_social_behaviors_by_group.csv",
        "batch_group_comparisons.csv",
    }
    expected_figures = {
        "batch_metrics_overview.png",
        "batch_metrics_overview.pdf",
        "batch_social_frequency.png",
        "batch_social_frequency.pdf",
        "batch_social_mean_bout_duration_s.png",
        "batch_social_mean_bout_duration_s.pdf",
        "batch_social_cumulative_duration_s.png",
        "batch_social_cumulative_duration_s.pdf",
    }
    assert expected_tables.issubset({Path(path).name for path in result["tables"]})
    assert expected_figures.issubset({Path(path).name for path in result["figures"]})
    assert result["animal_filter"] == "mouse1"
    assert (tmp_path / "batch_metrics_by_group.csv").exists()
    assert (tmp_path / "batch_social_frequency.png").stat().st_size > 0
    assert (tmp_path / "batch_social_frequency.pdf").stat().st_size > 0
    assert export_group_summaries.__globals__["_EXPORT_DPI"] == 600
    metric_summary = pd.read_csv(tmp_path / "batch_metrics_by_group.csv")
    control_speed = metric_summary.loc[
        (metric_summary["metric_base"] == "body_speed_cm_s")
        & (metric_summary["group"] == "control"),
        "mean",
    ].iloc[0]
    assert control_speed < 2.0


def test_metadata_group_summary_splits_metric_scales_and_contact_states(tmp_path):
    from dlc_processor.core import metadata_summary

    rows = []
    for recording, condition, speed, distance, social_freq, contact_freq, mobile_freq in [
        ("rec_a1", "control", 1.0, 12000.0, 2.0, 30.0, 80.0),
        ("rec_a2", "control", 1.1, 12500.0, 2.5, 32.0, 78.0),
        ("rec_b1", "treated", 2.0, 20000.0, 5.0, 48.0, 90.0),
        ("rec_b2", "treated", 2.1, 20500.0, 5.5, 50.0, 88.0),
    ]:
        common = {
            "recording": recording,
            "animal": "mouse1",
            "mouseId": recording.split("_", 1)[0],
            "condition": condition,
        }
        rows.append({
            **common,
            "scope": "animal",
            "partner": "",
            "metric": "mouse1__body_speed_cm_s",
            "metric_base": "body_speed_cm_s",
            "mean": speed,
            "is_behavior": False,
        })
        rows.append({
            **common,
            "scope": "animal",
            "partner": "",
            "metric": "mouse1__distance_traveled_cm",
            "metric_base": "distance_traveled_cm",
            "mean": distance,
            "is_behavior": False,
        })
        rows.append({
            **common,
            "scope": "animal",
            "partner": "",
            "metric": "mouse1__mobile",
            "metric_base": "mobile",
            "mean": 0.9,
            "fraction": 0.9,
            "bouts": mobile_freq,
            "frequency_per_min": mobile_freq,
            "mean_bout_duration_s": 3.0,
            "cumulative_duration_s": 300.0,
            "is_behavior": True,
        })
        rows.append({
            **common,
            "scope": "social_pair",
            "partner": "mouse2",
            "partner_mouseId": "intruder",
            "metric": "social_mouse1_vs_mouse2__nose_to_nose",
            "metric_base": "nose_to_nose",
            "mean": 0.1,
            "fraction": 0.1,
            "bouts": social_freq,
            "frequency_per_min": social_freq,
            "mean_bout_duration_s": 0.5,
            "cumulative_duration_s": 4.0,
            "is_behavior": True,
        })
        rows.append({
            **common,
            "scope": "social_pair",
            "partner": "mouse2",
            "partner_mouseId": "intruder",
            "metric": "social_mouse1_vs_mouse2__mask_contact",
            "metric_base": "mask_contact",
            "mean": 0.8,
            "fraction": 0.8,
            "bouts": contact_freq,
            "frequency_per_min": contact_freq,
            "mean_bout_duration_s": 6.0,
            "cumulative_duration_s": 250.0,
            "is_behavior": True,
        })

    result = metadata_summary.export_group_summaries(
        pd.DataFrame(rows),
        tmp_path,
        group_by=("condition",),
        animal_filter="mouse1",
    )

    figures = {Path(path).name for path in result["figures"]}
    assert "batch_metrics_overview.png" in figures
    assert "batch_metrics_overview.pdf" in figures
    assert "batch_cumulative_metrics.png" in figures
    assert "batch_cumulative_metrics.pdf" in figures
    assert "batch_mask_contact_frequency.png" in figures
    assert "batch_mask_contact_frequency.pdf" in figures

    social = pd.read_csv(tmp_path / "batch_social_behaviors_by_group.csv")
    assert set(social["metric_base"]) == {"nose_to_nose"}

    contact = pd.read_csv(tmp_path / "batch_mask_contact_by_group.csv")
    assert set(contact["metric_base"]) == {"mask_contact"}
    assert set(contact["category"]) == {"contact_signal"}

    metric_summary = pd.read_csv(tmp_path / "batch_metrics_by_group.csv")
    overview = metadata_summary._filter_metric_overview_table(metric_summary, cumulative=False)
    cumulative = metadata_summary._filter_metric_overview_table(metric_summary, cumulative=True)
    assert "distance_traveled_cm" not in set(overview["metric_base"])
    assert "distance_traveled_cm" in set(cumulative["metric_base"])


def test_social_summary_drops_redundant_passives_and_keeps_fighting():
    from dlc_processor.core import metadata_summary

    rows = []
    for metric, mean in [
        ("passive_anogenital", 0.8),
        ("passive_investigation", 0.75),
        ("passive_being_followed", 0.7),
        ("passive_being_chased", 0.65),
        ("passive_withdrawal", 0.55),
        ("b_nose2anogenital_a", 0.6),
        ("b_following_a", 0.5),
        ("fighting", 0.1),
    ]:
        rows.append({
            "recording": "rec",
            "scope": "social_pair",
            "animal": "mouse1",
            "partner": "mouse2",
            "metric": f"social_mouse1_vs_mouse2__{metric}",
            "metric_base": metric,
            "mean": mean,
            "bouts": 1,
            "is_behavior": True,
        })

    source = metadata_summary._social_behavior_summary_source(pd.DataFrame(rows))
    metrics = set(source["metric_base"].astype(str))
    assert not any(metric.startswith("passive_") for metric in metrics)
    assert "b_nose2anogenital_a" in metrics
    assert "b_following_a" in metrics
    assert "fighting" in metrics
    assert "attacks" in metrics


def test_social_overview_metric_selection_reserves_fighting_slot():
    from dlc_processor.core import metadata_summary

    rows = [
        {"metric_base": f"metric_{idx}", "mean": float(100 - idx), "category": "social_behavior"}
        for idx in range(12)
    ]
    rows.append({"metric_base": "fighting", "mean": 0.01, "category": "social_behavior"})
    selected = metadata_summary._overview_metric_selection(pd.DataFrame(rows), max_metrics=12)
    assert len(selected) == 12
    assert "fighting" in selected


def test_transition_state_columns_exclude_mask_contact_signal():
    from dlc_processor.core.metadata_summary import _social_behavior_columns_by_pair

    table = pd.DataFrame({
        "social_mouse1_vs_mouse2__mask_contact": [1, 1, 1, 0],
        "social_mouse1_vs_mouse2__nose_to_nose": [0, 1, 0, 0],
        "social_mouse1_vs_mouse2__passive_investigation": [1, 0, 1, 0],
        "social_mouse1_vs_mouse2__partner_distance_cm": [1.0, 2.0, 3.0, 4.0],
    })

    columns = _social_behavior_columns_by_pair(table)
    assert columns == {"mouse1_vs_mouse2": ["social_mouse1_vs_mouse2__nose_to_nose"]}


def test_metadata_mouse_id_filter_is_distinct_from_local_animal_label(tmp_path):
    from dlc_processor.core.metadata_summary import export_group_summaries

    rows = []
    for recording, mouse_id, condition, speed in [
        ("31098_2_trial", "31098", "het", 1.0),
        ("31096_2_trial", "31096", "wt", 2.0),
    ]:
        rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse1",
            "mouseId": mouse_id,
            "metric": "mouse1__body_speed_cm_s",
            "metric_base": "body_speed_cm_s",
            "mean": speed,
            "condition": condition,
            "is_behavior": False,
        })
        rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse2",
            "mouseId": f"{mouse_id}_partner",
            "metric": "mouse2__body_speed_cm_s",
            "metric_base": "body_speed_cm_s",
            "mean": 100.0,
            "condition": condition,
            "is_behavior": False,
        })

    result = export_group_summaries(
        pd.DataFrame(rows),
        tmp_path,
        group_by=("condition",),
        mouse_id_filter="31098",
    )

    assert result["mouse_id_filter"] == "31098"
    metric_summary = pd.read_csv(tmp_path / "batch_metrics_by_group.csv")
    assert set(metric_summary["group"]) == {"het"}
    speed_mean = metric_summary.loc[
        metric_summary["metric_base"] == "body_speed_cm_s",
        "mean",
    ].iloc[0]
    assert speed_mean == 1.0


def test_group_position_maps_export_average_by_condition(tmp_path):
    from dlc_processor.core.metadata_summary import export_group_position_maps

    exported = []
    summary_rows = []
    for idx, (recording, condition, x_offset) in enumerate([
        ("rec_a1", "control", 0.0),
        ("rec_a2", "control", 1.0),
        ("rec_b1", "treated", 7.0),
        ("rec_b2", "treated", 8.0),
    ]):
        table = pd.DataFrame({
            "frame_number": np.arange(5),
            "time_s": np.arange(5, dtype=float),
            "video_path": "",
            "mouse1__body_center_x": np.linspace(x_offset, x_offset + 1.0, 5),
            "mouse1__body_center_y": np.linspace(0.0, 1.0, 5),
            "mouse2__body_center_x": np.linspace(100.0, 101.0, 5),
            "mouse2__body_center_y": np.linspace(100.0, 101.0, 5),
        })
        path = tmp_path / f"{recording}_metrics_behavior.csv"
        table.to_csv(path, index=False)
        exported.append({"name": recording, "table_path": str(path)})
        summary_rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse1",
            "mouseId": recording.split("_", 1)[0],
            "metric_base": "body_speed_px_s",
            "mean": 1.0,
            "condition": condition,
            "is_behavior": False,
        })
        summary_rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse2",
            "mouseId": "intruder",
            "metric_base": "body_speed_px_s",
            "mean": 1.0,
            "condition": "wrong_group",
            "is_behavior": False,
        })

    result = export_group_position_maps(
        exported,
        pd.DataFrame(summary_rows),
        tmp_path,
        group_by=("condition",),
        animal_filter="mouse1",
        bins=12,
    )

    assert result["n_maps"] == 2
    assert result["animal_filter"] == "mouse1"
    index = pd.read_csv(tmp_path / "position_maps" / "batch_position_maps_index.csv")
    assert set(index["group"]) == {"control", "treated"}
    assert set(index["animal"]) == {"mouse1"}
    assert all(Path(path).exists() for path in result["tables"])
    assert all(Path(path).exists() for path in result["figures"])
    assert any(Path(path).suffix == ".pdf" for path in result["figures"])


def test_position_map_uses_bundled_mouse_top_asset():
    from dlc_processor.core.metadata_summary import _mouse_top_svg_path

    svg_path = _mouse_top_svg_path()
    assert svg_path is not None
    assert svg_path.exists()
    assert svg_path.name == "mouse_top.svg"
    # Standalone ships the asset under dlc_processor/assets/.
    assert svg_path.parts[-2] == "assets"


def test_transition_export_adds_entropy_switch_rate_and_heatmap(tmp_path):
    from dlc_processor.core.metadata_summary import export_group_transitions

    exported = []
    summary_rows = []
    for recording, condition, states in [
        ("31098_2_trial", "het", [1, 1, 0, 0, 1, 0]),
        ("31096_2_trial", "wt", [0, 0, 1, 1, 0, 1]),
    ]:
        table = pd.DataFrame({
            "frame_number": np.arange(len(states)),
            "time_s": np.arange(len(states), dtype=float),
            "social_mouse1_vs_mouse2__nose2nose": states,
            "social_mouse1_vs_mouse2__a_following_b": [1 - int(v) for v in states],
        })
        path = tmp_path / f"{recording}_metrics_behavior.csv"
        table.to_csv(path, index=False)
        exported.append({"name": recording, "table_path": str(path)})
        summary_rows.append({
            "recording": recording,
            "scope": "animal",
            "animal": "mouse1",
            "mouseId": recording.split("_", 1)[0],
            "metric_base": "body_speed_px_s",
            "mean": 1.0,
            "condition": condition,
            "is_behavior": False,
        })
        summary_rows.append({
            "recording": recording,
            "scope": "social",
            "animal": "mouse1",
            "partner": "mouse2",
            "mouseId": recording.split("_", 1)[0],
            "partner_mouseId": "intruder",
            "metric_base": "nose2nose",
            "mean": 0.5,
            "condition": condition,
            "is_behavior": True,
        })

    result = export_group_transitions(
        exported,
        pd.DataFrame(summary_rows),
        tmp_path,
        group_by=("condition",),
        animal_filter="mouse1",
        plot_style="box_strip",
        tab10_color_map="het=1, wt=2",
    )

    names = {Path(path).name for path in result["tables"]}
    assert "batch_transition_metrics.csv" in names
    assert "batch_transition_metrics_by_group.csv" in names
    assert "batch_transition_probabilities.csv" in names
    metrics = pd.read_csv(tmp_path / "transitions" / "batch_transition_metrics.csv")
    assert {"transition_entropy_bits", "switch_rate_per_min"}.issubset(metrics.columns)
    assert metrics["switch_rate_per_min"].gt(0).all()
    probabilities = pd.read_csv(tmp_path / "transitions" / "batch_transition_probabilities.csv")
    assert {"probability", "from_state_fraction", "transition_mass", "transition_count"}.issubset(probabilities.columns)
    assert probabilities["transition_mass"].le(probabilities["probability"] + 1e-12).all()
    figure_names = {Path(path).name for path in result["figures"]}
    assert "batch_transition_probability_average_network.png" in figure_names
    assert "batch_transition_probability_average_network.pdf" in figure_names
    assert "batch_transition_probability_group_networks.png" in figure_names
    assert "batch_transition_probability_group_networks.pdf" in figure_names
    assert any(name.startswith("batch_transition_probability_") for name in figure_names)


def test_transition_plot_uses_frequency_weighted_mass_and_wt_minus_het():
    from dlc_processor.core import metadata_summary

    probabilities = pd.DataFrame({
        "group": ["WT", "WT", "Het", "Het"],
        "from_state": ["fighting", "nose2nose", "fighting", "nose2nose"],
        "to_state": ["nose2nose", "fighting", "nose2nose", "fighting"],
        "probability": [1.0, 0.4, 1.0, 0.2],
        "transition_mass": [0.01, 0.20, 0.02, 0.10],
    })
    states = ["nose2nose", "fighting"]
    wt = metadata_summary._transition_probability_matrix(
        probabilities[probabilities["group"] == "WT"],
        states,
        value_col=metadata_summary._transition_plot_value_col(probabilities),
    )
    het = metadata_summary._transition_probability_matrix(
        probabilities[probabilities["group"] == "Het"],
        states,
        value_col=metadata_summary._transition_plot_value_col(probabilities),
    )

    assert metadata_summary._transition_plot_value_col(probabilities) == "transition_mass"
    assert wt.loc["fighting", "nose2nose"] == 0.01
    assert wt.loc["nose2nose", "fighting"] == 0.20
    assert (wt - het).loc["nose2nose", "fighting"] > 0.0
    assert (wt - het).loc["fighting", "nose2nose"] < 0.0
