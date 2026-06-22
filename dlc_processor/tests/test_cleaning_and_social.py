from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from dlc_processor.core.data_cleaner import fix_impossible_all_animals
from dlc_processor.core.kinematics import compute_kinematics
from dlc_processor.core.social_behaviors import SocialBehaviors


def _mouse_track_df() -> pd.DataFrame:
    n = 5
    tail_x = np.array([0, 10, 20, 30, 40], dtype=float)
    neck_x = np.array([20, 30, 40, 50, 60], dtype=float)
    nose_x = np.array([30, 40, 50, 60, 70], dtype=float)

    df = pd.DataFrame(
        {
            "tail_x": tail_x,
            "tail_y": np.full(n, 50.0),
            "tail_likelihood": np.full(n, 0.95),
            "neck_x": neck_x,
            "neck_y": np.full(n, 50.0),
            "neck_likelihood": np.full(n, 0.95),
            "nose_x": nose_x,
            "nose_y": np.full(n, 50.0),
            "nose_likelihood": np.full(n, 0.95),
            "left_ear_x": neck_x,
            "left_ear_y": np.full(n, 45.0),
            "left_ear_likelihood": np.full(n, 0.95),
            "right_ear_x": neck_x,
            "right_ear_y": np.full(n, 55.0),
            "right_ear_likelihood": np.full(n, 0.95),
            "left_hip_x": tail_x + 5.0,
            "left_hip_y": np.full(n, 45.0),
            "left_hip_likelihood": np.full(n, 0.95),
            "right_hip_x": tail_x + 5.0,
            "right_hip_y": np.full(n, 55.0),
            "right_hip_likelihood": np.full(n, 0.95),
        }
    )
    return df


def test_fix_impossible_replaces_nose_behind_neck_by_interpolation() -> None:
    df = _mouse_track_df()
    df.loc[2, "nose_x"] = 25.0

    fixed = fix_impossible_all_animals({"mouse": df}, max_gap_frames=3)["mouse"]

    assert fixed.loc[2, "nose_x"] > fixed.loc[2, "neck_x"]
    assert np.isclose(fixed.loc[2, "nose_x"], 50.0)


def test_fix_identities_from_masks_uses_mask_track_ids_as_ground_truth() -> None:
    from dlc_processor.core.data_cleaner import fix_identities_from_masks

    mouse1 = pd.DataFrame(
        {
            "nose_x": [101.0],
            "nose_y": [10.0],
            "nose_likelihood": [0.9],
            "tail_x": [100.0],
            "tail_y": [10.0],
            "tail_likelihood": [0.9],
        }
    )
    mouse2 = pd.DataFrame(
        {
            "nose_x": [11.0],
            "nose_y": [10.0],
            "nose_likelihood": [0.8],
            "tail_x": [10.0],
            "tail_y": [10.0],
            "tail_likelihood": [0.8],
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

    fixed, summary = fix_identities_from_masks(
        {"mouse1": mouse1, "mouse2": mouse2},
        store,
        animal_to_track_id={"mouse1": 1, "mouse2": 2},
        max_workers=1,
    )

    assert summary["frames_checked"] == 1
    assert summary["frames_corrected"] == 1
    assert fixed["mouse1"].loc[0, "nose_x"] == 11.0
    assert fixed["mouse1"].loc[0, "nose_likelihood"] == 0.8
    assert fixed["mouse2"].loc[0, "nose_x"] == 101.0


def test_fix_identities_from_masks_uses_exact_masks_when_bboxes_overlap() -> None:
    from dlc_processor.core.data_cleaner import fix_identities_from_masks

    mouse1 = pd.DataFrame(
        {
            "nose_x": [45.0],
            "nose_y": [20.0],
            "nose_likelihood": [0.9],
            "tail_x": [47.0],
            "tail_y": [30.0],
            "tail_likelihood": [0.9],
        }
    )
    mouse2 = pd.DataFrame(
        {
            "nose_x": [20.0],
            "nose_y": [20.0],
            "nose_likelihood": [0.8],
            "tail_x": [22.0],
            "tail_y": [30.0],
            "tail_likelihood": [0.8],
        }
    )
    mouse1.attrs["frame_numbers"] = np.array([100])
    mouse2.attrs["frame_numbers"] = np.array([100])
    store = SimpleNamespace(
        frames={
            100: [
                SimpleNamespace(
                    track_id=1,
                    bbox=(0.0, 0.0, 50.0, 80.0),
                    size=(80, 80),
                    segmentation=[[0, 0, 30, 0, 30, 80, 0, 80]],
                ),
                SimpleNamespace(
                    track_id=2,
                    bbox=(10.0, 0.0, 50.0, 80.0),
                    size=(80, 80),
                    segmentation=[[35, 0, 65, 0, 65, 80, 35, 80]],
                ),
            ]
        }
    )

    fixed, summary = fix_identities_from_masks(
        {"mouse1": mouse1, "mouse2": mouse2},
        store,
        animal_to_track_id={"mouse1": 1, "mouse2": 2},
        max_workers=1,
    )

    assert summary["frames_checked"] == 1
    assert summary["frames_corrected"] == 1
    assert summary["exact_mask_frames_checked"] == 1
    assert fixed["mouse1"].loc[0, "nose_x"] == 20.0
    assert fixed["mouse2"].loc[0, "nose_x"] == 45.0


def test_fix_identities_from_masks_handles_zero_based_mask_track_ids() -> None:
    from dlc_processor.core.data_cleaner import fix_identities_from_masks

    mouse1 = pd.DataFrame(
        {
            "nose_x": [101.0],
            "nose_y": [10.0],
            "nose_likelihood": [0.9],
            "tail_x": [100.0],
            "tail_y": [10.0],
            "tail_likelihood": [0.9],
        }
    )
    mouse2 = pd.DataFrame(
        {
            "nose_x": [11.0],
            "nose_y": [10.0],
            "nose_likelihood": [0.8],
            "tail_x": [10.0],
            "tail_y": [10.0],
            "tail_likelihood": [0.8],
        }
    )
    store = SimpleNamespace(
        frames={
            0: [
                SimpleNamespace(track_id=0, bbox=(0.0, 0.0, 20.0, 20.0)),
                SimpleNamespace(track_id=1, bbox=(90.0, 0.0, 20.0, 20.0)),
            ]
        }
    )

    fixed, summary = fix_identities_from_masks(
        {"mouse1": mouse1, "mouse2": mouse2},
        store,
        max_workers=1,
    )

    assert summary["frames_checked"] == 1
    assert summary["frames_corrected"] == 1
    assert fixed["mouse1"].loc[0, "nose_x"] == 11.0
    assert fixed["mouse2"].loc[0, "nose_x"] == 101.0


def test_fix_impossible_restores_same_side_ear_and_infers_nose_from_ears() -> None:
    df = _mouse_track_df()
    df["nose_x"] = np.nan
    df["nose_y"] = np.nan
    df["nose_likelihood"] = 0.0
    df.loc[2, "right_ear_y"] = 45.0

    fixed = fix_impossible_all_animals({"mouse": df}, max_gap_frames=3)["mouse"]

    assert np.isfinite(fixed["nose_x"]).all()
    assert np.isfinite(fixed["nose_y"]).all()
    assert (fixed["nose_x"] > fixed["neck_x"]).all()
    assert fixed.loc[2, "right_ear_y"] > fixed.loc[2, "neck_y"]


def _social_df(
    nose_x: list[float],
    tail_x: list[float],
    neck_x: list[float],
    hip_x: list[float],
) -> pd.DataFrame:
    n = len(nose_x)
    return pd.DataFrame(
        {
            "nose_x": np.asarray(nose_x, dtype=float),
            "nose_y": np.zeros(n, dtype=float),
            "tail_x": np.asarray(tail_x, dtype=float),
            "tail_y": np.zeros(n, dtype=float),
            "neck_x": np.asarray(neck_x, dtype=float),
            "neck_y": np.zeros(n, dtype=float),
            "left_hip_x": np.asarray(hip_x, dtype=float),
            "left_hip_y": np.zeros(n, dtype=float),
            "right_hip_x": np.asarray(hip_x, dtype=float),
            "right_hip_y": np.zeros(n, dtype=float),
        }
    )


def _social_df_xy(
    *,
    nose: tuple[float, float],
    rear: tuple[float, float],
    neck: tuple[float, float],
    left_hip: tuple[float, float],
    right_hip: tuple[float, float],
    tail_name: str = "tail",
    n: int = 5,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nose_x": np.full(n, nose[0], dtype=float),
            "nose_y": np.full(n, nose[1], dtype=float),
            f"{tail_name}_x": np.full(n, rear[0], dtype=float),
            f"{tail_name}_y": np.full(n, rear[1], dtype=float),
            "neck_x": np.full(n, neck[0], dtype=float),
            "neck_y": np.full(n, neck[1], dtype=float),
            "left_hip_x": np.full(n, left_hip[0], dtype=float),
            "left_hip_y": np.full(n, left_hip[1], dtype=float),
            "right_hip_x": np.full(n, right_hip[0], dtype=float),
            "right_hip_y": np.full(n, right_hip[1], dtype=float),
        }
    )


def _pose_sequence(
    center_x: np.ndarray,
    center_y: np.ndarray,
    angle_deg: np.ndarray,
    *,
    body_len: float = 20.0,
) -> pd.DataFrame:
    center_x = np.asarray(center_x, dtype=float)
    center_y = np.asarray(center_y, dtype=float)
    angle = np.deg2rad(np.asarray(angle_deg, dtype=float))
    ax = np.cos(angle)
    ay = np.sin(angle)
    px = -ay
    py = ax
    half = body_len / 2.0
    nose_x = center_x + ax * half
    nose_y = center_y + ay * half
    tail_x = center_x - ax * half
    tail_y = center_y - ay * half
    return pd.DataFrame({
        "nose_x": nose_x,
        "nose_y": nose_y,
        "tailbase_x": tail_x,
        "tailbase_y": tail_y,
        "neck_x": center_x + ax * (body_len * 0.20),
        "neck_y": center_y + ay * (body_len * 0.20),
        "left_hip_x": center_x - ax * (body_len * 0.28) + px * 3.0,
        "left_hip_y": center_y - ay * (body_len * 0.28) + py * 3.0,
        "right_hip_x": center_x - ax * (body_len * 0.28) - px * 3.0,
        "right_hip_y": center_y - ay * (body_len * 0.28) - py * 3.0,
    })


def test_nose_to_body_excludes_anogenital_overlap() -> None:
    df_a = _social_df(
        nose_x=[0, 0, 0, 0, 0],
        tail_x=[-20, -20, -20, -20, -20],
        neck_x=[-10, -10, -10, -10, -10],
        hip_x=[-15, -15, -15, -15, -15],
    )
    df_b = _social_df(
        nose_x=[100, 100, 100, 100, 100],
        tail_x=[100, 100, 1, 100, 100],
        neck_x=[120, 120, 120, 120, 120],
        hip_x=[130, 130, 1, 130, 130],
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=10.0,
        median_filter=1,
        follow_window=2,
    ).compute_all()

    assert "a_nose2anogenital_b" in arrays
    assert arrays["a_nose2anogenital_b"][2]
    assert not arrays["a_nose2body_b"][2]


def test_nose_to_nose_with_curled_tail_does_not_trigger_anogenital_or_body() -> None:
    df_a = _social_df_xy(
        nose=(0.0, 0.0),
        rear=(-20.0, 0.0),
        neck=(-8.0, 0.0),
        left_hip=(-15.0, -2.0),
        right_hip=(-15.0, 2.0),
    )
    df_b = _social_df_xy(
        nose=(2.0, 0.0),
        rear=(1.0, 0.0),
        neck=(6.0, 0.0),
        left_hip=(18.0, -2.0),
        right_hip=(18.0, 2.0),
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=7.0,
        median_filter=1,
        follow_window=2,
    ).compute_all()

    assert arrays["nose2nose"].all()
    assert not arrays["a_nose2anogenital_b"].any()
    assert not arrays["a_nose2body_b"].any()


def test_generic_tail_tip_proximity_is_not_anogenital() -> None:
    df_a = _social_df_xy(
        nose=(1.0, 0.0),
        rear=(-15.0, 0.0),
        neck=(-8.0, 0.0),
        left_hip=(-11.0, -2.0),
        right_hip=(-11.0, 2.0),
    )
    df_b = _social_df_xy(
        nose=(40.0, 0.0),
        rear=(0.0, 0.0),
        neck=(28.0, 0.0),
        left_hip=(10.0, -2.0),
        right_hip=(10.0, 2.0),
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        median_filter=1,
        follow_window=2,
    ).compute_all()

    assert not arrays["a_nose2anogenital_b"].any()
    assert not arrays["a_nose2body_b"].any()


def test_lateral_rear_contact_is_not_anogenital() -> None:
    df_a = _social_df_xy(
        nose=(10.0, 8.0),
        rear=(-10.0, 8.0),
        neck=(0.0, 8.0),
        left_hip=(-5.0, 6.0),
        right_hip=(-5.0, 10.0),
    )
    df_b = _social_df_xy(
        nose=(40.0, 0.0),
        rear=(10.0, 0.0),
        neck=(30.0, 0.0),
        left_hip=(14.0, -2.0),
        right_hip=(14.0, 2.0),
        tail_name="tailbase",
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=10.0,
        median_filter=1,
        follow_window=2,
    ).compute_all()

    assert not arrays["a_nose2anogenital_b"].any()


def test_following_with_nose_near_tail_is_anogenital() -> None:
    n = 18
    t = np.arange(n, dtype=float)
    # B moves rightward. A tracks B from behind with A's nose close to B's rear.
    df_b = pd.DataFrame({
        "nose_x": 130.0 + t * 3.0,
        "nose_y": np.zeros(n),
        "tailbase_x": 100.0 + t * 3.0,
        "tailbase_y": np.zeros(n),
        "neck_x": 120.0 + t * 3.0,
        "neck_y": np.zeros(n),
        "left_hip_x": 105.0 + t * 3.0,
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": 105.0 + t * 3.0,
        "right_hip_y": np.full(n, 3.0),
    })
    df_a = pd.DataFrame({
        "nose_x": df_b["tailbase_x"].to_numpy() - 3.0,
        "nose_y": np.zeros(n),
        "tailbase_x": df_b["tailbase_x"].to_numpy() - 28.0,
        "tailbase_y": np.zeros(n),
        "neck_x": df_b["tailbase_x"].to_numpy() - 12.0,
        "neck_y": np.zeros(n),
        "left_hip_x": df_b["tailbase_x"].to_numpy() - 23.0,
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": df_b["tailbase_x"].to_numpy() - 23.0,
        "right_hip_y": np.full(n, 3.0),
    })

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=8.0,
        follow_tol=12.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
    ).compute_all()

    assert arrays["a_nose2anogenital_b"].any()
    assert arrays["a_following_b"].any()


def test_fast_following_is_chasing() -> None:
    n = 24
    t = np.arange(n, dtype=float)
    df_b = pd.DataFrame({
        "nose_x": 130.0 + t * 5.0,
        "nose_y": np.zeros(n),
        "tailbase_x": 100.0 + t * 5.0,
        "tailbase_y": np.zeros(n),
        "neck_x": 120.0 + t * 5.0,
        "neck_y": np.zeros(n),
        "left_hip_x": 105.0 + t * 5.0,
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": 105.0 + t * 5.0,
        "right_hip_y": np.full(n, 3.0),
    })
    df_a = pd.DataFrame({
        "nose_x": df_b["tailbase_x"].to_numpy() - 5.0,
        "nose_y": np.zeros(n),
        "tailbase_x": df_b["tailbase_x"].to_numpy() - 30.0,
        "tailbase_y": np.zeros(n),
        "neck_x": df_b["tailbase_x"].to_numpy() - 15.0,
        "neck_y": np.zeros(n),
        "left_hip_x": df_b["tailbase_x"].to_numpy() - 25.0,
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": df_b["tailbase_x"].to_numpy() - 25.0,
        "right_hip_y": np.full(n, 3.0),
    })

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=8.0,
        follow_tol=14.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
    ).compute_all()

    assert arrays["a_following_b"].any()
    assert arrays["a_chasing_b"].any()
    assert not arrays["b_chasing_a"].any()
    assert not arrays["fighting"].any()


def test_slow_following_is_not_chasing() -> None:
    n = 24
    t = np.arange(n, dtype=float)
    df_b = pd.DataFrame({
        "nose_x": 130.0 + t * 0.3,
        "nose_y": np.zeros(n),
        "tailbase_x": 100.0 + t * 0.3,
        "tailbase_y": np.zeros(n),
        "neck_x": 120.0 + t * 0.3,
        "neck_y": np.zeros(n),
        "left_hip_x": 105.0 + t * 0.3,
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": 105.0 + t * 0.3,
        "right_hip_y": np.full(n, 3.0),
    })
    df_a = pd.DataFrame({
        "nose_x": df_b["tailbase_x"].to_numpy() - 5.0,
        "nose_y": np.zeros(n),
        "tailbase_x": df_b["tailbase_x"].to_numpy() - 30.0,
        "tailbase_y": np.zeros(n),
        "neck_x": df_b["tailbase_x"].to_numpy() - 15.0,
        "neck_y": np.zeros(n),
        "left_hip_x": df_b["tailbase_x"].to_numpy() - 25.0,
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": df_b["tailbase_x"].to_numpy() - 25.0,
        "right_hip_y": np.full(n, 3.0),
    })

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=8.0,
        follow_tol=14.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
    ).compute_all()

    assert arrays["a_following_b"].any()
    assert not arrays["a_chasing_b"].any()


def test_close_erratic_high_motion_contact_is_fighting_not_chasing() -> None:
    n = 16
    df_a = _pose_sequence(
        np.array([0, 5, -1, 6, 0, 7, 1, 8, 0, 6, -2, 5, 0, 7, 1, 6], dtype=float),
        np.array([0, 2, -2, 3, -3, 2, -2, 3, 0, -3, 2, -2, 3, -1, 2, -2], dtype=float),
        np.array([0, 90, -80, 120, -100, 70, -120, 100, -60, 140, -100, 80, -120, 100, -80, 90], dtype=float),
    )
    df_b = _pose_sequence(
        np.array([8, 3, 9, 2, 10, 4, 9, 3, 8, 2, 9, 4, 8, 3, 10, 4], dtype=float),
        np.array([1, -2, 2, -3, 3, -2, 2, -1, 1, 3, -2, 2, -3, 1, -2, 2], dtype=float),
        np.array([180, 80, -90, 60, -120, 110, -80, 120, -100, 70, -130, 90, -70, 120, -90, 80], dtype=float),
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=8.0,
        follow_tol=12.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert arrays["fighting"].any()
    assert not arrays["a_chasing_b"].any()
    assert not arrays["b_chasing_a"].any()


def test_quick_escape_after_contact_is_withdrawal() -> None:
    n = 12
    df_b = _pose_sequence(np.zeros(n), np.zeros(n), np.zeros(n))
    df_a = _pose_sequence(
        np.array([5, 5, 5, 6, 30, 45, 60, 75, 90, 105, 120, 135], dtype=float),
        np.zeros(n),
        np.array([180, 180, 180, 180, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    )
    mask_contact = np.array([True, True, True, True, False, False, False, False, False, False, False, False])

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=6.0,
        follow_tol=10.0,
        follow_window=5,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
        mask_contact=mask_contact,
    ).compute_all()

    assert arrays["a_withdrawal_after_contact_b"].any()
    assert not arrays["b_withdrawal_after_contact_a"].any()


def test_directional_approach_requires_focal_distance_closing() -> None:
    n = 14
    df_a = _pose_sequence(np.linspace(-115.0, -35.0, n), np.zeros(n), np.zeros(n))
    df_b = _pose_sequence(np.zeros(n), np.zeros(n), np.full(n, 180.0))

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=6.0,
        follow_tol=12.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
    ).compute_all()

    assert arrays["a_approaches_b"].any()
    assert not arrays["b_approaches_a"].any()
    assert not arrays["a_escapes_b"].any()


def test_general_withdrawal_is_retreat_without_partner_pressure() -> None:
    n = 14
    df_a = _pose_sequence(np.linspace(35.0, 120.0, n), np.zeros(n), np.zeros(n))
    df_b = _pose_sequence(np.zeros(n), np.zeros(n), np.zeros(n))

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=6.0,
        follow_tol=12.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
    ).compute_all()

    assert arrays["a_withdraws_from_b"].any()
    assert not arrays["a_escapes_b"].any()
    assert not arrays["b_withdraws_from_a"].any()


def test_escape_requires_fast_retreat_with_partner_pressure() -> None:
    n = 12
    df_a = _pose_sequence(
        np.array([12, 14, 18, 30, 50, 75, 102, 130, 158, 186, 214, 242], dtype=float),
        np.zeros(n),
        np.zeros(n),
    )
    df_b = _pose_sequence(
        np.array([0, 5, 12, 20, 32, 47, 63, 79, 95, 111, 127, 143], dtype=float),
        np.zeros(n),
        np.zeros(n),
    )
    mask_contact = np.array([True, True, False, False, False, False, False, False, False, False, False, False])

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=6.0,
        follow_tol=12.0,
        follow_window=4,
        min_follow_frames=2,
        stationary_threshold=0.1,
        median_filter=1,
        mask_contact=mask_contact,
    ).compute_all()

    assert arrays["a_escapes_b"].any()
    assert not arrays["b_escapes_a"].any()
    assert arrays["b_approaches_a"].any()


def test_mask_edges_touch_detects_social_contact() -> None:
    from dlc_processor.core.mask_social import pair_mask_contact

    class Store:
        def masks_for_frame(self, _frame):
            mask_a = np.zeros((20, 20), dtype=bool)
            mask_b = np.zeros((20, 20), dtype=bool)
            mask_a[5:12, 3:8] = True
            mask_b[5:12, 10:15] = True
            return [
                (1, mask_a, (3, 5, 5, 7), 1.0),
                (2, mask_b, (10, 5, 5, 7), 1.0),
            ]

    df_a = pd.DataFrame({"body_center_x": [5.0], "body_center_y": [8.0]})
    df_b = pd.DataFrame({"body_center_x": [12.0], "body_center_y": [8.0]})

    contact = pair_mask_contact(Store(), df_a, df_b, max_edge_gap_px=3)

    assert contact.tolist() == [True]


def test_mask_contact_candidate_frames_skip_unneeded_decodes() -> None:
    from dlc_processor.core.mask_social import pair_mask_contact

    class Store:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def masks_for_frame(self, frame):
            self.calls.append(int(frame))
            mask_a = np.zeros((20, 20), dtype=bool)
            mask_b = np.zeros((20, 20), dtype=bool)
            mask_a[5:12, 3:8] = True
            mask_b[5:12, 10:15] = True
            return [
                (1, mask_a, (3, 5, 5, 7), 1.0),
                (2, mask_b, (10, 5, 5, 7), 1.0),
            ]

    store = Store()
    df_a = pd.DataFrame({"body_center_x": [5.0, 5.0], "body_center_y": [8.0, 8.0]})
    df_b = pd.DataFrame({"body_center_x": [12.0, 12.0], "body_center_y": [8.0, 8.0]})

    contact = pair_mask_contact(
        store,
        df_a,
        df_b,
        max_edge_gap_px=3,
        candidate_frames=np.array([False, True]),
    )

    assert contact.tolist() == [False, True]
    assert store.calls == [1]


def test_mask_contact_prunes_far_bboxes_before_decoding() -> None:
    from dataclasses import dataclass

    from dlc_processor.core.mask_social import pair_mask_contact

    @dataclass
    class Ann:
        bbox: tuple[float, float, float, float]
        score: float = 1.0
        segmentation: object = None
        size: tuple[int, int] = (100, 100)

    class Store:
        def __init__(self) -> None:
            self.frames = {0: [Ann((0, 0, 5, 5)), Ann((80, 80, 5, 5))]}
            self.decode_calls = 0

        def masks_for_frame(self, _frame):
            raise AssertionError("fast path should not decode whole frames")

        def mask_for_annotation(self, _frame, _idx):
            self.decode_calls += 1
            return np.ones((100, 100), dtype=bool)

    store = Store()
    df_a = pd.DataFrame({"body_center_x": [2.0], "body_center_y": [2.0]})
    df_b = pd.DataFrame({"body_center_x": [82.0], "body_center_y": [82.0]})

    contact = pair_mask_contact(store, df_a, df_b, max_edge_gap_px=2)

    assert contact.tolist() == [False]
    assert store.decode_calls == 0


def test_fast_mask_contact_uses_bbox_without_decoding() -> None:
    from dataclasses import dataclass

    from dlc_processor.core.mask_social import pair_mask_contact

    @dataclass
    class Ann:
        bbox: tuple[float, float, float, float]
        score: float = 1.0
        segmentation: object = None
        size: tuple[int, int] = (100, 100)

    class Store:
        def __init__(self) -> None:
            self.frames = {0: [Ann((0, 0, 10, 10)), Ann((9, 0, 10, 10))]}
            self.decode_calls = 0

        def mask_for_annotation(self, _frame, _idx):
            self.decode_calls += 1
            return np.ones((100, 100), dtype=bool)

    store = Store()
    df_a = pd.DataFrame({"body_center_x": [5.0], "body_center_y": [5.0]})
    df_b = pd.DataFrame({"body_center_x": [14.0], "body_center_y": [5.0]})

    contact = pair_mask_contact(store, df_a, df_b, max_edge_gap_px=0, exact_masks=False)

    assert contact.tolist() == [True]
    assert store.decode_calls == 0


def test_mask_contact_supports_close_same_direction_anogenital() -> None:
    n = 6
    df_b = pd.DataFrame({
        "nose_x": np.full(n, 130.0),
        "nose_y": np.zeros(n),
        "tailbase_x": np.full(n, 100.0),
        "tailbase_y": np.zeros(n),
        "neck_x": np.full(n, 120.0),
        "neck_y": np.zeros(n),
        "left_hip_x": np.full(n, 105.0),
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": np.full(n, 105.0),
        "right_hip_y": np.full(n, 3.0),
    })
    df_a = pd.DataFrame({
        "nose_x": np.full(n, 91.0),  # just outside close_tol, still touching by mask
        "nose_y": np.zeros(n),
        "tailbase_x": np.full(n, 66.0),
        "tailbase_y": np.zeros(n),
        "neck_x": np.full(n, 80.0),
        "neck_y": np.zeros(n),
        "left_hip_x": np.full(n, 71.0),
        "left_hip_y": np.full(n, -3.0),
        "right_hip_x": np.full(n, 71.0),
        "right_hip_y": np.full(n, 3.0),
    })

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=8.0,
        follow_tol=8.0,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert arrays["mask_contact"].all()
    assert arrays["a_nose2anogenital_b"].any()


def test_mask_contact_expands_nose_tail_proximity_to_anogenital() -> None:
    n = 5
    df_b = _social_df_xy(
        nose=(40.0, 0.0),
        rear=(0.0, 0.0),
        neck=(20.0, 0.0),
        left_hip=(8.0, -3.0),
        right_hip=(8.0, 3.0),
        tail_name="tailbase",
        n=n,
    )
    df_a = _social_df_xy(
        nose=(-8.0, 0.0),
        rear=(-32.0, 0.0),
        neck=(-18.0, 0.0),
        left_hip=(-27.0, -3.0),
        right_hip=(-27.0, 3.0),
        tail_name="tailbase",
        n=n,
    )

    without_masks = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        follow_tol=3.0,
        median_filter=1,
    ).compute_all()
    with_masks = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        follow_tol=3.0,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert not without_masks["a_nose2anogenital_b"].any()
    assert with_masks["a_nose2anogenital_b"].all()


def test_mask_contact_flank_contact_stays_body_not_anogenital() -> None:
    n = 5
    df_b = _social_df_xy(
        nose=(40.0, 0.0),
        rear=(0.0, 0.0),
        neck=(20.0, 0.0),
        left_hip=(8.0, -3.0),
        right_hip=(8.0, 3.0),
        tail_name="tailbase",
        n=n,
    )
    df_a = _social_df_xy(
        nose=(8.0, 5.0),
        rear=(-18.0, 5.0),
        neck=(-5.0, 5.0),
        left_hip=(-13.0, 2.0),
        right_hip=(-13.0, 8.0),
        tail_name="tailbase",
        n=n,
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        follow_tol=3.0,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert not arrays["a_nose2anogenital_b"].any()
    assert arrays["a_nose2body_b"].all()


def test_mask_contact_expands_nose_body_proximity() -> None:
    n = 5
    df_b = _social_df_xy(
        nose=(40.0, 0.0),
        rear=(0.0, 0.0),
        neck=(20.0, 0.0),
        left_hip=(10.0, -3.0),
        right_hip=(10.0, 3.0),
        tail_name="tailbase",
        n=n,
    )
    df_a = _social_df_xy(
        nose=(20.0, 7.0),
        rear=(-4.0, 7.0),
        neck=(8.0, 7.0),
        left_hip=(0.0, 4.0),
        right_hip=(0.0, 10.0),
        tail_name="tailbase",
        n=n,
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        follow_tol=3.0,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert arrays["a_nose2body_b"].all()


def test_mask_contact_expands_nose_to_nose_proximity() -> None:
    n = 5
    df_a = _social_df_xy(
        nose=(0.0, 0.0),
        rear=(-25.0, 0.0),
        neck=(-10.0, 0.0),
        left_hip=(-18.0, -3.0),
        right_hip=(-18.0, 3.0),
        tail_name="tailbase",
        n=n,
    )
    df_b = _social_df_xy(
        nose=(7.0, 0.0),
        rear=(32.0, 0.0),
        neck=(17.0, 0.0),
        left_hip=(25.0, -3.0),
        right_hip=(25.0, 3.0),
        tail_name="tailbase",
        n=n,
    )

    arrays = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert arrays["nose2nose"].all()


def test_mask_contact_rescues_nose_to_nose_with_noisy_heading() -> None:
    n = 5
    df_a = _social_df_xy(
        nose=(0.0, 0.0),
        rear=(-25.0, 0.0),
        neck=(-10.0, 0.0),
        left_hip=(-18.0, -3.0),
        right_hip=(-18.0, 3.0),
        tail_name="tailbase",
        n=n,
    )
    # B's tail is on the wrong side for a clean opposite-heading test, but
    # the nose is still much closer to A's nose than either rear anchor.
    df_b = _social_df_xy(
        nose=(6.0, 0.0),
        rear=(-19.0, 0.0),
        neck=(-4.0, 0.0),
        left_hip=(-12.0, -3.0),
        right_hip=(-12.0, 3.0),
        tail_name="tailbase",
        n=n,
    )

    without_masks = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        median_filter=1,
    ).compute_all()
    with_masks = SocialBehaviors(
        df_a=df_a,
        df_b=df_b,
        close_tol=3.0,
        median_filter=1,
        mask_contact=np.ones(n, dtype=bool),
    ).compute_all()

    assert not without_masks["nose2nose"].any()
    assert with_masks["nose2nose"].all()


def test_compute_kinematics_appends_calibrated_cm_columns() -> None:
    df = _mouse_track_df()

    result = compute_kinematics(
        df,
        fps=5.0,
        per_bodypart=False,
        acceleration=False,
        body_elongation=False,
        curvature=False,
        head_direction=False,
        mobility_state=False,
        path_tortuosity=False,
        rearing=False,
        px_per_cm=10.0,
    )

    assert "body_speed_px_s" in result
    assert "body_speed_cm_s" in result
    assert "distance_traveled_px" in result
    assert "distance_traveled_cm" in result

    np.testing.assert_allclose(
        result["body_speed_cm_s"].to_numpy(),
        result["body_speed_px_s"].to_numpy() / 10.0,
    )


def test_compute_kinematics_uses_loaded_frame_times_for_speed() -> None:
    df = _mouse_track_df()

    result = compute_kinematics(
        df,
        fps=30.0,
        time_s=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        per_bodypart=False,
        acceleration=False,
        body_elongation=False,
        curvature=False,
        head_direction=False,
        mobility_state=False,
        path_tortuosity=False,
        rearing=False,
        px_per_cm=10.0,
    )

    np.testing.assert_allclose(result["body_speed_px_s"].to_numpy()[1:], [20.0, 20.0, 20.0, 20.0])
    np.testing.assert_allclose(
        result["distance_traveled_cm"].to_numpy(),
        result["distance_traveled_px"].to_numpy() / 10.0,
    )
    assert np.isclose(result.loc[result.index[-1], "distance_traveled_cm"], 4.0)


def test_compute_kinematics_calibration_skips_string_dtype_columns() -> None:
    df = _mouse_track_df()

    result = compute_kinematics(
        df,
        fps=5.0,
        per_bodypart=False,
        rearing=False,
        px_per_cm=10.0,
    )

    assert "mobility_state" in result
    assert result["mobility_state"].dtype.name in {"object", "string", "str"}
    assert "is_mobile" in result
    assert np.array_equal(result["is_mobile"].to_numpy(dtype=bool), ~result["is_immobile"].to_numpy(dtype=bool))
    assert "body_speed_cm_s" in result
    assert "distance_traveled_cm" in result


def test_mobility_state_promotes_to_behavior_arrays() -> None:
    from dlc_processor.tab_widget import _merge_mobility_behavior_arrays

    df = pd.DataFrame({"mobility_state": ["mobile", "immobile", "immobile", "mobile"]})
    arrays = _merge_mobility_behavior_arrays({"mouse1": df}, {"social_dummy": np.array([True, False, False, True])})

    assert "social_dummy" in arrays
    assert arrays["mouse1__mobile"].tolist() == [True, False, False, True]
    assert arrays["mouse1__immobile"].tolist() == [False, True, True, False]


def test_save_cleaned_h5_falls_back_to_csv_without_pytables(tmp_path, monkeypatch) -> None:
    from dlc_processor.core.data_cleaner import save_cleaned_h5
    from dlc_processor.core.dlc_loader import load_dlc_file

    def _raise_missing_tables(self, *args, **kwargs):
        raise ImportError("Missing optional dependency 'pytables'.")

    monkeypatch.setattr(pd.DataFrame, "to_hdf", _raise_missing_tables)
    out_path = tmp_path / "edited.h5"
    written = save_cleaned_h5({"mouse1": _mouse_track_df()}, str(out_path))

    assert written.endswith(".csv")
    assert (tmp_path / "edited.csv").exists()
    loaded = load_dlc_file(written)
    assert "mouse1" in loaded
    assert len(loaded["mouse1"]) == 5
