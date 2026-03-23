from __future__ import annotations

import numpy as np
import pandas as pd

from dlc_processor.core.data_cleaner import fix_impossible_all_animals
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
