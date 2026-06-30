"""Tests for the premium behaviour-overlay renderer and its data attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dlc_processor.core import overlay_renderer as orr


def test_animal_color_wraps_palette():
    assert orr.animal_color(0) == orr.ANIMAL_COLORS_BGR[0]
    assert orr.animal_color(len(orr.ANIMAL_COLORS_BGR)) == orr.ANIMAL_COLORS_BGR[0]


def test_draw_filled_rounded_rect_paints_pixels():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    orr.draw_filled_rounded_rect(img, 5, 5, 55, 35, (0, 0, 255), radius=8)
    # The centre of the rect must be painted with the requested colour.
    assert tuple(int(v) for v in img[20, 30]) == (0, 0, 255)


def test_draw_badge_returns_positive_size_and_marks_frame():
    img = np.full((80, 240, 3), 40, dtype=np.uint8)
    w, h = orr.draw_badge(img, 10, 10, "M1: nose-to-nose 0.59", (201, 208, 63), 0.7)
    assert w > 0 and h > 0
    assert img.sum() > 80 * 240 * 3 * 40  # something brighter was drawn


def test_render_overlay_draws_skeleton_and_badges():
    frame = np.full((120, 160, 3), 50, dtype=np.uint8)
    before = frame.copy()
    kp = {"nose": (40, 30), "neck": (40, 50), "tail": (40, 90)}
    animals = [
        orr.AnimalDraw(keypoints=kp, color=(201, 208, 63), label="M1",
                       badges=["M1: nose-to-nose 0.59"]),
    ]
    edges = [("nose", "neck"), ("neck", "tail")]
    out = orr.render_overlay(frame, animals, edges, orr.OverlayStyle())
    assert out is frame                      # rendered in place
    assert not np.array_equal(out, before)   # the frame changed


# ── Behaviour -> animal attribution (OverlayWorker) ───────────────────────────

def _worker(dfs, behavior_arrays):
    """Build an OverlayWorker without running its QThread __init__."""
    from dlc_processor.workers.overlay_worker import OverlayWorker
    w = OverlayWorker.__new__(OverlayWorker)
    w.animal_dfs = dfs
    w.behavior_arrays = behavior_arrays
    return w


def _animal_df(n=5, like=0.59):
    return pd.DataFrame({
        "nose_x": np.full(n, 10.0), "nose_y": np.full(n, 10.0),
        "nose_likelihood": np.full(n, like),
        "tail_x": np.full(n, 10.0), "tail_y": np.full(n, 30.0),
        "tail_likelihood": np.full(n, like),
    })


def test_mutual_behavior_maps_to_both_animals():
    dfs = {"resident": _animal_df(), "intruder": _animal_df()}
    w = _worker(dfs, {"nose2nose": np.ones(5, dtype=bool)})
    badges = w._behavior_badges(["resident", "intruder"], 2, show_scores=True)
    assert set(badges) == {0, 1}
    assert badges[0][0].startswith("M1: nose-to-nose")
    assert badges[1][0].startswith("M2: nose-to-nose")
    assert badges[0][0].endswith("0.59")


def test_directional_and_per_animal_attribution():
    dfs = {"resident": _animal_df(), "intruder": _animal_df()}
    animals = ["resident", "intruder"]
    w = _worker(dfs, {})
    assert w._animals_for_behavior("a_following_b", animals) == [0]
    assert w._animals_for_behavior("b_following_a", animals) == [1]
    assert w._animals_for_behavior("resident__immobile", animals) == [0]
    assert w._animals_for_behavior("intruder__mobile", animals) == [1]
    assert w._animals_for_behavior("nose2nose", animals) == [0, 1]


def test_animal_likelihood_is_mean_clamped():
    dfs = {"resident": _animal_df(like=0.8)}
    w = _worker(dfs, {})
    assert w._animal_likelihood("resident", 0) == pytest.approx(0.8)
    assert w._animal_likelihood("resident", 999) is None


def test_scores_can_be_disabled():
    dfs = {"resident": _animal_df(), "intruder": _animal_df()}
    w = _worker(dfs, {"nose2nose": np.ones(5, dtype=bool)})
    badges = w._behavior_badges(["resident", "intruder"], 1, show_scores=False)
    assert badges[0][0] == "M1: nose-to-nose"
