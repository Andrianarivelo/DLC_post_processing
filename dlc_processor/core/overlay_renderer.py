"""Premium overlay rendering primitives for behaviour videos.

This module is pure drawing: given keypoints, skeleton edges, per-animal colours
and pre-formatted behaviour badges, it paints a polished overlay onto an OpenCV
BGR frame. It has no knowledge of DataFrames, behaviour detection, or Qt, which
keeps it easy to test and reuse from both the live preview and the export worker.

The visual language mirrors high-quality social-interaction overlays:

- Each animal gets a distinct colour (teal, amber, ...). Skeletons are drawn
  with a dark outline beneath the coloured stroke so they read on any
  background, and keypoints are dark-ringed dots.
- Active behaviours are shown as rounded, semi-transparent "badges" anchored to
  each animal, e.g. ``M1: nose-to-nose 0.59`` in that animal's colour with a
  soft drop shadow, exactly like a broadcast-style annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Per-animal colours in BGR. Index 0 = teal, 1 = amber: these match the
# secondary accents in shared/ui_kit so the app and its exports feel unified.
ANIMAL_COLORS_BGR: list[tuple[int, int, int]] = [
    (201, 208, 63),    # teal      (#3fd0c9)
    (112, 176, 255),   # amber     (#ffb070)
    (122, 212, 95),    # green     (#5fd47a)
    (140, 111, 255),   # rose      (#ff6f8c)
    (255, 170, 120),   # periwinkle
    (90, 200, 250),    # gold
    (240, 150, 200),   # orchid
    (200, 200, 120),   # seafoam
]

_FONT = cv2.FONT_HERSHEY_DUPLEX
_TEXT_WHITE = (255, 255, 255)
_OUTLINE_DARK = (18, 16, 24)


def animal_color(index: int) -> tuple[int, int, int]:
    """Return the BGR colour for animal *index* (wraps around the palette)."""
    return ANIMAL_COLORS_BGR[max(0, int(index)) % len(ANIMAL_COLORS_BGR)]


@dataclass
class OverlayStyle:
    """Tunable appearance for :func:`render_overlay`."""

    draw_skeleton: bool = True
    draw_keypoints: bool = True
    draw_labels: bool = True          # small animal-name tag near the body
    draw_behaviors: bool = True       # behaviour badges
    fill_body: bool = False           # translucent convex-hull body fill
    outline: bool = True              # dark outline under strokes / dots
    line_thickness: int = 2
    kp_radius: int = 4
    badge_mode: str = "per_animal"    # "per_animal" | "banner"
    max_badges_per_animal: int = 4
    font_scale: float = 0.0           # 0 -> auto from frame width
    badge_alpha: float = 0.86


def auto_font_scale(frame_w: int) -> float:
    """Pick a readable font scale proportional to the frame width."""
    return float(np.clip(frame_w / 1000.0, 0.5, 1.05))


# ── Low-level shapes ──────────────────────────────────────────────────────────

def draw_filled_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Paint a filled rounded rectangle directly onto *img* (full opacity)."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    radius = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if radius <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        return
    # Fill the cross-shaped body of the rectangle with two overlapping strips.
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    # Fill the four corner gaps with quarter-ellipses. Each tuple is
    # (centre_x, centre_y, start_angle_deg) where cv2.ellipse draws a 90-deg
    # arc: top-left starts at 180 deg, top-right at 270, bottom-left at 90,
    # bottom-right at 0 (all in OpenCV's clockwise convention).
    for cx, cy, ang in (
        (x1 + radius, y1 + radius, 180),
        (x2 - radius, y1 + radius, 270),
        (x1 + radius, y2 - radius, 90),
        (x2 - radius, y2 - radius, 0),
    ):
        cv2.ellipse(img, (cx, cy), (radius, radius), ang, 0, 90, color, -1, cv2.LINE_AA)


def _outlined_text(
    img: np.ndarray, text: str, org: tuple[int, int],
    font_scale: float, color: tuple[int, int, int], thickness: int,
) -> None:
    """Draw *text* with a dark outline pass followed by the coloured fill pass.

    The two-pass approach (thicker dark stroke first, thinner coloured stroke
    on top) makes labels legible on any background without requiring a
    separate background rectangle.
    """
    cv2.putText(img, text, org, _FONT, font_scale, _OUTLINE_DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_badge(
    img: np.ndarray,
    x: int, y: int,
    text: str,
    color_bgr: tuple[int, int, int],
    font_scale: float,
    *,
    alpha: float = 0.86,
) -> tuple[int, int]:
    """Draw a rounded, shadowed behaviour badge with top-left corner at (x, y).

    Returns the (width, height) of the badge so callers can stack badges.
    """
    thickness = max(1, int(round(font_scale * 1.6)))
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad_x = int(round(th * 0.85))
    pad_y = int(round(th * 0.55))
    w = tw + 2 * pad_x
    h = th + baseline + 2 * pad_y
    x2, y2 = x + w, y + h
    radius = int(h * 0.42)

    overlay = img.copy()
    draw_filled_rounded_rect(overlay, x + 2, y + 4, x2 + 2, y2 + 4, (12, 10, 16), radius)  # shadow
    draw_filled_rounded_rect(overlay, x, y, x2, y2, color_bgr, radius)                      # pill
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    cv2.putText(
        img, text, (x + pad_x, y + pad_y + th),
        _FONT, font_scale, _TEXT_WHITE, thickness, cv2.LINE_AA,
    )
    return w, h


# ── Frame composition ─────────────────────────────────────────────────────────

@dataclass
class AnimalDraw:
    """Everything needed to draw one animal for one frame."""

    keypoints: dict[str, tuple[int, int]]      # bodypart -> (x, y) in pixels
    color: tuple[int, int, int]
    label: str = ""                            # small name tag near the body
    badges: list[str] = field(default_factory=list)  # pre-formatted badge texts


def render_overlay(
    frame: np.ndarray,
    animals: list[AnimalDraw],
    edges: list[tuple[str, str]],
    style: OverlayStyle,
) -> np.ndarray:
    """Render skeletons, keypoints, name tags and behaviour badges onto *frame*.

    *frame* is modified in place and also returned.
    """
    fh, fw = frame.shape[:2]
    fscale = style.font_scale or auto_font_scale(fw)
    lt = max(1, int(style.line_thickness))
    kr = max(2, int(style.kp_radius))

    # ── Skeleton + keypoints, per animal ──────────────────────────────────
    for a in animals:
        kp = a.keypoints
        if not kp:
            continue

        if style.fill_body and len(kp) >= 3:
            pts = np.array(list(kp.values()), dtype=np.int32)
            hull = cv2.convexHull(pts)
            tint = tuple(min(255, int(c) + 60) for c in a.color)
            overlay = frame.copy()
            cv2.fillConvexPoly(overlay, hull, tint, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        if style.draw_skeleton and edges:
            if style.outline:
                for b1, b2 in edges:
                    if b1 in kp and b2 in kp:
                        cv2.line(frame, kp[b1], kp[b2], _OUTLINE_DARK, lt + 3, cv2.LINE_AA)
            for b1, b2 in edges:
                if b1 in kp and b2 in kp:
                    cv2.line(frame, kp[b1], kp[b2], a.color, lt, cv2.LINE_AA)

        if style.draw_keypoints:
            for (ix, iy) in kp.values():
                if style.outline:
                    cv2.circle(frame, (ix, iy), kr + 2, _OUTLINE_DARK, -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), kr, a.color, -1, cv2.LINE_AA)

    # ── Small name tag near each body centroid ────────────────────────────
    if style.draw_labels:
        for a in animals:
            if not a.keypoints or not a.label:
                continue
            cx = int(np.mean([p[0] for p in a.keypoints.values()]))
            cy = int(np.mean([p[1] for p in a.keypoints.values()]))
            _outlined_text(
                frame, a.label, (cx - 16, cy - 14),
                fscale * 0.7, a.color, max(1, int(round(fscale))),
            )

    # ── Behaviour badges ──────────────────────────────────────────────────
    if style.draw_behaviors:
        if style.badge_mode == "banner":
            _render_badge_banner(frame, animals, style, fscale)
        else:
            _render_badges_per_animal(frame, animals, style, fscale)

    return frame


def _render_badges_per_animal(
    frame: np.ndarray, animals: list[AnimalDraw], style: OverlayStyle, fscale: float,
) -> None:
    """Draw each animal's badge stack anchored beneath it.

    When two animals are close (e.g. during nose-to-nose) their stacks would
    overlap, so overlapping blocks are pushed down and left-aligned into one
    tidy column, matching the stacked broadcast look.
    """
    fh, fw = frame.shape[:2]
    gap = 6

    blocks = []
    for a in animals:
        if not a.badges or not a.keypoints:
            continue
        texts = a.badges[: style.max_badges_per_animal]
        sizes = [_measure_badge(t, fscale) for t in texts]
        w = max(s[0] for s in sizes)
        h = sum(s[1] for s in sizes) + gap * (len(texts) - 1)
        cx = int(np.mean([p[0] for p in a.keypoints.values()]))
        anchor_y = int(max(p[1] for p in a.keypoints.values())) + 10
        x = int(np.clip(cx - w // 2, 6, max(6, fw - w - 6)))
        blocks.append({"texts": texts, "sizes": sizes, "color": a.color,
                       "w": w, "h": h, "x": x, "y": anchor_y})

    # Greedy top-to-bottom placement that resolves overlaps.
    blocks.sort(key=lambda b: b["y"])
    placed: list[dict] = []
    for b in blocks:
        changed = True
        while changed:
            changed = False
            for p in placed:
                overlap = not (
                    b["x"] + b["w"] < p["x"] or b["x"] > p["x"] + p["w"]
                    or b["y"] + b["h"] < p["y"] or b["y"] > p["y"] + p["h"]
                )
                if overlap:
                    b["y"] = p["y"] + p["h"] + gap
                    b["x"] = p["x"]
                    changed = True
        b["y"] = int(np.clip(b["y"], 6, max(6, fh - b["h"] - 6)))
        b["x"] = int(np.clip(b["x"], 6, max(6, fw - b["w"] - 6)))
        placed.append(b)

    for b in placed:
        y = b["y"]
        for text, (_w, sh) in zip(b["texts"], b["sizes"]):
            draw_badge(frame, b["x"], y, text, b["color"], fscale, alpha=style.badge_alpha)
            y += sh + gap


def _render_badge_banner(
    frame: np.ndarray, animals: list[AnimalDraw], style: OverlayStyle, fscale: float,
) -> None:
    """Draw all active badges as a centre-bottom stacked banner.

    Collects every badge from every animal in order, then draws them from
    bottom to top (iterating ``reversed``) so the first animal's badge ends
    up closest to the bottom edge and subsequent badges stack upward. This
    preserves reading order (earliest animal at the bottom) without requiring
    a separate sort pass.
    """
    fh, fw = frame.shape[:2]
    entries: list[tuple[str, tuple[int, int, int]]] = []
    for a in animals:
        for text in a.badges[: style.max_badges_per_animal]:
            entries.append((text, a.color))
    y_bottom = fh - 12
    for text, color in reversed(entries):
        w, h = _measure_badge(text, fscale)
        x = fw // 2 - w // 2
        draw_badge(frame, x, y_bottom - h, text, color, fscale, alpha=style.badge_alpha)
        y_bottom -= h + 6


def _measure_badge(text: str, font_scale: float) -> tuple[int, int]:
    """Return the (width, height) a badge drawn with *text* would occupy.

    Mirrors the padding arithmetic in :func:`draw_badge` without actually
    touching any image. Used by the placement engine to pre-compute block
    sizes before committing to a layout position.
    """
    thickness = max(1, int(round(font_scale * 1.6)))
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad_x = int(round(th * 0.85))
    pad_y = int(round(th * 0.55))
    return tw + 2 * pad_x, th + baseline + 2 * pad_y
