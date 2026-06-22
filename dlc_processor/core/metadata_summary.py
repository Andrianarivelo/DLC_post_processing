"""Metadata-aware batch summaries and figures for DLC Processor."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd


_POSITION_BINS = 80
_EXPORT_DPI = 600
_EGOCENTRIC_RADIUS_BODY_LENGTHS = 3.0
_CONTACT_RADIUS_BODY_LENGTHS = 1.25
_FRONTAL_HALF_ANGLE_DEG = 45.0
_CENTER_ALIASES = ("body_center", "body_centre", "center", "centre", "centroid")
_NOSE_ALIASES = ("nose", "snout")
_TAIL_ALIASES = ("tailbase", "tail_base", "tail")
_TRANSITION_PRIORITY = (
    "nose2nose",
    "a_nose2anogenital_b",
    "b_nose2anogenital_a",
    "a_nose2body_b",
    "b_nose2body_a",
    "a_chasing_b",
    "b_chasing_a",
    "a_escapes_b",
    "b_escapes_a",
    "a_approaches_b",
    "b_approaches_a",
    "a_withdraws_from_b",
    "b_withdraws_from_a",
    "a_following_b",
    "b_following_a",
    "sidebyside",
    "sidereside",
    "fighting",
    "attacks",
    "a_oriented_toward_b",
    "b_oriented_toward_a",
    "a_withdrawal_after_contact_b",
    "b_withdrawal_after_contact_a",
)
_MOBILITY_STATE_METRICS = {
    "mobile",
    "immobile",
    "is_mobile",
    "is_immobile",
    "freezing",
    "mobility_state",
}
_MASK_CONTACT_METRICS = {"mask_contact"}
_REDUNDANT_SOCIAL_SUMMARY_METRICS = {
    "passive_anogenital",
    "passive_investigation",
    "passive_being_followed",
    "passive_being_chased",
    "passive_withdrawal",
}
_SOCIAL_SUMMARY_REQUIRED_METRICS = ("fighting",)
_CUMULATIVE_OVERVIEW_TOKENS = (
    "distance_traveled",
    "distance_travelled",
    "cumulative_distance",
    "total_distance",
    "path_length",
    "track_length",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MOUSE_TOP_SVG_CANDIDATES = (
    _PROJECT_ROOT / "mousetracker" / "resources" / "mouse_top.svg",
    Path(__file__).resolve().parents[1] / "assets" / "mouse_top.svg",
)
_MOUSE_TOP_IMAGE_CACHE: dict[tuple[int, int, str], np.ndarray] = {}


def export_group_summaries(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    group_by: list[str] | tuple[str, ...] = ("condition",),
    animal_filter: str = "",
    mouse_id_filter: str = "",
    plot_style: str = "box_strip",
    tab10_color_map: str | dict[str, int] | None = None,
    comparison_method: str = "holm_ttest",
) -> dict[str, Any]:
    """Write grouped metric/social summaries and overview figures.

    Parameters
    ----------
    summary_df:
        Per-recording summary returned by ``batch_export_recordings``.
    output_dir:
        Destination folder for aggregate tables and PNG figures.
    group_by:
        Metadata columns used as comparison groups, e.g. ``condition`` or
        ``condition, genotype``.
    animal_filter:
        Optional focal local tracking label, e.g. ``mouse1``. This is not the
        biological mouse ID.
    mouse_id_filter:
        Optional focal biological mouse ID, e.g. ``31098``. This is matched
        against the metadata ``mouseId`` column and the partner mouse ID column.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df is None or summary_df.empty:
        return {"output_dir": str(out_dir), "tables": [], "figures": [], "message": "No summary rows."}

    df = summary_df.copy()
    df = _filter_by_animal(df, animal_filter)
    df = _filter_by_mouse_id(df, mouse_id_filter)
    if df.empty:
        focus = _filter_description(animal_filter, mouse_id_filter)
        return {
            "output_dir": str(out_dir),
            "tables": [],
            "figures": [],
            "message": f"No summary rows for {focus}.",
            "animal_filter": str(animal_filter or ""),
            "mouse_id_filter": str(mouse_id_filter or ""),
        }
    group_cols = [str(col) for col in group_by if str(col) and str(col) in df.columns]
    if not group_cols:
        group_cols = ["recording"] if "recording" in df.columns else []
    df["_group"] = _combine_group_columns(df, group_cols)

    metrics_table = _aggregate_values(df, value_col="mean", behavior=False)
    social_source = _social_behavior_summary_source(df)
    mask_contact_source = _mask_contact_summary_source(df)
    frequency_value_col = _behavior_frequency_value_col(df)
    social_tables = []
    mask_contact_tables = []
    for value_col in (frequency_value_col, "mean_bout_duration_s", "cumulative_duration_s"):
        if value_col in df.columns:
            social_tables.append(_aggregate_values(social_source, value_col=value_col, behavior=True))
            mask_contact_tables.append(_aggregate_values(mask_contact_source, value_col=value_col, behavior=True))

    tables: list[str] = []
    if not metrics_table.empty:
        path = out_dir / "batch_metrics_by_group.csv"
        metrics_table.to_csv(path, index=False)
        tables.append(str(path))
    if social_tables:
        social_table = pd.concat(social_tables, ignore_index=True)
        path = out_dir / "batch_social_behaviors_by_group.csv"
        social_table.to_csv(path, index=False)
        tables.append(str(path))
    else:
        social_table = pd.DataFrame()
    mask_contact_tables = [table for table in mask_contact_tables if not table.empty]
    if mask_contact_tables:
        mask_contact_table = pd.concat(mask_contact_tables, ignore_index=True)
        mask_contact_table["category"] = "contact_signal"
        path = out_dir / "batch_mask_contact_by_group.csv"
        mask_contact_table.to_csv(path, index=False)
        tables.append(str(path))
    else:
        mask_contact_table = pd.DataFrame()

    comparison = _comparison_table(df, method=comparison_method)
    if not comparison.empty:
        path = out_dir / "batch_group_comparisons.csv"
        comparison.to_csv(path, index=False)
        tables.append(str(path))

    figures = _write_figures(
        metrics_table,
        social_table,
        out_dir,
        df,
        mask_contact_table=mask_contact_table,
        plot_style=plot_style,
        tab10_color_map=tab10_color_map,
        comparison_table=comparison,
    )
    return {
        "output_dir": str(out_dir),
        "tables": tables,
        "figures": figures,
        "group_by": group_cols,
        "animal_filter": str(animal_filter or ""),
        "mouse_id_filter": str(mouse_id_filter or ""),
        "plot_style": _normalize_plot_style(plot_style),
        "comparison_method": str(comparison_method or "holm_ttest"),
    }


def export_group_position_maps(
    exported: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    group_by: list[str] | tuple[str, ...] = ("condition",),
    animal_filter: str = "",
    mouse_id_filter: str = "",
    bins: int = _POSITION_BINS,
) -> dict[str, Any]:
    """Export group-averaged position maps from batch framewise tables.

    When nose/tail coordinates are available, maps are egocentric: the focal
    animal is centered, its forward body axis points up, and the partner
    position is expressed in body lengths. This is the comparable social
    position map used for frontal/non-frontal contact quantification. If a
    table lacks enough body-axis information, the exporter falls back to the
    older normalized arena map.
    """
    out_dir = Path(output_dir) / "position_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not exported:
        return {"output_dir": str(out_dir), "tables": [], "figures": [], "message": "No exported tables."}

    filtered_summary = _filter_by_animal(summary_df.copy(), animal_filter) if summary_df is not None else pd.DataFrame()
    filtered_summary = _filter_by_mouse_id(filtered_summary, mouse_id_filter)
    group_cols = [str(col) for col in group_by if str(col)]
    summary_lookup = _summary_group_lookup(filtered_summary, group_cols)
    allowed_animals = _animals_by_recording(filtered_summary)
    map_rows: list[dict[str, Any]] = []

    bins = int(max(10, min(250, bins)))
    arena_edges = np.linspace(0.0, 1.0, bins + 1)
    arena_centers = (arena_edges[:-1] + arena_edges[1:]) / 2.0
    ego_edges = np.linspace(-_EGOCENTRIC_RADIUS_BODY_LENGTHS, _EGOCENTRIC_RADIUS_BODY_LENGTHS, bins + 1)
    ego_centers = (ego_edges[:-1] + ego_edges[1:]) / 2.0
    accum: dict[tuple[str, str], dict[str, Any]] = {}
    contact_metric_rows: list[dict[str, Any]] = []

    for item in exported:
        table_path = Path(str(item.get("table_path", "") or ""))
        if not table_path.exists():
            continue
        try:
            table = pd.read_csv(table_path)
        except Exception:
            continue
        recording = str(item.get("name", "") or table_path.stem.replace("_metrics_behavior", ""))
        all_animals = _position_animals(table, "")
        animals = _position_animals(table, animal_filter)
        if mouse_id_filter:
            allowed = allowed_animals.get(recording, set())
            animals = [animal for animal in animals if animal in allowed]
        for animal in animals:
            wrote_egocentric = False
            for partner in [item for item in all_animals if item != animal]:
                ego = _extract_egocentric_pair(table, animal, partner)
                if ego is None:
                    continue
                ego_x, ego_y = ego
                valid = np.isfinite(ego_x) & np.isfinite(ego_y)
                valid &= (np.hypot(ego_x, ego_y) <= _EGOCENTRIC_RADIUS_BODY_LENGTHS)
                if int(valid.sum()) < 2:
                    continue
                hist, _, _ = np.histogram2d(ego_x[valid], ego_y[valid], bins=[ego_edges, ego_edges])
                total = float(hist.sum())
                if total <= 0:
                    continue
                hist = hist / total
                group = summary_lookup.get((recording, animal)) or _group_from_table(table, group_cols) or "all"
                key = (str(group), str(animal), str(partner), "egocentric")
                slot = accum.setdefault(
                    key,
                    {
                        "sum": np.zeros_like(hist, dtype=np.float64),
                        "n_maps": 0,
                        "n_points": 0,
                        "recordings": [],
                        "edges": ego_edges,
                        "centers": ego_centers,
                    },
                )
                slot["sum"] += hist
                slot["n_maps"] += 1
                slot["n_points"] += int(valid.sum())
                slot["recordings"].append(recording)
                contact_metric_rows.append({
                    "recording": recording,
                    "group": group,
                    "animal": animal,
                    "partner": partner,
                    **_egocentric_contact_metrics(ego_x, ego_y),
                })
                wrote_egocentric = True

            if wrote_egocentric:
                continue

            coords = _extract_position_xy(table, animal)
            if coords is None:
                continue
            x, y = coords
            x_norm, y_norm = _normalize_position_xy(table, x, y)
            valid = np.isfinite(x_norm) & np.isfinite(y_norm)
            valid &= (x_norm >= 0.0) & (x_norm <= 1.0) & (y_norm >= 0.0) & (y_norm <= 1.0)
            if int(valid.sum()) < 2:
                continue
            hist, _, _ = np.histogram2d(x_norm[valid], y_norm[valid], bins=[arena_edges, arena_edges])
            total = float(hist.sum())
            if total <= 0:
                continue
            hist = hist / total
            group = summary_lookup.get((recording, animal)) or _group_from_table(table, group_cols) or "all"
            key = (str(group), str(animal), "", "arena")
            slot = accum.setdefault(
                key,
                {
                    "sum": np.zeros_like(hist, dtype=np.float64),
                    "n_maps": 0,
                    "n_points": 0,
                    "recordings": [],
                    "edges": arena_edges,
                    "centers": arena_centers,
                },
            )
            slot["sum"] += hist
            slot["n_maps"] += 1
            slot["n_points"] += int(valid.sum())
            slot["recordings"].append(recording)

    tables: list[str] = []
    figures: list[str] = []
    for (group, animal, partner, map_type), payload in sorted(accum.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        n_maps = max(int(payload["n_maps"]), 1)
        avg = payload["sum"] / n_maps
        safe_group = _safe_filename(group)
        safe_animal = _safe_filename(animal)
        safe_partner = _safe_filename(partner) if partner else "arena"
        stem = f"batch_position_map_{safe_group}_{safe_animal}_vs_{safe_partner}_{map_type}"
        csv_path = out_dir / f"{stem}.csv"
        _write_position_map_csv(
            csv_path,
            avg,
            payload["centers"],
            group=group,
            animal=animal,
            partner=partner,
            map_type=map_type,
            n_maps=n_maps,
            n_points=int(payload["n_points"]),
        )
        tables.append(str(csv_path))
        fig_path = out_dir / f"{stem}.png"
        if _write_position_map_png(
            fig_path,
            avg,
            title=f"Egocentric position: {animal} vs {partner} | {group}" if map_type == "egocentric" else f"Position map: {animal} | {group}",
            map_type=map_type,
        ):
            figures.extend(_saved_figure_paths(fig_path))
        map_rows.append({
            "group": group,
            "animal": animal,
            "partner": partner,
            "map_type": map_type,
            "n_maps": n_maps,
            "n_points": int(payload["n_points"]),
            "recordings": ";".join(payload["recordings"]),
            "csv_path": str(csv_path),
            "figure_path": str(fig_path) if fig_path.exists() else "",
        })

    figures.extend(_write_position_map_group_panels(accum, out_dir))

    if map_rows:
        index_path = out_dir / "batch_position_maps_index.csv"
        pd.DataFrame(map_rows).to_csv(index_path, index=False)
        tables.insert(0, str(index_path))
    if contact_metric_rows:
        metrics = pd.DataFrame(contact_metric_rows)
        metrics_path = out_dir / "batch_position_contact_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        tables.append(str(metrics_path))
        grouped = _aggregate_position_contact_metrics(metrics)
        if not grouped.empty:
            grouped_path = out_dir / "batch_position_contact_metrics_by_group.csv"
            grouped.to_csv(grouped_path, index=False)
            tables.append(str(grouped_path))
        fig_path = _write_position_contact_metrics_figure(metrics, out_dir)
        if fig_path:
            figures.extend(_saved_figure_paths(Path(fig_path)))

    return {
        "output_dir": str(out_dir),
        "tables": tables,
        "figures": figures,
        "n_maps": len(map_rows),
        "group_by": group_cols,
        "animal_filter": str(animal_filter or ""),
        "mouse_id_filter": str(mouse_id_filter or ""),
    }


def export_group_transitions(
    exported: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    group_by: list[str] | tuple[str, ...] = ("condition",),
    animal_filter: str = "",
    mouse_id_filter: str = "",
    plot_style: str = "box_strip",
    tab10_color_map: str | dict[str, int] | None = None,
    comparison_method: str = "holm_ttest",
) -> dict[str, Any]:
    """Export social-state transition probabilities and transition metrics.

    Framewise social behaviours are multi-label, so each frame is converted to a
    reproducible dominant state using a priority list. Frames with no active
    social boolean are labelled ``none``. Metrics are computed per recording and
    social pair, then aggregated by the selected metadata groups.
    """
    out_dir = Path(output_dir) / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not exported:
        return {"output_dir": str(out_dir), "tables": [], "figures": [], "message": "No exported tables."}

    filtered_summary = _filter_by_animal(summary_df.copy(), animal_filter) if summary_df is not None else pd.DataFrame()
    filtered_summary = _filter_by_mouse_id(filtered_summary, mouse_id_filter)
    group_cols = [str(col) for col in group_by if str(col)]
    summary_lookup = _summary_group_lookup(filtered_summary, group_cols)
    allowed_pairs = _social_pairs_by_recording(filtered_summary)

    metric_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for item in exported:
        table_path = Path(str(item.get("table_path", "") or ""))
        if not table_path.exists():
            continue
        try:
            table = pd.read_csv(table_path)
        except Exception:
            continue
        recording = str(item.get("name", "") or table_path.stem.replace("_metrics_behavior", ""))
        social_pairs = _social_behavior_columns_by_pair(table)
        if animal_filter or mouse_id_filter:
            allowed = allowed_pairs.get(recording, set())
            if allowed:
                social_pairs = {pair: cols for pair, cols in social_pairs.items() if pair in allowed}
        for pair, columns in social_pairs.items():
            if not columns:
                continue
            states = _dominant_social_state_sequence(table, columns)
            if len(states) < 2:
                continue
            metrics, probabilities = _transition_stats(states, _table_duration_min(table))
            animal = pair.split("_vs_", 1)[0] if "_vs_" in pair else ""
            group = summary_lookup.get((recording, animal)) or _group_from_table(table, group_cols) or "all"
            metric_rows.append({
                "recording": recording,
                "pair": pair,
                "animal": animal,
                "group": group,
                **metrics,
            })
            for (from_state, to_state), transition_values in probabilities.items():
                probability_rows.append({
                    "recording": recording,
                    "pair": pair,
                    "animal": animal,
                    "group": group,
                    "from_state": from_state,
                    "to_state": to_state,
                    **transition_values,
                })

    tables: list[str] = []
    figures: list[str] = []
    if not metric_rows:
        return {
            "output_dir": str(out_dir),
            "tables": tables,
            "figures": figures,
            "message": "No social transition states found.",
        }

    metrics = pd.DataFrame(metric_rows)
    metric_path = out_dir / "batch_transition_metrics.csv"
    metrics.to_csv(metric_path, index=False)
    tables.append(str(metric_path))

    transition_source = _transition_metrics_to_summary_source(metrics)
    metric_summary = _aggregate_values(transition_source, value_col="mean", behavior=False)
    if not metric_summary.empty:
        path = out_dir / "batch_transition_metrics_by_group.csv"
        metric_summary.to_csv(path, index=False)
        tables.append(str(path))

    comparison = _comparison_table(transition_source, method=comparison_method)
    if not comparison.empty:
        path = out_dir / "batch_transition_metric_comparisons.csv"
        comparison.to_csv(path, index=False)
        tables.append(str(path))

    probabilities = pd.DataFrame(probability_rows)
    if not probabilities.empty:
        probability_summary = _aggregate_transition_probabilities(probabilities)
        path = out_dir / "batch_transition_probabilities.csv"
        probability_summary.to_csv(path, index=False)
        tables.append(str(path))
        states = sorted(
            set(probability_summary["from_state"].astype(str))
            | set(probability_summary["to_state"].astype(str)),
            key=_transition_priority_index,
        )
        network_states = _top_transition_states(probability_summary, states, max_states=12)
        if network_states:
            legend_path = out_dir / "batch_transition_state_legend.csv"
            pd.DataFrame({
                "state_index": list(range(len(network_states))),
                "state": network_states,
                "label": [_metric_display_name(state) for state in network_states],
            }).to_csv(legend_path, index=False)
            tables.append(str(legend_path))
        figures.extend(_write_transition_heatmaps(probability_summary, out_dir))

    if not metric_summary.empty:
        fig_path = out_dir / "batch_transition_metrics_overview.png"
        _bar_strip_overview(
            metric_summary[metric_summary["value"] == "mean"],
            fig_path,
            title="Transition dynamics by group",
            y_label="Value",
            raw_data=transition_source,
            value_col="mean",
            plt=_matplotlib_pyplot(),
            plot_style=plot_style,
            tab10_color_map=tab10_color_map,
            comparison_table=comparison,
        )
        if fig_path.exists():
            figures.extend(_saved_figure_paths(fig_path))

    return {
        "output_dir": str(out_dir),
        "tables": tables,
        "figures": figures,
        "group_by": group_cols,
        "animal_filter": str(animal_filter or ""),
        "mouse_id_filter": str(mouse_id_filter or ""),
        "plot_style": _normalize_plot_style(plot_style),
        "comparison_method": str(comparison_method or "holm_ttest"),
    }


def _filter_by_animal(df: pd.DataFrame, animal_filter: str) -> pd.DataFrame:
    focus = str(animal_filter or "").strip()
    if not focus or focus.lower() in {"all", "*"}:
        return df
    if "animal" not in df.columns:
        return df.iloc[0:0].copy()
    animal = df["animal"].fillna("").astype(str)
    partner = df["partner"].fillna("").astype(str) if "partner" in df.columns else pd.Series("", index=df.index)
    scope = df["scope"].fillna("").astype(str) if "scope" in df.columns else pd.Series("", index=df.index)
    is_social = scope.str.startswith("social")
    metric_rows = ~is_social & animal.eq(focus)
    social_rows = is_social & (animal.eq(focus) | partner.eq(focus))
    return df[metric_rows | social_rows].copy()


def _filter_by_mouse_id(df: pd.DataFrame, mouse_id_filter: str) -> pd.DataFrame:
    focus = str(mouse_id_filter or "").strip()
    if df is None or df.empty or not focus or focus.lower() in {"all", "*"}:
        return df
    mouse_col = _first_existing_col(df, ("mouseId", "mouseid", "mouse_id", "subject_id", "subject"))
    partner_col = _first_existing_col(
        df,
        ("partner_mouseId", "partner_mouseid", "partner_mouse_id", "partner_subject_id", "partner_subject"),
    )
    if not mouse_col and not partner_col:
        return df.iloc[0:0].copy()
    scope = df["scope"].fillna("").astype(str) if "scope" in df.columns else pd.Series("", index=df.index)
    is_social = scope.str.startswith("social")
    primary = df[mouse_col].fillna("").astype(str).eq(focus) if mouse_col else pd.Series(False, index=df.index)
    partner = df[partner_col].fillna("").astype(str).eq(focus) if partner_col else pd.Series(False, index=df.index)
    return df[(~is_social & primary) | (is_social & (primary | partner))].copy()


def _filter_description(animal_filter: str, mouse_id_filter: str) -> str:
    parts = []
    if str(animal_filter or "").strip():
        parts.append(f"animal label '{animal_filter}'")
    if str(mouse_id_filter or "").strip():
        parts.append(f"mouseId '{mouse_id_filter}'")
    return " and ".join(parts) if parts else "selected filters"


def _first_existing_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        col = lookup.get(candidate.lower())
        if col:
            return col
    return ""


def _summary_group_lookup(df: pd.DataFrame, group_cols: list[str]) -> dict[tuple[str, str], str]:
    if df is None or df.empty or "recording" not in df.columns or "animal" not in df.columns:
        return {}
    available_group_cols = [col for col in group_cols if col in df.columns]
    if not available_group_cols:
        available_group_cols = ["recording"]
    working = df.copy()
    working["_group"] = _combine_group_columns(working, available_group_cols)
    lookup: dict[tuple[str, str], str] = {}
    scope = working["scope"].fillna("").astype(str) if "scope" in working.columns else pd.Series("", index=working.index)
    animal_rows = working[~scope.str.startswith("social")]
    for (_, row) in animal_rows.iterrows():
        recording = str(row.get("recording", "") or "")
        animal = str(row.get("animal", "") or "")
        group = str(row.get("_group", "") or "")
        if recording and animal and group:
            lookup.setdefault((recording, animal), group)
    return lookup


def _animals_by_recording(df: pd.DataFrame) -> dict[str, set[str]]:
    if df is None or df.empty or "recording" not in df.columns or "animal" not in df.columns:
        return {}
    scope = df["scope"].fillna("").astype(str) if "scope" in df.columns else pd.Series("", index=df.index)
    animal_rows = df[~scope.str.startswith("social")]
    out: dict[str, set[str]] = {}
    for _, row in animal_rows.iterrows():
        recording = str(row.get("recording", "") or "")
        animal = str(row.get("animal", "") or "")
        if recording and animal:
            out.setdefault(recording, set()).add(animal)
    return out


def _social_pairs_by_recording(df: pd.DataFrame) -> dict[str, set[str]]:
    if df is None or df.empty or "recording" not in df.columns:
        return {}
    scope = df["scope"].fillna("").astype(str) if "scope" in df.columns else pd.Series("", index=df.index)
    social_rows = df[scope.str.startswith("social")]
    out: dict[str, set[str]] = {}
    for _, row in social_rows.iterrows():
        recording = str(row.get("recording", "") or "")
        animal = str(row.get("animal", "") or "")
        partner = str(row.get("partner", "") or "")
        if recording and animal and partner:
            out.setdefault(recording, set()).add(f"{_safe_col(animal)}_vs_{_safe_col(partner)}")
    return out


def _social_behavior_columns_by_pair(table: pd.DataFrame) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = {}
    for col in table.columns:
        text = str(col)
        if not text.startswith("social_") or "__" not in text:
            continue
        pair, behavior = text[len("social_"):].split("__", 1)
        if _is_mask_contact_metric(behavior) or _is_mobility_state_metric(behavior):
            continue
        if _is_passive_social_metric(behavior):
            continue
        if behavior.endswith(("_px", "_cm", "_px_s", "_cm_s", "_deg")):
            continue
        values = pd.to_numeric(table[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        if not set(np.unique(values).tolist()).issubset({0.0, 1.0}):
            continue
        pairs.setdefault(pair, []).append(text)
    for pair, columns in pairs.items():
        pairs[pair] = sorted(columns, key=lambda col: _transition_priority_index(col.split("__", 1)[1]))
    return pairs


def _transition_priority_index(behavior: str) -> tuple[int, str]:
    try:
        return (_TRANSITION_PRIORITY.index(str(behavior)), str(behavior))
    except ValueError:
        return (len(_TRANSITION_PRIORITY), str(behavior))


def _dominant_social_state_sequence(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    n = len(table)
    if n == 0 or not columns:
        return np.array([], dtype=object)
    state = np.full(n, "none", dtype=object)
    claimed = np.zeros(n, dtype=bool)
    for col in columns:
        behavior = col.split("__", 1)[1]
        active = pd.to_numeric(table[col], errors="coerce").fillna(0).to_numpy(dtype=np.float64) > 0.5
        take = active & ~claimed
        state[take] = behavior
        claimed |= take
    return state


def _transition_stats(states: np.ndarray, duration_min: float) -> tuple[dict[str, float], dict[tuple[str, str], dict[str, float]]]:
    seq = np.asarray(states, dtype=object).reshape(-1)
    seq = seq[pd.notna(seq)]
    if len(seq) < 2:
        return {
            "transition_entropy_bits": 0.0,
            "switch_rate_per_min": 0.0,
            "n_transitions": 0,
            "n_switches": 0,
            "n_states": int(len(set(seq.tolist()))),
        }, {}
    from_states = seq[:-1].astype(str)
    to_states = seq[1:].astype(str)
    transitions = len(from_states)
    switches = int(np.sum(from_states != to_states))
    counts: dict[tuple[str, str], int] = {}
    row_counts: dict[str, int] = {}
    for src, dst in zip(from_states, to_states):
        key = (str(src), str(dst))
        counts[key] = counts.get(key, 0) + 1
        row_counts[str(src)] = row_counts.get(str(src), 0) + 1
    probabilities = {}
    for key, count in counts.items():
        from_total = row_counts.get(key[0], 0)
        probabilities[key] = {
            "probability": float(count / max(from_total, 1)),
            "from_state_fraction": float(from_total / max(transitions, 1)),
            "transition_mass": float(count / max(transitions, 1)),
            "transition_count": int(count),
        }
    entropy = 0.0
    for src, row_total in row_counts.items():
        if row_total <= 0:
            continue
        p = np.array([
            count / row_total
            for (from_state, _to_state), count in counts.items()
            if from_state == src
        ], dtype=np.float64)
        p = p[p > 0]
        row_entropy = float(-np.sum(p * np.log2(p))) if p.size else 0.0
        entropy += (row_total / transitions) * row_entropy
    metrics = {
        "transition_entropy_bits": float(entropy),
        "switch_rate_per_min": float(switches / max(float(duration_min), 1e-9)),
        "n_transitions": int(transitions),
        "n_switches": int(switches),
        "n_states": int(len(set(seq.astype(str).tolist()))),
    }
    return metrics, probabilities


def _table_duration_min(table: pd.DataFrame) -> float:
    if "time_s" in table.columns:
        times = pd.to_numeric(table["time_s"], errors="coerce").to_numpy(dtype=np.float64)
        finite = times[np.isfinite(times)]
        if finite.size >= 2:
            diffs = np.diff(finite)
            valid_dt = diffs[np.isfinite(diffs) & (diffs > 0)]
            dt = float(np.median(valid_dt)) if valid_dt.size else 0.0
            return max(float(finite[-1] - finite[0] + dt) / 60.0, 1e-9)
    return max(len(table) / 30.0 / 60.0, 1e-9)


def _transition_metrics_to_summary_source(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        for metric in ("transition_entropy_bits", "switch_rate_per_min"):
            rows.append({
                "recording": row.get("recording", ""),
                "scope": "transition",
                "animal": row.get("animal", ""),
                "partner": "",
                "pair": row.get("pair", ""),
                "metric": metric,
                "metric_base": metric,
                "mean": float(row.get(metric, np.nan)),
                "_group": str(row.get("group", "") or "unspecified"),
                "is_behavior": False,
            })
    return pd.DataFrame(rows)


def _aggregate_transition_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    if probabilities is None or probabilities.empty:
        return pd.DataFrame()
    value_cols = [
        col
        for col in ("probability", "from_state_fraction", "transition_mass", "transition_count")
        if col in probabilities.columns
    ]
    if not value_cols:
        return pd.DataFrame()
    grouped = probabilities.groupby(["group", "from_state", "to_state"], dropna=False)
    out = grouped[value_cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    rename: dict[str, str] = {}
    for col in value_cols:
        rename[f"{col}_mean"] = col
        rename[f"{col}_std"] = f"{col}_std"
        rename[f"{col}_count"] = f"{col}_n"
    out = out.rename(columns=rename)
    if "probability_n" in out.columns:
        out["n"] = out["probability_n"]
    else:
        first_n = f"{value_cols[0]}_n"
        out["n"] = out.get(first_n, 0)
    for col in value_cols:
        std_col = f"{col}_std"
        n_col = f"{col}_n"
        if std_col in out.columns and n_col in out.columns:
            out[f"{col}_sem"] = out[std_col] / np.sqrt(pd.to_numeric(out[n_col], errors="coerce").clip(lower=1))
    if "probability_std" in out.columns:
        out["std"] = out["probability_std"]
    if "probability_sem" in out.columns:
        out["sem"] = out["probability_sem"]
    return out


def _position_animals(table: pd.DataFrame, animal_filter: str) -> list[str]:
    focus = str(animal_filter or "").strip()
    if focus and focus.lower() not in {"all", "*"}:
        return [focus]
    animals = sorted({
        col.split("__", 1)[0]
        for col in table.columns
        if "__" in str(col) and str(col).endswith("_x")
    })
    return animals


def _extract_position_xy(table: pd.DataFrame, animal: str) -> tuple[np.ndarray, np.ndarray] | None:
    prefix = f"{animal}__"
    lower_to_bp = {
        col[len(prefix):-2].lower(): col[len(prefix):-2]
        for col in table.columns
        if str(col).startswith(prefix) and str(col).endswith("_x")
    }
    for name in _CENTER_ALIASES:
        bp = lower_to_bp.get(name.lower())
        if bp and f"{prefix}{bp}_y" in table.columns:
            return _numeric_array(table[f"{prefix}{bp}_x"]), _numeric_array(table[f"{prefix}{bp}_y"])

    nose = _first_alias(lower_to_bp, _NOSE_ALIASES)
    tail = _first_alias(lower_to_bp, _TAIL_ALIASES)
    if nose and tail and f"{prefix}{nose}_y" in table.columns and f"{prefix}{tail}_y" in table.columns:
        nose_x = _numeric_array(table[f"{prefix}{nose}_x"])
        nose_y = _numeric_array(table[f"{prefix}{nose}_y"])
        tail_x = _numeric_array(table[f"{prefix}{tail}_x"])
        tail_y = _numeric_array(table[f"{prefix}{tail}_y"])
        return (nose_x + tail_x) / 2.0, (nose_y + tail_y) / 2.0

    candidates = [bp for bp in lower_to_bp.values() if f"{prefix}{bp}_y" in table.columns]
    if candidates:
        bp = candidates[0]
        return _numeric_array(table[f"{prefix}{bp}_x"]), _numeric_array(table[f"{prefix}{bp}_y"])
    return None


def _first_alias(lower_to_bp: dict[str, str], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        bp = lower_to_bp.get(alias.lower())
        if bp:
            return bp
    return ""


def _numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _normalize_position_xy(table: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width, height = _table_video_size(table)
    if width <= 0 or height <= 0:
        finite_x = x[np.isfinite(x)]
        finite_y = y[np.isfinite(y)]
        width = float(np.nanmax(finite_x)) if finite_x.size else 0.0
        height = float(np.nanmax(finite_y)) if finite_y.size else 0.0
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return x / width, y / height


def _extract_egocentric_pair(
    table: pd.DataFrame,
    focal: str,
    partner: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    focal_pose = _extract_nose_tail_xy(table, focal)
    partner_xy = _extract_position_xy(table, partner)
    if focal_pose is None or partner_xy is None:
        return None
    nose_x, nose_y, tail_x, tail_y = focal_pose
    partner_x, partner_y = partner_xy
    n = min(len(nose_x), len(partner_x))
    if n <= 1:
        return None
    nose_x = nose_x[:n]
    nose_y = nose_y[:n]
    tail_x = tail_x[:n]
    tail_y = tail_y[:n]
    partner_x = partner_x[:n]
    partner_y = partner_y[:n]

    focal_cx = (nose_x + tail_x) / 2.0
    focal_cy = (nose_y + tail_y) / 2.0
    axis_x = nose_x - tail_x
    axis_y = nose_y - tail_y
    body_len = np.hypot(axis_x, axis_y)
    valid_body = np.isfinite(body_len) & (body_len > 1e-6)
    safe_len = np.where(valid_body, body_len, np.nan)
    axis_x = axis_x / safe_len
    axis_y = axis_y / safe_len

    dx = partner_x - focal_cx
    dy = partner_y - focal_cy
    ego_x = dx * (-axis_y) + dy * axis_x
    ego_y = dx * axis_x + dy * axis_y
    return ego_x / safe_len, ego_y / safe_len


def _extract_nose_tail_xy(table: pd.DataFrame, animal: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    prefix = f"{animal}__"
    lower_to_bp = {
        col[len(prefix):-2].lower(): col[len(prefix):-2]
        for col in table.columns
        if str(col).startswith(prefix) and str(col).endswith("_x")
    }
    nose = _first_alias(lower_to_bp, _NOSE_ALIASES)
    tail = _first_alias(lower_to_bp, _TAIL_ALIASES)
    if not nose or not tail:
        return None
    required = [f"{prefix}{nose}_x", f"{prefix}{nose}_y", f"{prefix}{tail}_x", f"{prefix}{tail}_y"]
    if any(col not in table.columns for col in required):
        return None
    return (
        _numeric_array(table[required[0]]),
        _numeric_array(table[required[1]]),
        _numeric_array(table[required[2]]),
        _numeric_array(table[required[3]]),
    )


def _egocentric_contact_metrics(ego_x: np.ndarray, ego_y: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(ego_x) & np.isfinite(ego_y)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "n_valid_frames": 0,
            "contact_probability": np.nan,
            "frontal_contact_probability": np.nan,
            "non_frontal_contact_probability": np.nan,
            "front_investigation_probability": np.nan,
            "back_investigation_probability": np.nan,
            "front_back_investigation_ratio": np.nan,
            "frontal_contact_fraction": np.nan,
            "lateral_contact_probability": np.nan,
            "rear_contact_probability": np.nan,
        }
    x = ego_x[valid]
    y = ego_y[valid]
    distance = np.hypot(x, y)
    angle = np.degrees(np.arctan2(x, y))
    contact = distance <= _CONTACT_RADIUS_BODY_LENGTHS
    frontal = contact & (np.abs(angle) <= _FRONTAL_HALF_ANGLE_DEG)
    rear = contact & (np.abs(angle) >= 135.0)
    lateral = contact & ~(frontal | rear)
    n_contact = int(contact.sum())
    return {
        "n_valid_frames": n_valid,
        "contact_probability": float(n_contact / n_valid),
        "frontal_contact_probability": float(frontal.sum() / n_valid),
        "non_frontal_contact_probability": float((contact & ~frontal).sum() / n_valid),
        "front_investigation_probability": float(frontal.sum() / n_valid),
        "back_investigation_probability": float(rear.sum() / n_valid),
        "front_back_investigation_ratio": float(frontal.sum() / rear.sum()) if rear.sum() else np.nan,
        "frontal_contact_fraction": float(frontal.sum() / n_contact) if n_contact else 0.0,
        "lateral_contact_probability": float(lateral.sum() / n_valid),
        "rear_contact_probability": float(rear.sum() / n_valid),
    }


def _aggregate_position_contact_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "contact_probability",
        "frontal_contact_probability",
        "non_frontal_contact_probability",
        "front_investigation_probability",
        "back_investigation_probability",
        "front_back_investigation_ratio",
        "frontal_contact_fraction",
        "lateral_contact_probability",
        "rear_contact_probability",
    ]
    rows: list[dict[str, Any]] = []
    group_cols = ["group", "animal", "partner"]
    for keys, sub in metrics.groupby(group_cols, dropna=False):
        group, animal, partner = keys
        row: dict[str, Any] = {
            "group": group,
            "animal": animal,
            "partner": partner,
            "n_recordings": int(sub["recording"].nunique()) if "recording" in sub else int(len(sub)),
        }
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _write_position_contact_metrics_figure(metrics: pd.DataFrame, out_dir: Path) -> str:
    plt = _matplotlib_pyplot()
    if plt is None or metrics is None or metrics.empty or "group" not in metrics.columns:
        return ""
    metric_cols = [
        ("front_investigation_probability", "Front investigation"),
        ("back_investigation_probability", "Back investigation"),
    ]
    metric_cols = [(col, label) for col, label in metric_cols if col in metrics.columns]
    if not metric_cols:
        return ""
    groups = _ordered_groups(list(metrics["group"].dropna().astype(str).unique()))
    if not groups:
        return ""
    fig, axes = plt.subplots(1, len(metric_cols), squeeze=False, figsize=(4.2 * len(metric_cols), 3.4), facecolor="white", constrained_layout=True)
    rng = np.random.default_rng(12345)
    for ax, (col, label) in zip(axes[0], metric_cols):
        ax.set_facecolor("white")
        values_by_group = []
        positions = []
        for idx, group in enumerate(groups):
            vals = pd.to_numeric(metrics.loc[metrics["group"].astype(str) == str(group), col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            values_by_group.append(vals)
            if vals.size:
                positions.append(float(idx))
        box_values = [values_by_group[idx] for idx in range(len(groups)) if values_by_group[idx].size]
        box_positions = [float(idx) for idx in range(len(groups)) if values_by_group[idx].size]
        if box_values:
            bp = ax.boxplot(
                box_values,
                positions=box_positions,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "white", "linewidth": 1.4},
                whiskerprops={"color": "#555555", "linewidth": 0.9},
                capprops={"color": "#555555", "linewidth": 0.9},
            )
            for patch_idx, patch in enumerate(bp["boxes"]):
                group_idx = int(round(box_positions[patch_idx]))
                color = _pastel_tab10_color(plt, group_idx)
                patch.set_facecolor(color)
                patch.set_alpha(0.95)
                patch.set_edgecolor("#4b5563")
                patch.set_linewidth(0.9)
        for idx, group in enumerate(groups):
            vals = values_by_group[idx]
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(
                np.full(vals.size, float(idx)) + jitter,
                vals,
                s=24,
                facecolor="white",
                edgecolor="#374151",
                linewidth=0.6,
                alpha=0.96,
                zorder=4,
            )
            mean = float(np.nanmean(vals))
            if np.isfinite(mean):
                ax.scatter([float(idx)], [mean], s=34, marker="D", facecolor="white", edgecolor="#111827", linewidth=0.65, zorder=5)
        ax.set_title(label, fontsize=10.5, color="#111111")
        ax.set_ylabel("Probability", fontsize=9.5, color="#111111")
        ax.set_xticks(np.arange(len(groups), dtype=np.float64))
        ax.set_xticklabels([_group_tick_label(group, values_by_group[idx].size) for idx, group in enumerate(groups)], fontsize=8.5)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6b7280")
        ax.spines["bottom"].set_color("#6b7280")
        ax.tick_params(axis="y", labelsize=8.5, colors="#333333")
    fig.suptitle("Egocentric investigation zones", fontsize=12, color="#111111")
    path = out_dir / "batch_position_front_back_investigation.png"
    _save_figure(fig, path, facecolor="white")
    plt.close(fig)
    return str(path)


def _table_video_size(table: pd.DataFrame) -> tuple[float, float]:
    video_path = ""
    if "video_path" in table.columns and len(table):
        video_path = str(table["video_path"].iloc[0] or "")
    if not video_path:
        return 0.0, 0.0
    path = Path(video_path)
    if not path.exists():
        return 0.0, 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        cap.release()
        return width, height
    except Exception:
        return 0.0, 0.0


def _group_from_table(table: pd.DataFrame, group_cols: list[str]) -> str:
    parts: list[str] = []
    for col in group_cols:
        candidates = [col, f"metadata_{_safe_col(col)}"]
        value = ""
        for candidate in candidates:
            if candidate in table.columns and len(table):
                value = str(table[candidate].iloc[0] or "")
                break
        parts.append(value if value else "unspecified")
    return " / ".join(parts) if parts else ""


def _write_position_map_csv(
    path: Path,
    density: np.ndarray,
    centers: np.ndarray,
    *,
    group: str,
    animal: str,
    partner: str = "",
    map_type: str = "arena",
    n_maps: int,
    n_points: int,
) -> None:
    grid_x, grid_y = np.meshgrid(centers, centers, indexing="ij")
    x_col = "ego_x_body_lengths" if map_type == "egocentric" else "x_norm_center"
    y_col = "ego_y_body_lengths" if map_type == "egocentric" else "y_norm_center"
    rows = pd.DataFrame({
        "group": group,
        "animal": animal,
        "partner": partner,
        "map_type": map_type,
        x_col: grid_x.ravel(),
        y_col: grid_y.ravel(),
        "probability": density.ravel(),
        "n_maps": int(n_maps),
        "n_points": int(n_points),
    })
    rows.to_csv(path, index=False)


def _write_position_map_png(path: Path, density: np.ndarray, *, title: str, map_type: str = "arena") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(7.0, 6.2), facecolor="white", constrained_layout=True)
    ax.set_facecolor("white")
    if map_type == "egocentric":
        radius = _EGOCENTRIC_RADIUS_BODY_LENGTHS
        display = density.astype(np.float64).copy()
        centers = np.linspace(-radius, radius, density.shape[0])
        xx, yy = np.meshgrid(centers, centers, indexing="ij")
        display[np.hypot(xx, yy) > radius] = np.nan
        display[display <= 0.0] = np.nan
        cmap = _white_magma_colormap(plt)
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        im = ax.imshow(
            display.T,
            origin="lower",
            extent=(-radius, radius, -radius, radius),
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
            vmin=0.0,
        )
        _draw_egocentric_grid(ax, radius)
        _draw_center_mouse(ax, radius)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_xlabel("Partner lateral position (body lengths)", fontsize=10.5, color="#111111", labelpad=7)
        ax.set_ylabel("Partner forward position (body lengths)", fontsize=10.5, color="#111111", labelpad=9)
    else:
        display = density.astype(np.float64).copy()
        display[display <= 0.0] = np.nan
        cmap = _white_magma_colormap(plt)
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        im = ax.imshow(
            display.T,
            origin="upper",
            extent=(0.0, 1.0, 1.0, 0.0),
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
            vmin=0.0,
        )
        ax.set_xlabel("Normalized X", fontsize=10.5, color="#111111", labelpad=7)
        ax.set_ylabel("Normalized Y", fontsize=10.5, color="#111111", labelpad=9)
    ax.set_title("\n".join(textwrap.wrap(str(title), width=72)), fontsize=11, pad=12)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)
    ax.tick_params(colors="#111111", labelsize=9.5, width=1.0, length=4.0)
    cbar = fig.colorbar(im, ax=ax, label="Mean occupancy probability", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#111111", labelsize=9)
    cbar.outline.set_edgecolor("#222222")
    cbar.outline.set_linewidth(0.8)
    _save_figure(fig, path, facecolor="white")
    plt.close(fig)
    return True


def _write_position_map_group_panels(accum: dict[tuple[str, str, str, str], dict[str, Any]], out_dir: Path) -> list[str]:
    figures: list[str] = []
    plt = _matplotlib_pyplot()
    if plt is None or not accum:
        return figures
    combos = sorted({(animal, partner, map_type) for (_group, animal, partner, map_type) in accum})
    for animal, partner, map_type in combos:
        group_payloads = {
            str(group): payload
            for (group, a, p, mt), payload in accum.items()
            if a == animal and p == partner and mt == map_type and int(payload.get("n_maps", 0) or 0) > 0
        }
        groups = _ordered_groups(list(group_payloads))
        if len(groups) < 2:
            continue
        group_a, group_b = groups[:2]
        density_a = group_payloads[group_a]["sum"] / max(int(group_payloads[group_a]["n_maps"]), 1)
        density_b = group_payloads[group_b]["sum"] / max(int(group_payloads[group_b]["n_maps"]), 1)
        diff = density_a - density_b
        vmax = max(float(np.nanmax(density_a)) if density_a.size else 0.0, float(np.nanmax(density_b)) if density_b.size else 0.0, 1e-9)
        diff_abs = max(float(np.nanmax(np.abs(diff))) if diff.size else 0.0, 1e-9)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13.2, 4.25),
            facecolor="white",
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]},
        )
        im_a = _draw_position_map_panel(
            axes[0],
            density_a,
            map_type=map_type,
            title=str(group_a),
            plt=plt,
            vmin=0.0,
            vmax=vmax,
        )
        im_b = _draw_position_map_panel(
            axes[1],
            density_b,
            map_type=map_type,
            title=str(group_b),
            plt=plt,
            vmin=0.0,
            vmax=vmax,
        )
        im_diff = _draw_position_map_panel(
            axes[2],
            diff,
            map_type=map_type,
            title=f"{group_a} minus {group_b}",
            plt=plt,
            vmin=-diff_abs,
            vmax=diff_abs,
            difference=True,
        )
        fig.suptitle(
            f"Egocentric position: {animal} vs {partner}" if map_type == "egocentric" else f"Position map: {animal}",
            fontsize=13,
            color="#111111",
        )
        if im_a is not None or im_b is not None:
            image_for_cbar = im_b if im_b is not None else im_a
            cbar = fig.colorbar(image_for_cbar, ax=axes[:2], label="Mean occupancy probability", fraction=0.035, pad=0.02)
            cbar.ax.tick_params(labelsize=8.5, colors="#111111")
        if im_diff is not None:
            cbar = fig.colorbar(im_diff, ax=axes[2], label="Probability difference", fraction=0.048, pad=0.04)
            cbar.ax.tick_params(labelsize=8.5, colors="#111111")
        safe_animal = _safe_filename(animal)
        safe_partner = _safe_filename(partner) if partner else "arena"
        path = out_dir / f"batch_position_map_{safe_animal}_vs_{safe_partner}_{map_type}_group_panels.png"
        saved = _save_figure(fig, path, facecolor="white")
        plt.close(fig)
        figures.extend(saved)
    return figures


def _draw_position_map_panel(
    ax,
    density: np.ndarray,
    *,
    map_type: str,
    title: str,
    plt,
    vmin: float,
    vmax: float,
    difference: bool = False,
):
    ax.set_facecolor("white")
    display = np.asarray(density, dtype=np.float64).copy()
    if map_type == "egocentric":
        radius = _EGOCENTRIC_RADIUS_BODY_LENGTHS
        centers = np.linspace(-radius, radius, display.shape[0])
        xx, yy = np.meshgrid(centers, centers, indexing="ij")
        display[np.hypot(xx, yy) > radius] = np.nan
        if not difference:
            display[display <= 0.0] = np.nan
        cmap = plt.get_cmap("RdBu_r").copy() if difference else _white_magma_colormap(plt)
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        im = ax.imshow(
            display.T,
            origin="lower",
            extent=(-radius, radius, -radius, radius),
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        _draw_egocentric_grid(ax, radius)
        _draw_center_mouse(ax, radius)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_xlabel("Partner lateral position (body lengths)", fontsize=9.5, color="#111111", labelpad=6)
        ax.set_ylabel("Partner forward position (body lengths)", fontsize=9.5, color="#111111", labelpad=7)
    else:
        if not difference:
            display[display <= 0.0] = np.nan
        cmap = plt.get_cmap("RdBu_r").copy() if difference else _white_magma_colormap(plt)
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        im = ax.imshow(
            display.T,
            origin="upper",
            extent=(0.0, 1.0, 1.0, 0.0),
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_xlabel("Normalized X", fontsize=9.5, color="#111111", labelpad=6)
        ax.set_ylabel("Normalized Y", fontsize=9.5, color="#111111", labelpad=7)
    ax.set_title(str(title), fontsize=10.5, color="#111111", pad=8)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.05)
    ax.tick_params(colors="#111111", labelsize=8.8, width=1.0, length=3.8)
    return im


def _white_magma_colormap(plt):
    from matplotlib.colors import ListedColormap

    base = plt.get_cmap("magma")(np.linspace(0.0, 1.0, 256))
    n_blend = 56
    blend = np.linspace(0.0, 1.0, n_blend)[:, None]
    base[:n_blend, :3] = (1.0 - blend) * np.ones((n_blend, 3)) + blend * base[:n_blend, :3]
    base[:, 3] = 1.0
    cmap = ListedColormap(base, name="white_magma")
    cmap.set_under((1.0, 1.0, 1.0, 1.0))
    return cmap


def _draw_egocentric_grid(ax, radius: float) -> None:
    import matplotlib.patches as patches

    grid_color = "#7a7f87"
    ax.add_patch(
        patches.Circle(
            (0.0, 0.0),
            radius=radius,
            facecolor="none",
            edgecolor="#6b7280",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
            zorder=4,
        )
    )
    for r in np.linspace(radius / 4.0, radius, 4):
        ax.add_patch(
            patches.Circle(
                (0.0, 0.0),
                radius=float(r),
                facecolor="none",
                edgecolor=grid_color,
                linewidth=0.55,
                linestyle=":",
                alpha=0.42,
                zorder=4,
            )
        )
        ax.text(
            float(r) * 0.72,
            float(r) * 0.72,
            f"{r:.1f}",
            color="#4b5563",
            fontsize=7,
            ha="left",
            va="bottom",
            zorder=7,
        )
    ax.axhline(0.0, color=grid_color, lw=0.6, ls=":", alpha=0.55, zorder=4)
    ax.axvline(0.0, color=grid_color, lw=0.6, ls=":", alpha=0.55, zorder=4)


def _draw_center_mouse(ax, radius: float) -> None:
    rgba = _render_mouse_top_svg(render_h=520)
    if rgba is None:
        _draw_fallback_center_mouse(ax)
        return
    aspect = rgba.shape[1] / max(float(rgba.shape[0]), 1.0)
    height = min(float(radius) * 0.56, 1.55)
    width = height * aspect
    ax.imshow(
        rgba,
        origin="upper",
        extent=(-width / 2.0, width / 2.0, -height / 2.0, height / 2.0),
        interpolation="lanczos",
        zorder=8,
    )


def _draw_fallback_center_mouse(ax) -> None:
    from matplotlib.patches import Circle, Ellipse

    ax.add_patch(Ellipse((0, 0), width=0.42, height=0.95, angle=0, facecolor="#d8d8d8", edgecolor="#555555", lw=1.0, alpha=0.92, zorder=8))
    ax.add_patch(Circle((0, 0.55), radius=0.18, facecolor="#d8d8d8", edgecolor="#555555", lw=1.0, alpha=0.92, zorder=9))
    ax.plot([0, 0], [-0.48, -0.88], color="#777777", lw=1.2, zorder=8)


def _render_mouse_top_svg(*, render_h: int = 520) -> np.ndarray | None:
    svg_path = _mouse_top_svg_path()
    if svg_path is None:
        return None
    key = (int(render_h), int(render_h), str(svg_path))
    cached = _MOUSE_TOP_IMAGE_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    try:
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(str(svg_path))
        view = renderer.viewBoxF()
        aspect = float(view.width()) / max(float(view.height()), 1.0)
        render_w = max(int(round(float(render_h) * aspect)), 8)
        qimg = QImage(render_w, int(render_h), QImage.Format.Format_RGBA8888)
        qimg.fill(Qt.GlobalColor.transparent)
        painter = QPainter(qimg)
        renderer.render(painter, QRectF(0.0, 0.0, float(render_w), float(render_h)))
        painter.end()
        ptr = qimg.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=qimg.sizeInBytes())
        arr = arr.reshape((qimg.height(), qimg.bytesPerLine()))[:, : render_w * 4]
        rgba = arr.reshape((qimg.height(), render_w, 4)).copy()
        alpha = rgba[:, :, 3].astype(np.float64)
        rgba[:, :, 3] = np.clip(alpha * 0.72, 0, 255).astype(np.uint8)
        _MOUSE_TOP_IMAGE_CACHE[key] = rgba
        return rgba.copy()
    except Exception:
        return None


def _mouse_top_svg_path() -> Path | None:
    for candidate in _MOUSE_TOP_SVG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _write_transition_heatmaps(probability_summary: pd.DataFrame, out_dir: Path) -> list[str]:
    figures: list[str] = []
    plt = _matplotlib_pyplot()
    if plt is None or probability_summary.empty:
        return figures
    states = sorted(
        set(probability_summary["from_state"].astype(str))
        | set(probability_summary["to_state"].astype(str)),
        key=_transition_priority_index,
    )
    if not states:
        return figures
    network_path = out_dir / "batch_transition_probability_average_network.png"
    if _write_transition_network(probability_summary, network_path, states=states, plt=plt):
        figures.extend(_saved_figure_paths(network_path))
    panel_path = out_dir / "batch_transition_probability_group_networks.png"
    if _write_transition_group_network_panel(probability_summary, panel_path, states=states, plt=plt):
        figures.extend(_saved_figure_paths(panel_path))
    for group, sub in probability_summary.groupby("group", dropna=False):
        matrix = _transition_probability_matrix(sub, states, value_col=_transition_plot_value_col(sub))
        fig, ax = plt.subplots(figsize=(max(6.0, len(states) * 0.55), max(5.2, len(states) * 0.45)), facecolor="white")
        ax.set_facecolor("white")
        im = ax.imshow(matrix.to_numpy(dtype=np.float64), cmap="Greys", vmin=0.0, vmax=max(0.05, float(matrix.to_numpy().max())))
        ax.set_title(f"Frequency-weighted transition probability: {group}")
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")
        ax.set_xticks(np.arange(len(states)))
        ax.set_yticks(np.arange(len(states)))
        ax.set_xticklabels([_metric_display_name(state) for state in states], rotation=35, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_yticklabels([_metric_display_name(state) for state in states], fontsize=8)
        ax.tick_params(colors="#222222")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, label=_transition_plot_label(sub), fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        path = out_dir / f"batch_transition_probability_{_safe_filename(str(group))}.png"
        saved = _save_figure(fig, path, facecolor="white")
        plt.close(fig)
        figures.extend(saved)
    return figures


def _write_transition_network(
    probability_summary: pd.DataFrame,
    path: Path,
    *,
    states: list[str],
    plt,
) -> bool:
    if probability_summary.empty or not states:
        return False
    states = _top_transition_states(probability_summary, states, max_states=10)
    if len(states) < 2:
        return False
    value_col = _transition_plot_value_col(probability_summary)
    matrix = _transition_probability_matrix(probability_summary, states, value_col=value_col)
    max_prob = _transition_max_probability([matrix])
    if max_prob <= 0.0:
        return False

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin=0.0, vmax=max(0.05, max_prob))
    edge_cmap = plt.get_cmap("Blues")
    sm = ScalarMappable(norm=norm, cmap=edge_cmap)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="white", constrained_layout=True)
    _draw_transition_network_panel(ax, matrix, states, title="Average", plt=plt, norm=norm, edge_cmap=edge_cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.055, pad=0.035, label=_transition_plot_label(probability_summary))
    cbar.ax.tick_params(labelsize=8.5, colors="#222222")
    cbar.outline.set_edgecolor("#222222")
    _save_figure(fig, path, facecolor="white")
    plt.close(fig)
    return True


def _write_transition_group_network_panel(
    probability_summary: pd.DataFrame,
    path: Path,
    *,
    states: list[str],
    plt,
) -> bool:
    groups = _ordered_groups(list(probability_summary["group"].dropna().astype(str).unique()))
    if len(groups) < 2 or not states:
        return False
    group_a, group_b = groups[:2]
    selected = probability_summary[probability_summary["group"].astype(str).isin([group_a, group_b])].copy()
    states = _top_transition_states(selected, states, max_states=10)
    if len(states) < 2:
        return False
    value_col = _transition_plot_value_col(selected)
    matrix_a = _transition_probability_matrix(selected[selected["group"].astype(str) == group_a], states, value_col=value_col)
    matrix_b = _transition_probability_matrix(selected[selected["group"].astype(str) == group_b], states, value_col=value_col)
    max_prob = _transition_max_probability([matrix_a, matrix_b])
    if max_prob <= 0.0:
        return False

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin=0.0, vmax=max(0.05, max_prob))
    edge_cmap = plt.get_cmap("Blues")
    sm = ScalarMappable(norm=norm, cmap=edge_cmap)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.0),
        facecolor="white",
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]},
    )
    _draw_transition_network_panel(axes[0], matrix_a, states, title=str(group_a), plt=plt, norm=norm, edge_cmap=edge_cmap)
    _draw_transition_network_panel(axes[1], matrix_b, states, title=str(group_b), plt=plt, norm=norm, edge_cmap=edge_cmap)
    diff = matrix_a.to_numpy(dtype=np.float64) - matrix_b.to_numpy(dtype=np.float64)
    diff_matrix = pd.DataFrame(diff, index=states, columns=states)
    diff_abs = max(float(np.nanmax(np.abs(diff))) if diff.size else 0.0, 1e-9)
    diff_sm = _draw_transition_difference_network_panel(
        axes[2],
        diff_matrix,
        states,
        title=f"{group_a} minus {group_b}",
        plt=plt,
        vmax=diff_abs,
    )
    fig.suptitle("Frequency-weighted transition probability", fontsize=14, color="#111111")
    cbar = fig.colorbar(sm, ax=axes[:2], orientation="horizontal", fraction=0.055, pad=0.035, label=_transition_plot_label(selected))
    cbar.ax.tick_params(labelsize=8.5, colors="#222222")
    cbar.outline.set_edgecolor("#222222")
    if diff_sm is not None:
        diff_cbar = fig.colorbar(diff_sm, ax=axes[2], fraction=0.046, pad=0.04, label="Weighted probability difference")
        diff_cbar.ax.tick_params(labelsize=8.2, colors="#222222")
        diff_cbar.outline.set_edgecolor("#222222")
    _save_figure(fig, path, facecolor="white")
    plt.close(fig)
    return True


def _draw_transition_network_panel(ax, matrix: pd.DataFrame, states: list[str], *, title: str, plt, norm, edge_cmap) -> None:
    from matplotlib.patches import FancyArrowPatch

    values = matrix.to_numpy(dtype=np.float64)
    edge_rows = _transition_edge_rows(values)
    max_prob = max((prob for _, _, prob in edge_rows), default=0.0)
    keep_threshold = max(0.005, max_prob * 0.055)
    edge_rows = sorted(edge_rows, key=lambda item: item[2])
    strong_edges = [item for item in edge_rows if item[2] >= keep_threshold]
    min_edges = min(len(edge_rows), max(len(states) * 2, 4))
    if len(strong_edges) < min_edges:
        strong_edges = edge_rows[-min(len(edge_rows), len(states) * 4):]

    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_aspect("equal")
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, len(states), endpoint=False)
    radius = 1.0
    pos = {
        idx: np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64)
        for idx, angle in enumerate(angles)
    }

    for i, j, prob in strong_edges:
        start = pos[i]
        end = pos[j]
        norm_prob = float(norm(prob))
        delta = (j - i) % len(states)
        rad = 0.18 if delta < len(states) / 2 else -0.18
        color = edge_cmap(0.18 + 0.82 * norm_prob)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.0 + 13.0 * norm_prob,
            linewidth=0.75 + 4.2 * norm_prob,
            color=color,
            alpha=0.22 + 0.70 * norm_prob,
            shrinkA=24,
            shrinkB=24,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    for idx, state in enumerate(states):
        x, y = pos[idx]
        node_color = _pastel_tab10_color(plt, idx)
        ax.scatter([x], [y], s=920, facecolor=node_color, edgecolor="white", linewidth=1.8, zorder=5)
        ax.scatter([x], [y], s=920, facecolor="none", edgecolor="#334155", linewidth=0.55, zorder=6)
        ax.text(
            x,
            y,
            _transition_node_label(state),
            ha="center",
            va="center",
            color="#111827",
            fontsize=6.9,
            weight="bold",
            linespacing=0.92,
            zorder=7,
        )
    ax.set_title(str(title), fontsize=11, color="#111111", pad=10)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.24, 1.24)


def _draw_transition_difference_network_panel(ax, matrix: pd.DataFrame, states: list[str], *, title: str, plt, vmax: float):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import FancyArrowPatch

    values = matrix.to_numpy(dtype=np.float64)
    edge_rows: list[tuple[int, int, float]] = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values[i, j])
            if i != j and np.isfinite(value) and abs(value) > 0.0:
                edge_rows.append((i, j, value))
    edge_rows = sorted(edge_rows, key=lambda item: abs(item[2]))
    max_abs = max((abs(value) for _, _, value in edge_rows), default=0.0)
    keep_threshold = max(0.002, max_abs * 0.07)
    strong_edges = [item for item in edge_rows if abs(item[2]) >= keep_threshold]
    min_edges = min(len(edge_rows), max(len(states) * 2, 4))
    if len(strong_edges) < min_edges:
        strong_edges = edge_rows[-min(len(edge_rows), len(states) * 4):]

    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_aspect("equal")
    norm = TwoSlopeNorm(vmin=-max(float(vmax), 1e-9), vcenter=0.0, vmax=max(float(vmax), 1e-9))
    cmap = plt.get_cmap("RdBu_r")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, len(states), endpoint=False)
    radius = 1.0
    pos = {
        idx: np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64)
        for idx, angle in enumerate(angles)
    }

    for i, j, value in strong_edges:
        start = pos[i]
        end = pos[j]
        mag = abs(value) / max(max_abs, 1e-9)
        delta = (j - i) % len(states)
        rad = 0.18 if delta < len(states) / 2 else -0.18
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.0 + 13.0 * mag,
            linewidth=0.75 + 4.2 * mag,
            color=sm.to_rgba(value),
            alpha=0.25 + 0.68 * mag,
            shrinkA=24,
            shrinkB=24,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)

    for idx, state in enumerate(states):
        x, y = pos[idx]
        node_color = _pastel_tab10_color(plt, idx)
        ax.scatter([x], [y], s=920, facecolor=node_color, edgecolor="white", linewidth=1.8, zorder=5)
        ax.scatter([x], [y], s=920, facecolor="none", edgecolor="#334155", linewidth=0.55, zorder=6)
        ax.text(
            x,
            y,
            _transition_node_label(state),
            ha="center",
            va="center",
            color="#111827",
            fontsize=6.9,
            weight="bold",
            linespacing=0.92,
            zorder=7,
        )
    ax.set_title(str(title), fontsize=11, color="#111111", pad=10)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.24, 1.24)
    return sm


def _transition_edge_rows(values: np.ndarray) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            prob = float(values[i, j])
            if i != j and np.isfinite(prob) and prob > 0.0:
                rows.append((i, j, prob))
    return rows


def _transition_max_probability(matrices: list[pd.DataFrame]) -> float:
    max_prob = 0.0
    for matrix in matrices:
        values = matrix.to_numpy(dtype=np.float64)
        edge_rows = _transition_edge_rows(values)
        if edge_rows:
            max_prob = max(max_prob, max(prob for _, _, prob in edge_rows))
    return max_prob


def _pastel_tab10_color(plt, idx: int) -> tuple[float, float, float, float]:
    rgba = np.asarray(plt.get_cmap("tab10")(int(idx) % 10), dtype=np.float64)
    rgba[:3] = rgba[:3] * 0.55 + 0.45
    rgba[3] = 1.0
    return tuple(float(v) for v in rgba)


def _transition_node_label(state: str) -> str:
    label = _metric_display_name(state)
    words = label.replace("-", "- ").split()
    if len(label) <= 12 or len(words) <= 1:
        return label.replace("- ", "-")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > 11:
            lines.append(current.replace("- ", "-"))
            current = word
        else:
            current = candidate
    if current:
        lines.append(current.replace("- ", "-"))
    return "\n".join(lines[:3])


def _transition_plot_value_col(probability_summary: pd.DataFrame) -> str:
    if probability_summary is not None and "transition_mass" in probability_summary.columns:
        values = pd.to_numeric(probability_summary["transition_mass"], errors="coerce")
        if np.isfinite(values).any():
            return "transition_mass"
    return "probability"


def _transition_plot_label(probability_summary: pd.DataFrame) -> str:
    return (
        "Transition mass (state frequency x probability)"
        if _transition_plot_value_col(probability_summary) == "transition_mass"
        else "Transition probability"
    )


def _transition_probability_matrix(probability_summary: pd.DataFrame, states: list[str], *, value_col: str = "probability") -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    if probability_summary.empty:
        return matrix
    if value_col not in probability_summary.columns:
        value_col = "probability"
    grouped = (
        probability_summary.groupby(["from_state", "to_state"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )
    for _, row in grouped.iterrows():
        src = str(row.get("from_state", "") or "")
        dst = str(row.get("to_state", "") or "")
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] = float(row.get(value_col, 0.0) or 0.0)
    return matrix


def _top_transition_states(probability_summary: pd.DataFrame, states: list[str], *, max_states: int) -> list[str]:
    if len(states) <= max_states:
        return states
    value_col = _transition_plot_value_col(probability_summary)
    rows: list[tuple[str, float]] = []
    for state in states:
        outgoing = probability_summary.loc[probability_summary["from_state"].astype(str) == str(state), value_col]
        incoming = probability_summary.loc[probability_summary["to_state"].astype(str) == str(state), value_col]
        mass = float(pd.to_numeric(outgoing, errors="coerce").fillna(0.0).sum())
        mass += float(pd.to_numeric(incoming, errors="coerce").fillna(0.0).sum())
        rows.append((state, mass))
    keep = {state for state, _mass in sorted(rows, key=lambda item: item[1], reverse=True)[:max_states]}
    return [state for state in states if state in keep]


def _safe_filename(value: str) -> str:
    text = str(value or "unspecified").strip() or "unspecified"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe.strip("_") or "unspecified"


def _safe_col(value: str) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _is_passive_social_metric(metric: Any) -> bool:
    return _metric_leaf(metric).startswith("passive_")


def _save_figure(fig, path: Path, *, facecolor: str = "white") -> list[str]:
    """Save a figure as high-resolution PNG and matching PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    fig.savefig(path, dpi=_EXPORT_DPI, facecolor=facecolor)
    if path.exists():
        saved.append(str(path))
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=_EXPORT_DPI, facecolor=facecolor)
    if pdf_path.exists():
        saved.append(str(pdf_path))
    return saved


def _saved_figure_paths(path: Path) -> list[str]:
    path = Path(path)
    paths = [candidate for candidate in (path, path.with_suffix(".pdf")) if candidate.exists()]
    return [str(candidate) for candidate in paths]


def _combine_group_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["all"] * len(df), index=df.index)
    values = df[group_cols].fillna("").astype(str)
    if len(group_cols) == 1:
        out = values[group_cols[0]].replace("", "unspecified")
        return out
    return values.apply(
        lambda row: " / ".join(part if part else "unspecified" for part in row),
        axis=1,
    )


def _social_behavior_summary_source(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for plotted social behavior summaries, excluding state/contact helpers."""
    if df is None or df.empty:
        return pd.DataFrame()
    mask = _behavior_flag(df)
    if "scope" in df.columns:
        mask &= df["scope"].fillna("").astype(str).str.startswith("social")
    metric = _metric_base_text(df)
    metric_leaf = metric.map(_metric_leaf)
    mask &= ~metric.map(_is_mobility_state_metric)
    mask &= ~metric.map(_is_mask_contact_metric)
    mask &= ~metric_leaf.map(_is_passive_social_metric)
    mask &= ~metric_leaf.isin(_REDUNDANT_SOCIAL_SUMMARY_METRICS)
    return _add_attack_alias_rows(df[mask].copy())


def _add_attack_alias_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "metric_base" not in df.columns:
        return df
    bases = df["metric_base"].fillna("").astype(str).map(_metric_leaf)
    if not bases.eq("fighting").any() or bases.eq("attacks").any():
        return df
    attack_rows = df[bases.eq("fighting")].copy()
    attack_rows["metric_base"] = "attacks"
    if "metric" in attack_rows.columns:
        attack_rows["metric"] = attack_rows["metric"].astype(str).str.replace("fighting", "attacks", regex=False)
    return pd.concat([df, attack_rows], ignore_index=True)


def _behavior_frequency_value_col(df: pd.DataFrame) -> str:
    if df is not None and "bouts" in df.columns:
        values = pd.to_numeric(df["bouts"], errors="coerce")
        if bool(np.isfinite(values).any()):
            return "bouts"
    return "frequency_per_min"


def _mask_contact_summary_source(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mask = _behavior_flag(df)
    if "scope" in df.columns:
        mask &= df["scope"].fillna("").astype(str).str.startswith("social")
    mask &= _metric_base_text(df).map(_is_mask_contact_metric)
    return df[mask].copy()


def _behavior_flag(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if "is_behavior" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["is_behavior"].fillna(False).astype(bool)


def _metric_base_text(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    if "metric_base" not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df["metric_base"].fillna("").astype(str)


def _metric_leaf(metric: Any) -> str:
    text = str(metric or "").strip().lower()
    if "__" in text:
        text = text.rsplit("__", 1)[-1]
    return text


def _is_mobility_state_metric(metric: Any) -> bool:
    return _metric_leaf(metric) in _MOBILITY_STATE_METRICS


def _is_mask_contact_metric(metric: Any) -> bool:
    return _metric_leaf(metric) in _MASK_CONTACT_METRICS


def _is_cumulative_overview_metric(metric: Any) -> bool:
    text = _metric_leaf(metric)
    return any(token in text for token in _CUMULATIVE_OVERVIEW_TOKENS)


def _filter_metric_overview_table(metrics_table: pd.DataFrame, *, cumulative: bool) -> pd.DataFrame:
    if metrics_table is None or metrics_table.empty or "metric_base" not in metrics_table.columns:
        return pd.DataFrame()
    is_cumulative = metrics_table["metric_base"].map(_is_cumulative_overview_metric)
    return metrics_table[is_cumulative if cumulative else ~is_cumulative].copy()


def _aggregate_values(df: pd.DataFrame, *, value_col: str, behavior: bool) -> pd.DataFrame:
    if value_col not in df.columns or "metric_base" not in df.columns:
        return pd.DataFrame()
    mask = df.get("is_behavior", False).fillna(False).astype(bool) if "is_behavior" in df.columns else pd.Series(False, index=df.index)
    data = df[mask] if behavior else df[~mask]
    data = data.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data[np.isfinite(data[value_col])]
    if data.empty:
        return pd.DataFrame()
    grouped = data.groupby(["metric_base", "_group"], dropna=False)[value_col]
    out = grouped.agg(["mean", "std", "count", "min", "max"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["value"] = value_col
    out["category"] = "social_behavior" if behavior else "metric"
    return out.rename(columns={"_group": "group", "count": "n"})


def _comparison_table(df: pd.DataFrame, *, method: str = "holm_ttest") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    df = _add_attack_alias_rows(df.copy())
    value_cols = ["mean", _behavior_frequency_value_col(df), "mean_bout_duration_s", "cumulative_duration_s"]
    for value_col in dict.fromkeys(value_cols):
        if value_col not in df.columns:
            continue
        data = df.copy()
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
        data = data[np.isfinite(data[value_col])]
        if data.empty or "metric_base" not in data.columns:
            continue
        for metric, sub in data.groupby("metric_base"):
            grouped = {
                str(group): vals[value_col].dropna().to_numpy(dtype=np.float64)
                for group, vals in sub.groupby("_group")
                if len(vals[value_col].dropna()) >= 2
            }
            grouped = {label: values[np.isfinite(values)] for label, values in grouped.items()}
            grouped = {label: values for label, values in grouped.items() if values.size >= 2}
            if len(grouped) < 2:
                continue
            for item in _pairwise_comparisons(grouped, method=method):
                rows.append({
                    "metric_base": metric,
                    "value": value_col,
                    **item,
                })
    return pd.DataFrame(rows)


def _pairwise_comparisons(grouped: dict[str, np.ndarray], *, method: str) -> list[dict[str, Any]]:
    requested = str(method or "holm_ttest").lower()
    if requested == "tukey":
        tukey_rows = _tukey_comparisons(grouped)
        if tukey_rows:
            return tukey_rows
    return _holm_ttest_comparisons(grouped)


def _holm_ttest_comparisons(grouped: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    try:
        from scipy import stats
    except Exception:
        return []
    labels = sorted(grouped)
    raw_rows: list[dict[str, Any]] = []
    for i, group_a in enumerate(labels):
        for group_b in labels[i + 1:]:
            stat = stats.ttest_ind(grouped[group_a], grouped[group_b], equal_var=False, nan_policy="omit")
            raw_rows.append({
                "group_a": group_a,
                "group_b": group_b,
                "groups": f"{group_a} vs {group_b}",
                "test": "welch_ttest_holm",
                "p_value": float(stat.pvalue),
            })
    adjusted = _holm_adjust([row["p_value"] for row in raw_rows])
    for row, p_adj in zip(raw_rows, adjusted):
        row["p_adj"] = p_adj
        row["significant"] = bool(np.isfinite(p_adj) and p_adj < 0.05)
    return raw_rows


def _tukey_comparisons(grouped: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except Exception:
        return []
    values: list[float] = []
    labels: list[str] = []
    for label, arr in sorted(grouped.items()):
        finite = arr[np.isfinite(arr)]
        values.extend(finite.tolist())
        labels.extend([label] * len(finite))
    if len(set(labels)) < 2:
        return []
    try:
        result = pairwise_tukeyhsd(endog=np.asarray(values, dtype=np.float64), groups=np.asarray(labels, dtype=object), alpha=0.05)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    table_rows = getattr(result, "_results_table", None)
    data = getattr(table_rows, "data", [])
    for raw in data[1:]:
        group_a = str(raw[0])
        group_b = str(raw[1])
        p_adj = float(raw[3])
        rows.append({
            "group_a": group_a,
            "group_b": group_b,
            "groups": f"{group_a} vs {group_b}",
            "test": "tukey_hsd",
            "p_value": p_adj,
            "p_adj": p_adj,
            "significant": bool(np.isfinite(p_adj) and p_adj < 0.05),
        })
    return rows


def _holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out.tolist()
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, original_idx in enumerate(order):
        adjusted = min((m - rank) * p[original_idx], 1.0)
        running = max(running, adjusted)
        out[original_idx] = running
    return out.tolist()


def _write_figures(
    metrics_table: pd.DataFrame,
    social_table: pd.DataFrame,
    out_dir: Path,
    source_df: pd.DataFrame,
    *,
    mask_contact_table: pd.DataFrame,
    plot_style: str,
    tab10_color_map: str | dict[str, int] | None,
    comparison_table: pd.DataFrame,
) -> list[str]:
    figures: list[str] = []
    plt = _matplotlib_pyplot()
    if plt is None:
        return figures

    if not metrics_table.empty:
        overview_metrics = _filter_metric_overview_table(metrics_table, cumulative=False)
        if not overview_metrics.empty:
            fig_path = out_dir / "batch_metrics_overview.png"
            _bar_strip_overview(
                overview_metrics[overview_metrics["value"] == "mean"],
                fig_path,
                title="Metrics by group",
                y_label="Mean value",
                raw_data=source_df,
                value_col="mean",
                plt=plt,
                plot_style=plot_style,
                tab10_color_map=tab10_color_map,
                comparison_table=comparison_table,
            )
            if fig_path.exists():
                figures.extend(_saved_figure_paths(fig_path))
        cumulative_metrics = _filter_metric_overview_table(metrics_table, cumulative=True)
        if not cumulative_metrics.empty:
            fig_path = out_dir / "batch_cumulative_metrics.png"
            _bar_strip_overview(
                cumulative_metrics[cumulative_metrics["value"] == "mean"],
                fig_path,
                title="Cumulative metrics by group",
                y_label="Mean cumulative value",
                raw_data=source_df,
                value_col="mean",
                plt=plt,
                plot_style=plot_style,
                tab10_color_map=tab10_color_map,
                comparison_table=comparison_table,
            )
            if fig_path.exists():
                figures.extend(_saved_figure_paths(fig_path))

    if not social_table.empty:
        social_frequency_value = "bouts" if social_table["value"].astype(str).eq("bouts").any() else "frequency_per_min"
        for value, stem, title, ylabel in (
            (social_frequency_value, "frequency", "Social behavior frequency", "Bouts"),
            ("mean_bout_duration_s", "mean_bout_duration_s", "Social behavior bout duration", "Mean bout duration (s)"),
            ("cumulative_duration_s", "cumulative_duration_s", "Social behavior cumulative duration", "Cumulative duration (s)"),
        ):
            fig_path = out_dir / f"batch_social_{stem}.png"
            _bar_strip_overview(
                social_table[social_table["value"] == value],
                fig_path,
                title=title,
                y_label=ylabel,
                raw_data=source_df,
                value_col=value,
                plt=plt,
                plot_style=plot_style,
                tab10_color_map=tab10_color_map,
                comparison_table=comparison_table,
            )
            if fig_path.exists():
                figures.extend(_saved_figure_paths(fig_path))
    if mask_contact_table is not None and not mask_contact_table.empty:
        mask_frequency_value = "bouts" if mask_contact_table["value"].astype(str).eq("bouts").any() else "frequency_per_min"
        for value, stem, title, ylabel in (
            (mask_frequency_value, "frequency", "Mask contact frequency", "Bouts"),
            ("mean_bout_duration_s", "mean_bout_duration_s", "Mask contact bout duration", "Mean bout duration (s)"),
            ("cumulative_duration_s", "cumulative_duration_s", "Mask contact cumulative duration", "Cumulative duration (s)"),
        ):
            fig_path = out_dir / f"batch_mask_contact_{stem}.png"
            _bar_strip_overview(
                mask_contact_table[mask_contact_table["value"] == value],
                fig_path,
                title=title,
                y_label=ylabel,
                raw_data=source_df,
                value_col=value,
                plt=plt,
                plot_style=plot_style,
                tab10_color_map=tab10_color_map,
                comparison_table=comparison_table,
            )
            if fig_path.exists():
                figures.extend(_saved_figure_paths(fig_path))
    return figures


def _bar_strip_overview(
    data: pd.DataFrame,
    path: Path,
    *,
    title: str,
    y_label: str,
    raw_data: pd.DataFrame,
    value_col: str,
    plt,
    plot_style: str = "box_strip",
    tab10_color_map: str | dict[str, int] | None = None,
    comparison_table: pd.DataFrame | None = None,
) -> None:
    if plt is None:
        return
    if data.empty:
        return
    top_metrics = _overview_metric_selection(data, max_metrics=12)
    data = data[data["metric_base"].isin(top_metrics)].copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="metric_base", columns="group", values="mean", aggfunc="mean").loc[top_metrics]
    errors = data.pivot_table(index="metric_base", columns="group", values="sem", aggfunc="mean").reindex_like(pivot)
    groups = _ordered_groups(list(pivot.columns))
    pivot = pivot.reindex(columns=groups)
    errors = errors.reindex(columns=groups)
    metrics = list(pivot.index)
    x = np.arange(len(metrics), dtype=np.float64)
    plot_style = _normalize_plot_style(plot_style)
    color_map = _parse_tab10_color_map(tab10_color_map)
    width = min(0.72 / max(len(groups), 1), 0.26)
    fig, ax = plt.subplots(figsize=(max(9, len(top_metrics) * 0.9), 5.4))

    raw = raw_data.copy() if raw_data is not None else pd.DataFrame()
    if not raw.empty and value_col in raw.columns and "_group" in raw.columns:
        raw = raw[raw["metric_base"].isin(metrics)].copy()
        raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    else:
        raw = pd.DataFrame()
    if plot_style == "box_strip":
        _box_strip_overview(
            data,
            path,
            metrics=metrics,
            groups=groups,
            title=title,
            y_label=y_label,
            raw=raw,
            value_col=value_col,
            plt=plt,
            tab10_color_map=tab10_color_map,
            comparison_table=comparison_table,
        )
        return

    for group_idx, group in enumerate(groups):
        xpos = x + (group_idx - (len(groups) - 1) / 2.0) * width
        heights = pivot[group].to_numpy(dtype=np.float64)
        yerr = errors[group].to_numpy(dtype=np.float64) if group in errors.columns else None
        color = _tab10_color(plt, str(group), group_idx, color_map)
        if plot_style == "box_strip" and not raw.empty:
            box_values = []
            box_positions = []
            for metric_idx, metric in enumerate(metrics):
                points = raw[
                    (raw["metric_base"].astype(str) == str(metric))
                    & (raw["_group"].astype(str) == str(group))
                ][value_col].dropna().to_numpy(dtype=np.float64)
                points = points[np.isfinite(points)]
                if points.size:
                    box_values.append(points)
                    box_positions.append(float(xpos[metric_idx]))
            if box_values:
                bp = ax.boxplot(
                    box_values,
                    positions=box_positions,
                    widths=max(width * 0.82, 0.04),
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.28)
                    patch.set_edgecolor(color)
                    patch.set_linewidth(1.1)
                for key in ("whiskers", "caps", "medians"):
                    for artist in bp[key]:
                        artist.set_color(color)
                        artist.set_linewidth(1.0)
                ax.scatter([], [], color=color, alpha=0.7, label=str(group))
        else:
            ax.bar(
                xpos,
                heights,
                width=width,
                yerr=yerr,
                capsize=3,
                color=color,
                alpha=0.68,
                edgecolor="#1f2937",
                linewidth=0.7,
                label=str(group),
            )
        if raw.empty:
            continue
        for metric_idx, metric in enumerate(metrics):
            points = raw[
                (raw["metric_base"].astype(str) == str(metric))
                & (raw["_group"].astype(str) == str(group))
            ][value_col].dropna().to_numpy(dtype=np.float64)
            points = points[np.isfinite(points)]
            if points.size == 0:
                continue
            jitter = _strip_jitter(points.size, width)
            ax.scatter(
                np.full(points.size, xpos[metric_idx]) + jitter,
                points,
                s=18,
                facecolor=color,
                edgecolor="#111827",
                linewidth=0.35,
                alpha=0.9,
                zorder=4,
            )

    _annotate_pairwise_pvalues(
        ax,
        comparison_table,
        metrics=metrics,
        groups=groups,
        value_col=value_col,
        raw=raw,
        pivot=pivot,
        errors=errors,
        width=width,
    )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save_figure(fig, path, facecolor="white")
    plt.close(fig)


def _overview_metric_selection(data: pd.DataFrame, *, max_metrics: int = 12) -> list[str]:
    top_metrics = (
        data.groupby("metric_base")["mean"]
        .mean()
        .abs()
        .sort_values(ascending=False)
        .head(max_metrics)
        .index
        .tolist()
    )
    if data is None or data.empty or "metric_base" not in data.columns:
        return top_metrics
    category = data["category"].fillna("").astype(str) if "category" in data.columns else pd.Series("", index=data.index)
    is_social_summary = category.eq("social_behavior").any()
    if not is_social_summary:
        return top_metrics
    available = set(data["metric_base"].fillna("").astype(str).map(_metric_leaf))
    for required in _SOCIAL_SUMMARY_REQUIRED_METRICS:
        if required not in available or required in top_metrics:
            continue
        if len(top_metrics) >= max_metrics and top_metrics:
            top_metrics = top_metrics[:-1]
        top_metrics.append(required)
    return top_metrics


def _box_strip_overview(
    data: pd.DataFrame,
    path: Path,
    *,
    metrics: list[str],
    groups: list[str],
    title: str,
    y_label: str,
    raw: pd.DataFrame,
    value_col: str,
    plt,
    tab10_color_map: str | dict[str, int] | None = None,
    comparison_table: pd.DataFrame | None = None,
) -> None:
    if data.empty or not metrics or not groups:
        return
    n_metrics = len(metrics)
    n_cols = 1 if n_metrics == 1 else min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_w = max(3.0 * n_cols, 3.4 if n_metrics == 1 else 6.0)
    fig_h = max(2.85 * n_rows, 3.2)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(fig_w, fig_h), facecolor="white")
    color_map = _parse_tab10_color_map(tab10_color_map)

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx // n_cols][metric_idx % n_cols]
        ax.set_facecolor("white")
        values_by_group: dict[str, np.ndarray] = {}
        fallback_by_group: dict[str, np.ndarray] = {}
        for group in groups:
            points = np.array([], dtype=np.float64)
            if raw is not None and not raw.empty:
                points = raw[
                    (raw["metric_base"].astype(str) == str(metric))
                    & (raw["_group"].astype(str) == str(group))
                ][value_col].dropna().to_numpy(dtype=np.float64)
                points = points[np.isfinite(points)]
            if points.size == 0:
                agg = data[
                    (data["metric_base"].astype(str) == str(metric))
                    & (data["group"].astype(str) == str(group))
                ]
                points = _aggregate_row_to_points(agg)
                fallback_by_group[str(group)] = points
            values_by_group[str(group)] = points

        positions: list[float] = []
        box_values: list[np.ndarray] = []
        for group_idx, group in enumerate(groups):
            points = values_by_group.get(str(group), np.array([], dtype=np.float64))
            if points.size:
                positions.append(float(group_idx))
                box_values.append(points)
        if box_values:
            bp = ax.boxplot(
                box_values,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "white", "linewidth": 1.4},
                whiskerprops={"color": "#555555", "linewidth": 0.9},
                capprops={"color": "#555555", "linewidth": 0.9},
            )
            for patch_idx, patch in enumerate(bp["boxes"]):
                group_idx = int(round(positions[patch_idx]))
                color = _tab10_color(plt, str(groups[group_idx]), group_idx, color_map)
                patch.set_facecolor(color)
                patch.set_alpha(0.82)
                patch.set_edgecolor("#4b5563")
                patch.set_linewidth(0.9)

        for group_idx, group in enumerate(groups):
            points = values_by_group.get(str(group), np.array([], dtype=np.float64))
            if points.size == 0:
                continue
            jitter = _strip_jitter(points.size, 0.55)
            face = "white" if str(group) not in fallback_by_group else _tab10_color(plt, str(group), group_idx, color_map)
            ax.scatter(
                np.full(points.size, float(group_idx)) + jitter,
                points,
                s=24,
                facecolor=face,
                edgecolor="#374151",
                linewidth=0.55,
                alpha=0.96,
                zorder=4,
            )
            mean = float(np.nanmean(points))
            if np.isfinite(mean):
                ax.scatter(
                    [float(group_idx)],
                    [mean],
                    s=34,
                    marker="D",
                    facecolor="white",
                    edgecolor="#111827",
                    linewidth=0.55,
                    zorder=5,
                )

        _annotate_boxstrip_pvalues(
            ax,
            comparison_table,
            metric=str(metric),
            groups=groups,
            value_col=value_col,
            values_by_group=values_by_group,
        )
        ax.set_title(title if n_metrics == 1 else _metric_display_name(metric), fontsize=9.5, pad=8)
        ax.set_xticks(np.arange(len(groups), dtype=np.float64))
        ax.set_xticklabels([
            _group_tick_label(group, values_by_group.get(str(group), np.array([], dtype=np.float64)).size)
            for group in groups
        ], fontsize=8)
        ax.tick_params(axis="y", labelsize=8, colors="#333333")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6b7280")
        ax.spines["bottom"].set_color("#6b7280")
        if metric_idx % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=8.5)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xlim(-0.65, max(len(groups) - 0.35, 0.65))

    for unused_idx in range(n_metrics, n_rows * n_cols):
        axes[unused_idx // n_cols][unused_idx % n_cols].axis("off")
    if n_metrics > 1:
        fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout()
    if n_metrics > 1:
        fig.subplots_adjust(top=0.91)
    _save_figure(fig, path, facecolor="white")
    plt.close(fig)


def _aggregate_row_to_points(rows: pd.DataFrame) -> np.ndarray:
    if rows is None or rows.empty:
        return np.array([], dtype=np.float64)
    row = rows.iloc[0]
    mean = float(row.get("mean", np.nan))
    if not np.isfinite(mean):
        return np.array([], dtype=np.float64)
    n = int(max(float(row.get("n", 1) or 1), 1))
    std = float(row.get("std", np.nan))
    min_v = float(row.get("min", np.nan))
    max_v = float(row.get("max", np.nan))
    if n <= 1 or not np.isfinite(std) or std <= 0:
        return np.array([mean], dtype=np.float64)
    offsets = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    points = mean + offsets * std
    if np.isfinite(min_v) and np.isfinite(max_v) and min_v < max_v:
        points = np.clip(points, min_v, max_v)
    return points


def _group_tick_label(group: str, n: int) -> str:
    text = str(group or "unspecified")
    return f"{text}\nn={int(max(n, 0))}"


def _ordered_groups(groups: list[Any]) -> list[str]:
    return sorted([str(group) for group in groups], key=_group_sort_key)


def _group_sort_key(group: str) -> tuple[int, str]:
    text = str(group or "").strip()
    key = text.lower().replace(" ", "").replace("-", "_")
    priority = {
        "wt": 0,
        "wildtype": 0,
        "wild_type": 0,
        "control": 0,
        "ctrl": 0,
        "het": 1,
        "heterozygous": 1,
        "treated": 2,
        "ko": 2,
        "mutant": 2,
        "homozygous": 2,
        "unspecified": 99,
        "unknown": 99,
        "": 99,
    }.get(key, 20)
    return (priority, text.casefold())


def _metric_display_name(metric: Any) -> str:
    text = _metric_leaf(metric)
    mapping = {
        "mask_contact": "Mask contact",
        "nose2nose": "Nose-to-nose",
        "nose_to_nose": "Nose-to-nose",
        "a_nose2anogenital_b": "A to B anogenital",
        "b_nose2anogenital_a": "B to A anogenital",
        "a_nose2body_b": "A to B body",
        "b_nose2body_a": "B to A body",
        "a_following_b": "A follows B",
        "b_following_a": "B follows A",
        "a_chasing_b": "A chases B",
        "b_chasing_a": "B chases A",
        "a_approaches_b": "A approaches B",
        "b_approaches_a": "B approaches A",
        "a_withdraws_from_b": "A withdraws from B",
        "b_withdraws_from_a": "B withdraws from A",
        "a_escapes_b": "A escapes B",
        "b_escapes_a": "B escapes A",
        "sidebyside": "Side-by-side",
        "sidereside": "Side-reverse",
        "fighting": "Fighting",
        "attacks": "Attacks",
        "a_oriented_toward_b": "A oriented toward B",
        "b_oriented_toward_a": "B oriented toward A",
        "a_withdrawal_after_contact_b": "A withdrawal after contact B",
        "b_withdrawal_after_contact_a": "B withdrawal after contact A",
        "passive_anogenital": "Passive anogenital",
        "passive_investigation": "Passive investigation",
        "passive_being_followed": "Passive followed",
        "passive_being_chased": "Passive chased",
        "passive_withdrawal": "Passive withdrawal",
        "transition_entropy_bits": "Transition entropy",
        "switch_rate_per_min": "Switch rate",
        "body_speed_cm_s": "Speed (cm/s)",
        "body_speed_px_s": "Speed (px/s)",
        "distance_traveled_cm": "Cumulative distance (cm)",
        "distance_traveled_px": "Cumulative distance (px)",
    }
    return mapping.get(text, text.replace("_", " ").capitalize())


def _annotate_boxstrip_pvalues(
    ax,
    comparison_table: pd.DataFrame | None,
    *,
    metric: str,
    groups: list[str],
    value_col: str,
    values_by_group: dict[str, np.ndarray],
) -> None:
    if comparison_table is None or comparison_table.empty or len(groups) < 2:
        return
    required = {"metric_base", "value", "group_a", "group_b", "p_adj"}
    if not required.issubset(set(comparison_table.columns)):
        return
    comps = comparison_table.copy()
    comps = comps[
        (comps["metric_base"].astype(str) == str(metric))
        & (comps["value"].astype(str) == str(value_col))
    ]
    if comps.empty:
        return
    comps["p_adj"] = pd.to_numeric(comps["p_adj"], errors="coerce")
    comps = comps[np.isfinite(comps["p_adj"])]
    if comps.empty:
        return
    if len(groups) > 2:
        significant = comps[comps["p_adj"] < 0.05].sort_values("p_adj")
        if not significant.empty:
            comps = significant.head(3)
        else:
            primary_groups = [group for group in groups if _group_sort_key(str(group))[0] < 99]
            primary_groups = primary_groups[:2] if len(primary_groups) >= 2 else groups[:2]
            wanted = {str(primary_groups[0]), str(primary_groups[1])}
            primary_comp = comps[
                comps.apply(
                    lambda row: {str(row.get("group_a", "") or ""), str(row.get("group_b", "") or "")} == wanted,
                    axis=1,
                )
            ]
            comps = primary_comp.head(1) if not primary_comp.empty else comps.sort_values("p_adj").head(1)
    else:
        comps = comps.sort_values("p_adj").head(1)
    group_to_idx = {str(group): idx for idx, group in enumerate(groups)}
    y_values = np.concatenate([
        arr[np.isfinite(arr)]
        for arr in values_by_group.values()
        if arr is not None and arr.size
    ]) if values_by_group else np.array([], dtype=np.float64)
    if y_values.size == 0:
        return
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    span = max(y_max - y_min, abs(y_max) * 0.15, 1.0)
    level = 0
    for _, row in comps.iterrows():
        group_a = str(row.get("group_a", "") or "")
        group_b = str(row.get("group_b", "") or "")
        if group_a not in group_to_idx or group_b not in group_to_idx:
            continue
        x1 = group_to_idx[group_a]
        x2 = group_to_idx[group_b]
        if x2 < x1:
            x1, x2 = x2, x1
        y = y_max + span * (0.10 + 0.12 * level)
        h = span * 0.035
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333333", linewidth=0.75, clip_on=False)
        ax.text(
            (x1 + x2) / 2.0,
            y + h + span * 0.015,
            _p_label_boxstrip(float(row.get("p_adj", np.nan))),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            clip_on=False,
        )
        level += 1
    if level:
        ax.set_ylim(top=y_max + span * (0.24 + 0.12 * max(level - 1, 0)))


def _p_label_boxstrip(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    suffix = " n.s." if p_value >= 0.05 else ""
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}{suffix}"


def _strip_jitter(n_points: int, width: float) -> np.ndarray:
    if n_points <= 1:
        return np.zeros(max(n_points, 0), dtype=np.float64)
    span = max(float(width) * 0.32, 0.01)
    return np.linspace(-span, span, int(n_points), dtype=np.float64)


def _normalize_plot_style(value: str) -> str:
    text = str(value or "box_strip").strip().lower()
    if text in {"bar", "bar_strip", "bar+strip", "barplot", "barplot_strip"}:
        return "bar_strip"
    if text in {"box", "boxplot", "box_strip", "boxplot_strip", "boxplot+stripplot"}:
        return "box_strip"
    return "box_strip"


def _parse_tab10_color_map(value: str | dict[str, int] | None) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(k).strip(): int(v) % 10 for k, v in value.items() if str(k).strip()}
    text = str(value or "").strip()
    if not text:
        return {}
    out: dict[str, int] = {}
    for chunk in text.replace(";", ",").split(","):
        if "=" not in chunk:
            continue
        key, raw_idx = chunk.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            out[key] = int(raw_idx.strip()) % 10
        except Exception:
            continue
    return out


def _tab10_color(plt, group: str, group_idx: int, color_map: dict[str, int]):
    cmap = plt.get_cmap("tab10")
    color_idx = color_map.get(str(group), group_idx)
    return cmap(int(color_idx) % 10)


def _matplotlib_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _annotate_pairwise_pvalues(
    ax,
    comparison_table: pd.DataFrame | None,
    *,
    metrics: list[str],
    groups: list[str],
    value_col: str,
    raw: pd.DataFrame,
    pivot: pd.DataFrame,
    errors: pd.DataFrame,
    width: float,
) -> None:
    if comparison_table is None or comparison_table.empty:
        return
    if "p_adj" not in comparison_table.columns:
        return
    comps = comparison_table.copy()
    comps = comps[comps["value"].astype(str) == str(value_col)]
    comps["p_adj"] = pd.to_numeric(comps["p_adj"], errors="coerce")
    comps = comps[np.isfinite(comps["p_adj"]) & (comps["p_adj"] < 0.05)]
    if comps.empty:
        return
    group_to_idx = {str(group): idx for idx, group in enumerate(groups)}
    metric_to_idx = {str(metric): idx for idx, metric in enumerate(metrics)}
    y_min, y_max = _overview_data_limits(raw, pivot, errors, value_col)
    span = max(float(y_max - y_min), abs(float(y_max)), 1.0)
    step = span * 0.08
    top_needed = y_max
    levels_by_metric: dict[str, int] = {}
    for _, row in comps.sort_values("p_adj").iterrows():
        metric = str(row.get("metric_base", "") or "")
        group_a = str(row.get("group_a", "") or "")
        group_b = str(row.get("group_b", "") or "")
        if metric not in metric_to_idx or group_a not in group_to_idx or group_b not in group_to_idx:
            continue
        level = levels_by_metric.get(metric, 0)
        if level >= 3:
            continue
        levels_by_metric[metric] = level + 1
        metric_idx = metric_to_idx[metric]
        x_center = float(metric_idx)
        x1 = x_center + (group_to_idx[group_a] - (len(groups) - 1) / 2.0) * width
        x2 = x_center + (group_to_idx[group_b] - (len(groups) - 1) / 2.0) * width
        if x2 < x1:
            x1, x2 = x2, x1
        y = y_max + step * (level + 1)
        h = step * 0.22
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#111827", linewidth=0.8, clip_on=False)
        ax.text(
            (x1 + x2) / 2.0,
            y + h,
            _p_label(float(row.get("p_adj", np.nan))),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#111827",
            clip_on=False,
        )
        top_needed = max(top_needed, y + h + step * 0.4)
    if top_needed > y_max:
        ax.set_ylim(top=top_needed)


def _overview_data_limits(raw: pd.DataFrame, pivot: pd.DataFrame, errors: pd.DataFrame, value_col: str) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    if raw is not None and not raw.empty and value_col in raw.columns:
        vals.append(pd.to_numeric(raw[value_col], errors="coerce").to_numpy(dtype=np.float64))
    if pivot is not None and not pivot.empty:
        vals.append(pivot.to_numpy(dtype=np.float64).ravel())
    if pivot is not None and errors is not None and not pivot.empty and not errors.empty:
        vals.append((pivot.to_numpy(dtype=np.float64) + errors.fillna(0).to_numpy(dtype=np.float64)).ravel())
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in vals if arr.size]) if vals else np.array([], dtype=np.float64)
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def _p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"
