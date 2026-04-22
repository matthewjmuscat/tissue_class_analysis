from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import AutoMinorLocator

from qa.config import QAFigureExportConfig
from qa.notation import (
    contrast_math,
    contrast_metric_math,
    family_display_math,
    metric_math,
    metric_with_family_math,
)
from qa.plot_data import (
    CONTRAST_ORDER,
    FAMILY_GROUP_ORDER,
    HEADLINE_METRIC_COLUMNS,
    SAFETY_DISTANCE_METRIC_COLUMNS,
)


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

MPL_FONT_RC = {
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.unicode_minus": True,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.dash_joinstyle": "round",
}

MPL_FACE_RC = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

GROUP_COLOR_MAP = {
    "Real": "#244b79",
    "Centroid": "#b26c3a",
    "Optimal": "#3f8a82",
}

CONTRAST_COLOR_MAP = {
    "centroid_minus_real": "#b26c3a",
    "optimal_minus_real": "#3f8a82",
    "optimal_minus_centroid": "#77628b",
}

PAIR_LINE_COLOR = "#c7c7c7"
GRID_COLOR = "#b8b8b8"
REFERENCE_LINE_COLOR = "#6f6f6f"
HIGHLIGHT_COLOR = "#8C2F39"
FAMILY_FILL_ALPHA = 0.14
PROFILE_FILL_ALPHA = 0.24
DANGER_ZONE_COLOR = "#e5989b"
PROVISIONAL_SAFETY_MARGIN_MM = 5.0
ANNOT_BBOX = dict(
    facecolor="white",
    edgecolor="black",
    alpha=1.0,
    linewidth=0.7,
    boxstyle="round,pad=0.25",
)

ALL_12_ZONE_ORDER = [
    "LP-Apex",
    "LP-Mid",
    "LP-Base",
    "LA-Apex",
    "LA-Mid",
    "LA-Base",
    "RP-Apex",
    "RP-Mid",
    "RP-Base",
    "RA-Apex",
    "RA-Mid",
    "RA-Base",
]


@contextmanager
def _font_rc(export_config: QAFigureExportConfig):
    with mpl.rc_context(
        MPL_FONT_RC
        | MPL_FACE_RC
        | {
            "axes.labelsize": export_config.axes_label_fontsize,
            "xtick.labelsize": export_config.tick_label_fontsize,
            "ytick.labelsize": export_config.tick_label_fontsize,
            "legend.fontsize": export_config.legend_fontsize,
            "axes.titlesize": export_config.title_fontsize,
        }
    ):
        sns.set_theme(style="white", rc=MPL_FONT_RC | MPL_FACE_RC)
        yield


def _save_figure_multi(
    fig: mpl.figure.Figure,
    save_dir: str | Path,
    file_stem: str,
    export_config: QAFigureExportConfig,
) -> list[Path]:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in export_config.save_formats:
        path = out_dir / f"{file_stem}.{str(fmt).lstrip('.')}"
        fig.savefig(path, bbox_inches="tight", dpi=export_config.dpi)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def _add_panel_label(
    ax,
    label: str,
    export_config: QAFigureExportConfig,
    *,
    x: float = -0.10,
    y: float = 1.095,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=export_config.panel_label_fontsize - 1,
        fontweight="bold",
        clip_on=False,
    )


def _add_biopsy_heading(
    ax,
    label: str,
    export_config: QAFigureExportConfig,
    *,
    x: float = 0.00,
    y: float = 1.025,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=export_config.title_fontsize + 3,
        clip_on=False,
    )


def _add_outside_panel_box(
    fig: mpl.figure.Figure,
    ax,
    text: str | None,
    export_config: QAFigureExportConfig,
    *,
    x_align: str = "right",
    x_pad: float = 0.0,
    y_pad: float = 0.020,
) -> None:
    if not text:
        return
    pos = ax.get_position()
    x = pos.x1 + x_pad if x_align == "left" else pos.x1 - x_pad
    fig.text(
        x,
        pos.y1 + y_pad,
        text,
        transform=fig.transFigure,
        ha=x_align,
        va="bottom",
        fontsize=export_config.annotation_fontsize - 2,
        bbox=ANNOT_BBOX,
    )


def _add_inside_panel_box(
    ax,
    text: str | None,
    export_config: QAFigureExportConfig,
    *,
    x: float = 0.97,
    y: float = 0.03,
    ha: str = "right",
    va: str = "bottom",
) -> None:
    if not text:
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=export_config.annotation_fontsize - 1,
        bbox=ANNOT_BBOX,
        zorder=6,
    )


def _style_axes(
    ax,
    export_config: QAFigureExportConfig,
    *,
    y_minor_ticks: int = 2,
    x_minor_ticks: int | None = None,
) -> None:
    ax.grid(visible=True, which="major", linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_ticks))
    if x_minor_ticks is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_ticks))
    ax.tick_params(
        axis="both",
        which="major",
        length=5,
        width=0.9,
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=3,
        width=0.6,
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(axis="y", which="both", labelleft=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_delta_line(summary_row: pd.Series, short_label: str) -> str:
    return (
        f"{short_label}: {summary_row['observed_mean_delta']:+.2f} "
        f"[{summary_row['bootstrap_ci_lower']:.2f}, {summary_row['bootstrap_ci_upper']:.2f}] "
        f"{summary_row['significance_label']}"
    )


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return "p < 0.01"
    return f"p = {p_value:.2f}"


def _draw_significance_bracket(
    ax,
    x0: float,
    x1: float,
    y: float,
    label: str,
    *,
    line_height: float,
    label_pad: float,
    fontsize: int,
) -> None:
    ax.plot(
        [x0, x0, x1, x1],
        [y, y + line_height, y + line_height, y],
        color="black",
        linewidth=1.0,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        0.5 * (x0 + x1),
        y + line_height + label_pad,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
        zorder=6,
    )


def _spread_positions(n_points: int, center: float, half_width: float = 0.12) -> np.ndarray:
    if n_points <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - half_width, center + half_width, n_points)


def _jitter_positions(
    n_points: int,
    center: float,
    *,
    half_width: float = 0.12,
    seed: int = 0,
) -> np.ndarray:
    if n_points <= 0:
        return np.array([], dtype=float)
    if n_points <= 1:
        return np.array([center], dtype=float)
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-half_width, half_width, size=n_points)


def _rank_continuous_features(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    feature_cols: Sequence[str],
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for feature_col in feature_cols:
        if feature_col not in df.columns:
            continue
        sub = df[[feature_col, outcome_col]].dropna().copy()
        if len(sub) < 4:
            continue
        rho = float(sub[feature_col].corr(sub[outcome_col], method="spearman"))
        ranked.append((feature_col, abs(rho)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _rank_categorical_features(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    feature_cols: Sequence[str],
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for feature_col in feature_cols:
        if feature_col not in df.columns:
            continue
        grouped = (
            df[[feature_col, outcome_col]]
            .dropna()
            .groupby(feature_col, sort=True)[outcome_col]
            .median()
        )
        if len(grouped) < 2:
            continue
        spread = float(grouped.max() - grouped.min())
        ranked.append((feature_col, spread))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _format_regression_box(
    x: np.ndarray,
    y: np.ndarray,
    *,
    inv_units: str | None = None,
) -> str:
    fit = stats.linregress(x, y)
    slope_units = "" if not inv_units else f" {inv_units}"
    p_text = _format_p_value(float(fit.pvalue))
    rho_s = pd.Series(x).corr(pd.Series(y), method="spearman")
    return "\n".join(
        [
            rf"OLS slope = {fit.slope:+.3f}{slope_units}",
            rf"$R^2 = {fit.rvalue ** 2:.2f}$, {p_text}",
            rf"$\rho_s = {rho_s:+.2f}$",
        ]
    )


def _choose_annotation_offset(
    ax,
    x: float,
    y: float,
    *,
    all_points: np.ndarray,
    used_label_positions: list[np.ndarray],
    forbidden_axes_regions: Sequence[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> tuple[int, int]:
    candidate_offsets = [
        (16, 16),
        (16, -16),
        (-16, 16),
        (-16, -16),
        (24, 0),
        (-24, 0),
        (0, 24),
        (0, -24),
        (30, 12),
        (-30, 12),
        (30, -12),
        (-30, -12),
    ]
    anchor_disp = ax.transData.transform((x, y))
    point_disp = ax.transData.transform(all_points)

    best_offset = candidate_offsets[0]
    best_score = float("-inf")
    for dx, dy in candidate_offsets:
        label_pos = anchor_disp + np.array([dx, dy], dtype=float)
        point_dists = np.linalg.norm(point_disp - label_pos, axis=1)
        min_point_dist = float(np.min(point_dists)) if len(point_dists) else 0.0
        used_dists = [float(np.linalg.norm(prev - label_pos)) for prev in used_label_positions]
        min_used_dist = min(used_dists) if used_dists else 1000.0
        direction_bonus = 6.0 if dy > 0 else 0.0
        penalty = 0.0
        if forbidden_axes_regions:
            label_axes = ax.transAxes.inverted().transform(label_pos)
            for (x0, x1), (y0, y1) in forbidden_axes_regions:
                if x0 <= label_axes[0] <= x1 and y0 <= label_axes[1] <= y1:
                    penalty += 500.0
        score = min_point_dist + 0.7 * min_used_dist + direction_bonus - penalty
        if score > best_score:
            best_score = score
            best_offset = (dx, dy)
    return best_offset


def _draw_covariance_ellipse(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    edgecolor: str,
    linewidth: float = 1.7,
    linestyle: tuple[int, tuple[int, ...]] | str = (0, (5, 3)),
    alpha: float = 1.0,
) -> None:
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if np.any(vals <= 0):
        return
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    chi2_scale = float(np.sqrt(stats.chi2.ppf(0.95, 2)))
    width, height = 2.0 * chi2_scale * np.sqrt(vals)
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(ellipse)


def _draw_feature_regression_panel(
    fig: mpl.figure.Figure,
    ax,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color: str,
    x_label: str,
    y_label: str,
    panel_label: str,
    export_config: QAFigureExportConfig,
    y_lim: tuple[float, float],
    inv_units: str | None = None,
) -> str | None:
    sub = df[[x_col, y_col]].dropna().copy()
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    summary_text: str | None = None
    ax.scatter(
        x,
        y,
        s=62,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
        zorder=3,
    )
    if len(x) >= 2:
        fit = stats.linregress(x, y)
        x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(
            x_grid,
            fit.intercept + fit.slope * x_grid,
            color=color,
            linewidth=1.8,
            linestyle=(0, (5, 3)),
            zorder=2,
        )
        summary_text = _format_regression_box(x, y, inv_units=inv_units)
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_title("")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    _style_axes(ax, export_config, x_minor_ticks=2)
    ax.tick_params(axis="y", which="both", labelleft=True)
    _add_panel_label(ax, panel_label, export_config)
    return summary_text


def _draw_feature_category_panel(
    ax,
    df: pd.DataFrame,
    *,
    feature_col: str,
    outcome_col: str,
    color: str,
    x_label: str,
    panel_label: str,
    export_config: QAFigureExportConfig,
    y_label: str,
    y_lim: tuple[float, float],
    category_order: Sequence[str] | None = None,
    display_map: dict[str, str] | None = None,
    rotate_xticks: bool = False,
    jitter_seed_base: int = 0,
) -> None:
    sub = df[[feature_col, outcome_col]].dropna().copy()
    if category_order is None:
        category_order = (
            sub[feature_col]
            .astype(str)
            .value_counts()
            .sort_index()
            .index
            .tolist()
        )

    sub[feature_col] = sub[feature_col].astype(str)
    positions = np.arange(len(category_order), dtype=float)
    boxplot_data: list[np.ndarray] = []
    boxplot_positions: list[float] = []
    for idx, category in enumerate(category_order):
        cat_df = sub[sub[feature_col] == str(category)].copy()
        values = cat_df[outcome_col].to_numpy(dtype=float)
        if len(values) > 0:
            boxplot_data.append(values)
            boxplot_positions.append(float(positions[idx]))
        xs = _jitter_positions(
            len(values),
            positions[idx],
            half_width=0.12,
            seed=jitter_seed_base + 97 * (idx + 1) + len(values),
        )
        ax.scatter(
            xs,
            values,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )

    if boxplot_data:
        bp = ax.boxplot(
            boxplot_data,
            positions=boxplot_positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "#4a4a4a", "linewidth": 1.0},
            capprops={"color": "#4a4a4a", "linewidth": 1.0},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.15)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)

    counts = sub[feature_col].value_counts()
    tick_labels = []
    for cat in category_order:
        display = display_map.get(str(cat), str(cat)) if display_map else str(cat)
        tick_labels.append(f"{display}\n(n={int(counts.get(cat, 0))})")
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_title("")
    ax.set_xticks(positions, tick_labels)
    if rotate_xticks:
        ax.tick_params(axis="x", labelrotation=28)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    _style_axes(ax, export_config)
    ax.tick_params(axis="y", which="both", labelleft=True)
    _add_panel_label(ax, panel_label, export_config)


def _shared_marker_legend() -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D([0], [0], color=PAIR_LINE_COLOR, linewidth=1.2),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=GROUP_COLOR_MAP["Real"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=GROUP_COLOR_MAP["Centroid"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=GROUP_COLOR_MAP["Optimal"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        ),
        Line2D([0], [0], color="black", linewidth=2.2),
    ]
    labels = [
        "Matched real-core trajectory",
        family_display_math("Real"),
        family_display_math("Centroid"),
        family_display_math("Optimal"),
        "Group mean",
    ]
    return handles, labels


def _summary_box_text(
    summary_sub: pd.DataFrame | pd.Series,
    *,
    contrast_keys: Sequence[str],
    include_p_values: bool = False,
) -> str:
    lines: list[str] = []
    for contrast_key in contrast_keys:
        if contrast_key not in summary_sub.index:
            continue
        row = summary_sub.loc[contrast_key]
        base = (
            rf"{contrast_math(contrast_key)} = {row['observed_mean_delta']:+.2f} "
            rf"[{row['bootstrap_ci_lower']:.2f}, {row['bootstrap_ci_upper']:.2f}]"
        )
        if include_p_values:
            base += f" {_format_p_value(float(row['bootstrap_p_two_sided']))}"
        else:
            base += f" {row['significance_label']}"
        lines.append(base)
    return "\n".join(lines)


def plot_headline_family_comparison(
    family_comparison_long_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_01_headline_family_comparison",
    metrics: Sequence[str] = HEADLINE_METRIC_COLUMNS,
) -> list[Path]:
    metrics = list(metrics)
    with _font_rc(export_config):
        fig, axes = plt.subplots(1, len(metrics), figsize=(14.2, 6.0), dpi=export_config.dpi)
        if len(metrics) == 1:
            axes = [axes]

        handles, labels = _shared_marker_legend()
        outside_box_specs: list[tuple[object, str]] = []

        for panel_idx, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
            sub = family_comparison_long_df[family_comparison_long_df["metric"] == metric].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            metric_label = metric_math(metric)
            y_values = sub["value"].to_numpy(dtype=float)
            value_min = float(np.nanmin(y_values))
            value_max = float(np.nanmax(y_values))
            span = max(value_max - value_min, 0.12)

            for _, obs_df in sub.groupby("Observation ID", sort=False):
                obs_df = obs_df.sort_values("group_order")
                x_vals = obs_df["group_order"].to_numpy(dtype=float) + obs_df["pair_offset"].to_numpy(dtype=float)
                ax.plot(
                    x_vals,
                    obs_df["value"].to_numpy(dtype=float),
                    color=PAIR_LINE_COLOR,
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=1,
                )

            for group_order, group_key in enumerate(FAMILY_GROUP_ORDER):
                group_df = sub[sub["group_key"] == group_key]
                x_vals = (
                    group_df["group_order"].to_numpy(dtype=float)
                    + group_df["pair_offset"].to_numpy(dtype=float)
                )
                ax.scatter(
                    x_vals,
                    group_df["value"].to_numpy(dtype=float),
                    s=36,
                    color=GROUP_COLOR_MAP[group_key],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                mean_val = float(group_df["value"].mean())
                ax.plot(
                    [group_order - 0.17, group_order + 0.17],
                    [mean_val, mean_val],
                    color="black",
                    linewidth=2.4,
                    solid_capstyle="round",
                    zorder=4,
                )

            summary_sub = bootstrap_summary_df[bootstrap_summary_df["metric"] == metric].copy()
            summary_sub = summary_sub.set_index("contrast_key")
            bracket_height = 0.025 * span
            bracket_pad = 0.02 * span
            bracket_y0 = value_max + 0.10 * span
            bracket_specs = [
                ("centroid_minus_real", 0.0, 1.0, bracket_y0),
                ("optimal_minus_centroid", 1.0, 2.0, bracket_y0 + 0.10 * span),
                ("optimal_minus_real", 0.0, 2.0, bracket_y0 + 0.20 * span),
            ]
            for contrast_key, x0, x1, y in bracket_specs:
                if contrast_key not in summary_sub.index:
                    continue
                _draw_significance_bracket(
                    ax,
                    x0,
                    x1,
                    y,
                    str(summary_sub.at[contrast_key, "significance_label"]),
                    line_height=bracket_height,
                    label_pad=bracket_pad,
                    fontsize=export_config.annotation_fontsize + 1,
                )

            ax.set_title("")
            ax.set_ylabel(metric_label)
            ax.set_xticks(
                [0, 1, 2],
                [
                    "Real\n$(R)$",
                    "Centroid\n$(C)$",
                    "Optimal\n$(O)$",
                ],
            )
            ax.set_xlim(-0.35, 2.35)
            ax.set_ylim(
                min(0.0, value_min - 0.05 * span),
                value_max + 0.38 * span,
            )
            _style_axes(ax, export_config)
            _add_panel_label(ax, chr(ord("A") + panel_idx), export_config)
            outside_box_specs.append(
                (
                    ax,
                    _summary_box_text(
                        summary_sub,
                        contrast_keys=[
                            "centroid_minus_real",
                            "optimal_minus_real",
                            "optimal_minus_centroid",
                        ],
                        include_p_values=False,
                    ),
                )
            )

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(labels),
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.30)
        for ax, text in outside_box_specs:
            _add_outside_panel_box(fig, ax, text, export_config, y_pad=0.022)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def plot_headline_headroom(
    headroom_long_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_02_headline_headroom",
    metrics: Sequence[str] = HEADLINE_METRIC_COLUMNS,
) -> list[Path]:
    metrics = list(metrics)
    contrast_positions = {contrast: idx for idx, contrast in enumerate(CONTRAST_ORDER)}

    with _font_rc(export_config):
        fig, axes = plt.subplots(1, len(metrics), figsize=(14.2, 6.0), dpi=export_config.dpi)
        if len(metrics) == 1:
            axes = [axes]
        outside_box_specs: list[tuple[object, str]] = []

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=CONTRAST_COLOR_MAP["centroid_minus_real"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=CONTRAST_COLOR_MAP["optimal_minus_real"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=CONTRAST_COLOR_MAP["optimal_minus_centroid"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markersize=6,
                markerfacecolor="black",
                markeredgecolor="black",
            ),
        ]
        legend_labels = [
            contrast_math("centroid_minus_real"),
            contrast_math("optimal_minus_real"),
            contrast_math("optimal_minus_centroid"),
            "Mean and 95% patient-cluster bootstrap CI",
        ]

        for panel_idx, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
            sub = headroom_long_df[headroom_long_df["metric"] == metric].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            metric_label = metric_math(metric)
            summary_sub = bootstrap_summary_df[bootstrap_summary_df["metric"] == metric].copy()
            summary_sub = summary_sub.set_index("contrast_key")

            delta_vals = sub["delta_value"].to_numpy(dtype=float)
            delta_min = float(np.nanmin(delta_vals))
            delta_max = float(np.nanmax(delta_vals))
            span = max(delta_max - delta_min, 0.06)
            boxplot_data = []
            for contrast_key in CONTRAST_ORDER:
                contrast_df = sub[sub["contrast_key"] == contrast_key].sort_values(
                    ["Base patient ID", "Relative DIL index", "Observation ID"]
                )
                values = contrast_df["delta_value"].to_numpy(dtype=float)
                boxplot_data.append(values)
                xs = _spread_positions(len(values), contrast_positions[contrast_key], half_width=0.12)
                ax.scatter(
                    xs,
                    values,
                    s=34,
                    color=CONTRAST_COLOR_MAP[contrast_key],
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.95,
                    zorder=3,
                )

            bp = ax.boxplot(
                boxplot_data,
                positions=np.arange(len(CONTRAST_ORDER), dtype=float),
                widths=0.48,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
                whiskerprops={"color": "#4a4a4a", "linewidth": 1.0},
                capprops={"color": "#4a4a4a", "linewidth": 1.0},
            )
            for patch, contrast_key in zip(bp["boxes"], CONTRAST_ORDER, strict=False):
                patch.set_facecolor(CONTRAST_COLOR_MAP[contrast_key])
                patch.set_alpha(0.18)
                patch.set_edgecolor(CONTRAST_COLOR_MAP[contrast_key])
                patch.set_linewidth(1.2)

            ci_marker_offset = 0.28
            for contrast_key in CONTRAST_ORDER:
                if contrast_key not in summary_sub.index:
                    continue
                row = summary_sub.loc[contrast_key]
                center = contrast_positions[contrast_key]
                ci_center = center + ci_marker_offset
                mean_val = float(row["observed_mean_delta"])
                yerr_lower = mean_val - float(row["bootstrap_ci_lower"])
                yerr_upper = float(row["bootstrap_ci_upper"]) - mean_val
                ax.errorbar(
                    ci_center,
                    mean_val,
                    yerr=np.array([[yerr_lower], [yerr_upper]]),
                    fmt="D",
                    markersize=5.5,
                    color="black",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=3.0,
                    zorder=4,
                )
                ax.plot(
                    [center + 0.24, ci_center - 0.03],
                    [mean_val, mean_val],
                    color="black",
                    linewidth=0.8,
                    zorder=4,
                )
                ax.text(
                    center,
                    delta_max + 0.09 * span,
                    str(row["significance_label"]),
                    ha="center",
                    va="bottom",
                    fontsize=export_config.annotation_fontsize + 1,
                    color="black",
                    zorder=5,
                )

            ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
            ax.set_title("")
            ax.set_ylabel(rf"Matched-family delta in {metric_label}")
            ax.set_xticks(
                [0, 1, 2],
                [
                    "$\\Delta^{(C-R)}$",
                    "$\\Delta^{(O-R)}$",
                    "$\\Delta^{(O-C)}$",
                ],
            )
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(
                min(0.0, delta_min - 0.12 * span),
                delta_max + 0.18 * span,
            )
            _style_axes(ax, export_config)
            _add_panel_label(ax, chr(ord("A") + panel_idx), export_config)
            outside_box_specs.append(
                (
                    ax,
                    _summary_box_text(
                        summary_sub,
                        contrast_keys=[
                            "centroid_minus_real",
                            "optimal_minus_real",
                            "optimal_minus_centroid",
                        ],
                        include_p_values=True,
                    ),
                )
            )

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=4,
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.72, bottom=0.20, wspace=0.30)
        for ax, text in outside_box_specs:
            _add_outside_panel_box(fig, ax, text, export_config, y_pad=0.022)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def plot_reference_disagreement(
    reference_disagreement_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_03_centroid_vs_optimal_disagreement",
    metrics: Sequence[str] = HEADLINE_METRIC_COLUMNS,
    top_n_labels: int = 4,
) -> list[Path]:
    metrics = list(metrics)
    with _font_rc(export_config):
        fig, axes = plt.subplots(1, len(metrics), figsize=(12.4, 5.8), dpi=export_config.dpi)
        if len(metrics) == 1:
            axes = [axes]

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=GROUP_COLOR_MAP["Real"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=HIGHLIGHT_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.8,
            ),
            Line2D([0], [0], color=REFERENCE_LINE_COLOR, linewidth=1.1, linestyle=(0, (4, 3))),
        ]
        legend_labels = [
            "All lesion families",
            rf"Top {top_n_labels} $|\Delta^{{(O-C)}}|$ families",
            "Identity line",
        ]

        for panel_idx, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
            sub = reference_disagreement_df[reference_disagreement_df["metric"] == metric].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            label_col = (
                "Family figure label"
                if "Family figure label" in sub.columns
                else "Family display label"
            )

            metric_label = metric_math(metric)
            x = sub["centroid_value"].to_numpy(dtype=float)
            y = sub["optimal_value"].to_numpy(dtype=float)
            low = max(0.0, min(np.nanmin(x), np.nanmin(y)) - 0.04)
            high = min(1.02, max(np.nanmax(x), np.nanmax(y)) + 0.04)

            ax.plot([low, high], [low, high], color=REFERENCE_LINE_COLOR, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
            ax.scatter(
                x,
                y,
                s=55,
                color=GROUP_COLOR_MAP["Real"],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
                zorder=2,
            )

            if top_n_labels > 0:
                highlight_df = sub.nlargest(top_n_labels, "abs_delta_value").copy()
                ax.scatter(
                    highlight_df["centroid_value"],
                    highlight_df["optimal_value"],
                    s=75,
                    color=HIGHLIGHT_COLOR,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
                all_points = sub[["centroid_value", "optimal_value"]].to_numpy(dtype=float)
                used_label_positions: list[np.ndarray] = []
                forbidden_regions = [((0.02, 0.72), (0.66, 0.99))]
                for _, row in highlight_df.iterrows():
                    dx, dy = _choose_annotation_offset(
                        ax,
                        float(row["centroid_value"]),
                        float(row["optimal_value"]),
                        all_points=all_points,
                        used_label_positions=used_label_positions,
                        forbidden_axes_regions=forbidden_regions,
                    )
                    txt = ax.annotate(
                        str(row[label_col]),
                        xy=(row["centroid_value"], row["optimal_value"]),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha="left" if dx >= 0 else "right",
                        va="bottom" if dy >= 0 else "top",
                        fontsize=export_config.annotation_fontsize,
                        color="black",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
                        arrowprops=dict(
                            arrowstyle="-",
                            color="#4a4a4a",
                            linewidth=0.7,
                            shrinkA=3,
                            shrinkB=3,
                        ),
                        zorder=4,
                    )
                    txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
                    used_label_positions.append(
                        ax.transData.transform((float(row["centroid_value"]), float(row["optimal_value"])))
                        + np.array([dx, dy], dtype=float)
                    )

            mean_delta = float(sub["delta_value"].mean())
            median_delta = float(sub["delta_value"].median())
            max_abs_idx = sub["abs_delta_value"].idxmax()
            max_abs_row = sub.loc[max_abs_idx]
            stats_lines = [
                rf"$\overline{{\Delta}}^{{(O-C)}}$ = {mean_delta:+.3f}",
                rf"$\mathrm{{med}}(\Delta^{{(O-C)}})$ = {median_delta:+.3f}",
                rf"$\max |\Delta^{{(O-C)}}|$ = {max_abs_row['abs_delta_value']:.3f}",
                f"Top family: {max_abs_row[label_col]}",
            ]

            ax.text(
                0.03,
                0.97,
                "\n".join(stats_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=export_config.annotation_fontsize,
                bbox=ANNOT_BBOX,
            )

            ax.set_title(metric_label, pad=24)
            ax.set_xlabel(metric_with_family_math(metric, "Centroid"))
            ax.set_ylabel(metric_with_family_math(metric, "Optimal"))
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            _style_axes(ax, export_config)
            _add_panel_label(ax, chr(ord("A") + panel_idx), export_config)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.24)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def _safety_min_box_text(metric: str, sub: pd.DataFrame) -> str:
    minima_by_family: list[str] = []
    for family_name in FAMILY_GROUP_ORDER:
        family_sub = sub[sub["group_key"] == family_name]
        if family_sub.empty:
            continue
        minima_by_family.append(f"{family_name[0]}: {float(family_sub['value'].min()):.2f}")
    return "Minima (mm) " + ", ".join(minima_by_family)


def plot_safety_distance_family_comparison(
    safety_distance_long_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_04_safety_distance_family_comparison",
    metrics: Sequence[str] = SAFETY_DISTANCE_METRIC_COLUMNS,
) -> list[Path]:
    metrics = list(metrics)
    with _font_rc(export_config):
        fig, axes = plt.subplots(1, len(metrics), figsize=(14.2, 5.8), dpi=export_config.dpi)
        if len(metrics) == 1:
            axes = [axes]

        handles, labels = _shared_marker_legend()
        handles = list(handles) + [
            Patch(
                facecolor=DANGER_ZONE_COLOR,
                edgecolor=DANGER_ZONE_COLOR,
                alpha=0.16,
            )
        ]
        labels = list(labels) + [rf"Provisional {PROVISIONAL_SAFETY_MARGIN_MM:.0f} mm caution zone"]
        outside_box_specs: list[tuple[object, str]] = []

        for panel_idx, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
            sub = safety_distance_long_df[safety_distance_long_df["metric"] == metric].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            metric_label = metric_math(metric)
            y_values = sub["value"].to_numpy(dtype=float)
            value_min = float(np.nanmin(y_values))
            value_max = float(np.nanmax(y_values))
            span = max(value_max - value_min, 1.0)
            ax.axhspan(
                0.0,
                PROVISIONAL_SAFETY_MARGIN_MM,
                color=DANGER_ZONE_COLOR,
                alpha=0.16,
                zorder=0,
            )
            ax.axhline(
                PROVISIONAL_SAFETY_MARGIN_MM,
                color=DANGER_ZONE_COLOR,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=0.5,
            )

            for _, obs_df in sub.groupby("Observation ID", sort=False):
                obs_df = obs_df.sort_values("group_order")
                x_vals = obs_df["group_order"].to_numpy(dtype=float) + obs_df["pair_offset"].to_numpy(dtype=float)
                ax.plot(
                    x_vals,
                    obs_df["value"].to_numpy(dtype=float),
                    color=PAIR_LINE_COLOR,
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=1,
                )

            for group_order, group_key in enumerate(FAMILY_GROUP_ORDER):
                group_df = sub[sub["group_key"] == group_key]
                x_vals = (
                    group_df["group_order"].to_numpy(dtype=float)
                    + group_df["pair_offset"].to_numpy(dtype=float)
                )
                ax.scatter(
                    x_vals,
                    group_df["value"].to_numpy(dtype=float),
                    s=36,
                    color=GROUP_COLOR_MAP[group_key],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                mean_val = float(group_df["value"].mean())
                ax.plot(
                    [group_order - 0.17, group_order + 0.17],
                    [mean_val, mean_val],
                    color="black",
                    linewidth=2.4,
                    solid_capstyle="round",
                    zorder=4,
                )

            summary_sub = bootstrap_summary_df[bootstrap_summary_df["metric"] == metric].copy()
            summary_sub = summary_sub.set_index("contrast_key")
            bracket_height = 0.06 * span
            bracket_pad = 0.03 * span
            bracket_y0 = value_max + 0.10 * span
            bracket_specs = [
                ("centroid_minus_real", 0.0, 1.0, bracket_y0),
                ("optimal_minus_centroid", 1.0, 2.0, bracket_y0 + 0.18 * span),
                ("optimal_minus_real", 0.0, 2.0, bracket_y0 + 0.36 * span),
            ]
            for contrast_key, x0, x1, y in bracket_specs:
                if contrast_key not in summary_sub.index:
                    continue
                _draw_significance_bracket(
                    ax,
                    x0,
                    x1,
                    y,
                    str(summary_sub.at[contrast_key, "significance_label"]),
                    line_height=bracket_height,
                    label_pad=bracket_pad,
                    fontsize=export_config.annotation_fontsize + 1,
                )

            ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
            ax.set_title("")
            ax.set_ylabel(rf"Mean nearest-neighbour distance {metric_label} (mm)")
            ax.set_xticks(
                [0, 1, 2],
                [
                    "Real\n$(R)$",
                    "Centroid\n$(C)$",
                    "Optimal\n$(O)$",
                ],
            )
            ax.set_xlim(-0.35, 2.35)
            ax.set_ylim(
                min(0.0, value_min - 0.12 * span),
                value_max + 0.62 * span,
            )
            _style_axes(ax, export_config)
            _add_panel_label(ax, chr(ord("A") + panel_idx), export_config)
            outside_box_specs.append((ax, _safety_min_box_text(metric, sub)))

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=3,
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.72, bottom=0.18, wspace=0.30)
        for ax, text in outside_box_specs:
            _add_outside_panel_box(fig, ax, text, export_config, y_pad=0.022)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def _profile_step_edges_and_values(
    fam_df: pd.DataFrame,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = fam_df.sort_values("Voxel index")
    edges = np.concatenate(
        [
            ordered["Voxel begin (Z)"].to_numpy(dtype=float),
            np.array([float(ordered["Voxel end (Z)"].iloc[-1])], dtype=float),
        ]
    )
    values = ordered[value_col].to_numpy(dtype=float)
    return edges, values


def _draw_profile_family_curve(
    ax,
    fam_df: pd.DataFrame,
    family_name: str,
    style: dict[str, object],
    *,
    step_mode: bool,
) -> None:
    fam_df = fam_df.sort_values("Voxel index")
    if fam_df.empty:
        return

    if step_mode:
        edges, y = _profile_step_edges_and_values(fam_df, "Binomial estimator")
        _, y_low = _profile_step_edges_and_values(fam_df, "CI lower vals")
        _, y_high = _profile_step_edges_and_values(fam_df, "CI upper vals")
        x_repeat = np.repeat(edges, 2)[1:-1]
        y_repeat = np.repeat(y, 2)
        y_low_repeat = np.repeat(y_low, 2)
        y_high_repeat = np.repeat(y_high, 2)
        ax.fill_between(
            x_repeat,
            y_low_repeat,
            y_high_repeat,
            color=style["color"],
            alpha=PROFILE_FILL_ALPHA,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_repeat,
            y_repeat,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            zorder=3,
        )
        return

    x = fam_df["Voxel mid Z"].to_numpy(dtype=float)
    y = fam_df["Binomial estimator"].to_numpy(dtype=float)
    y_low = fam_df["CI lower vals"].to_numpy(dtype=float)
    y_high = fam_df["CI upper vals"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        y_low,
        y_high,
        color=style["color"],
        alpha=PROFILE_FILL_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        y,
        color=style["color"],
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        zorder=3,
    )


def _profile_box_text(case_df: pd.DataFrame) -> str:
    case_row = case_df.iloc[0]
    return "\n".join(
        [
            rf"{contrast_metric_math('DIL Global Mean BE', 'centroid_minus_real')} = "
            rf"{float(case_row['Panel delta_centroid_minus_real__DIL Global Mean BE']):+.2f}",
            rf"{contrast_metric_math('DIL Global Mean BE', 'optimal_minus_real')} = "
            rf"{float(case_row['Panel delta_optimal_minus_real__DIL Global Mean BE']):+.2f}",
            rf"{contrast_metric_math('DIL Global Mean BE', 'optimal_minus_centroid')} = "
            rf"{float(case_row['Panel delta_optimal_minus_centroid__DIL Global Mean BE']):+.2f}",
        ]
    )


def _plot_selected_dil_profiles_impl(
    selected_profile_long_df: pd.DataFrame,
    selected_profile_cases_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str,
    step_mode: bool,
) -> list[Path]:
    if selected_profile_long_df.empty or selected_profile_cases_df.empty:
        return []

    family_line_specs = {
        "Real": {"color": GROUP_COLOR_MAP["Real"], "linestyle": "-", "linewidth": 2.2},
        "Centroid": {"color": GROUP_COLOR_MAP["Centroid"], "linestyle": "-", "linewidth": 2.0},
        "Optimal": {"color": GROUP_COLOR_MAP["Optimal"], "linestyle": "-", "linewidth": 2.0},
    }

    with _font_rc(export_config):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), dpi=export_config.dpi, sharey=True)
        axes_flat = list(np.ravel(axes))
        legend_handles = [
            Line2D([0], [0], color=GROUP_COLOR_MAP["Real"], linewidth=2.2),
            Line2D([0], [0], color=GROUP_COLOR_MAP["Centroid"], linewidth=2.0),
            Line2D([0], [0], color=GROUP_COLOR_MAP["Optimal"], linewidth=2.0),
            Patch(facecolor="#6f6f6f", edgecolor="none", alpha=FAMILY_FILL_ALPHA),
        ]
        legend_labels = [
            family_display_math("Real"),
            family_display_math("Centroid"),
            family_display_math("Optimal"),
            "95% CI",
        ]
        outside_box_specs: list[tuple[object, str]] = []

        case_order = (
            selected_profile_cases_df.sort_values("Selection order")["Biopsy heading"].tolist()
        )
        case_df_map = {
            heading: selected_profile_long_df[
                selected_profile_long_df["Biopsy heading"].astype(str) == str(heading)
            ].copy()
            for heading in case_order
        }

        for panel_idx, (ax, heading) in enumerate(zip(axes_flat, case_order, strict=False)):
            case_df = case_df_map[heading]
            if case_df.empty:
                ax.set_visible(False)
                continue

            x_min = float(case_df["Voxel begin (Z)"].min())
            x_max = float(case_df["Voxel end (Z)"].max())
            for family_name in FAMILY_GROUP_ORDER:
                fam_df = case_df[case_df["Family group"] == family_name].sort_values("Voxel index")
                if fam_df.empty:
                    continue
                style = family_line_specs[family_name]
                _draw_profile_family_curve(ax, fam_df, family_name, style, step_mode=step_mode)

            ax.set_xlim(x_min - 0.4, x_max + 0.4)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(str(case_df["Selection reason"].iloc[0]), pad=24)
            ax.set_xlabel(r"Axial position along biopsy $z$ (mm)")
            if panel_idx % 2 == 0:
                ax.set_ylabel(r"Voxelwise DIL probability $\mathcal{P}_{D}(z)$")
            _style_axes(ax, export_config, x_minor_ticks=2)
            _add_biopsy_heading(ax, heading, export_config)
            outside_box_specs.append((ax, _profile_box_text(case_df)))

        for ax in axes_flat[len(case_order):]:
            ax.set_visible(False)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=4,
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fontsize=export_config.legend_fontsize,
        )
        fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.52, wspace=0.18)
        for ax, text in outside_box_specs:
            _add_outside_panel_box(fig, ax, text, export_config, y_pad=0.010)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def plot_selected_dil_profiles(
    selected_profile_long_df: pd.DataFrame,
    selected_profile_cases_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_05_selected_dil_profiles",
) -> list[Path]:
    return _plot_selected_dil_profiles_impl(
        selected_profile_long_df,
        selected_profile_cases_df,
        save_dir,
        export_config=export_config,
        file_stem=file_stem,
        step_mode=False,
    )


def plot_selected_dil_profiles_step(
    selected_profile_long_df: pd.DataFrame,
    selected_profile_cases_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_06_selected_dil_profiles_step",
) -> list[Path]:
    return _plot_selected_dil_profiles_impl(
        selected_profile_long_df,
        selected_profile_cases_df,
        save_dir,
        export_config=export_config,
        file_stem=file_stem,
        step_mode=True,
    )


def _category_tick_labels(sub: pd.DataFrame, category_col: str, category_order: Sequence[str]) -> list[str]:
    counts = sub[category_col].value_counts()
    labels: list[str] = []
    for category in category_order:
        n = int(counts.get(category, 0))
        if category in {"Posterior", "Anterior"}:
            display = category
        elif category in {"Apex", "Mid", "Base"}:
            display = category
        else:
            display = category
        labels.append(f"{display}\n(n={n})")
    return labels


CONTINUOUS_FEATURE_SPECS = {
    "DIL Volume": {
        "x_label": r"DIL volume (mm$^3$)",
        "inv_units": r"mm$^{-3}$",
    },
    "DIL Maximum 3D diameter": {
        "x_label": r"DIL maximum 3D diameter (mm)",
        "inv_units": r"mm$^{-1}$",
    },
    "Prostate Volume": {
        "x_label": r"Prostate volume (mm$^3$)",
        "inv_units": r"mm$^{-3}$",
    },
    "DIL DIL centroid (Y, prostate frame)": {
        "x_label": r"DIL AP centroid coordinate $y$"
        "\n"
        r"[Posterior(+), Anterior(-)] (mm)",
        "inv_units": r"mm$^{-1}$",
    },
    "DIL DIL centroid distance (prostate frame)": {
        "x_label": r"DIL centroid distance from prostate-frame origin (mm)",
        "inv_units": r"mm$^{-1}$",
    },
    "DIL DIL centroid (Z, prostate frame)": {
        "x_label": r"DIL SI centroid coordinate $z$ (mm)",
        "inv_units": r"mm$^{-1}$",
    },
}

COARSE_CATEGORY_SPECS = {
    "DIL DIL prostate sextant (LR)": {
        "x_label": "Left-right lesion location",
        "order": ["Left", "Right"],
        "display_map": {"Left": "Left", "Right": "Right"},
        "rotate_xticks": False,
    },
    "DIL DIL prostate sextant (AP)": {
        "x_label": "Anterior-posterior lesion location",
        "order": ["Posterior", "Anterior"],
        "display_map": {"Posterior": "Post", "Anterior": "Ant"},
        "rotate_xticks": False,
    },
    "DIL DIL prostate sextant (SI)": {
        "x_label": "Superior-inferior lesion location",
        "order": ["Apex (Inferior)", "Mid", "Base (Superior)"],
        "display_map": {
            "Apex (Inferior)": "Apex",
            "Mid": "Mid",
            "Base (Superior)": "Base",
        },
        "rotate_xticks": False,
    },
    "DIL double sextant zone": {
        "x_label": "Double-sextant lesion location",
        "order": ["LP", "LA", "RP", "RA"],
        "display_map": {"LP": "LP", "LA": "LA", "RP": "RP", "RA": "RA"},
        "rotate_xticks": False,
    },
}


def _difficulty_y_limits(
    df: pd.DataFrame,
    y_col: str,
    *,
    low_floor: float,
    high_floor: float,
) -> tuple[float, float]:
    y_vals = df[y_col].dropna().to_numpy(dtype=float)
    y_min = float(np.nanmin(y_vals))
    y_max = float(np.nanmax(y_vals))
    y_span = max(y_max - y_min, 0.08)
    return (
        min(low_floor, y_min - 0.10 * y_span),
        max(high_floor, y_max + 0.12 * y_span),
    )


def _top_continuous_features(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    n_features: int,
) -> list[str]:
    ranked = _rank_continuous_features(
        df,
        outcome_col=outcome_col,
        feature_cols=tuple(CONTINUOUS_FEATURE_SPECS.keys()),
    )
    selected = [feature for feature, _ in ranked[:n_features]]
    if len(selected) < n_features:
        for feature in CONTINUOUS_FEATURE_SPECS:
            if feature not in selected and feature in df.columns:
                selected.append(feature)
            if len(selected) == n_features:
                break
    return selected


def _plot_difficulty_continuous_grid(
    df: pd.DataFrame,
    save_dir: str | Path,
    *,
    y_col: str,
    y_label: str,
    color: str,
    y_lim: tuple[float, float],
    export_config: QAFigureExportConfig,
    file_stem: str,
) -> list[Path]:
    selected_features = _top_continuous_features(df, outcome_col=y_col, n_features=4)
    if not selected_features:
        return []

    with _font_rc(export_config):
        fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2), dpi=export_config.dpi, sharey=True)
        axes_flat = list(np.ravel(axes))
        outside_box_specs: list[tuple[object, str]] = []

        for panel_idx, (ax, feature_col) in enumerate(zip(axes_flat, selected_features, strict=False)):
            feature_spec = CONTINUOUS_FEATURE_SPECS[feature_col]
            summary_text = _draw_feature_regression_panel(
                fig,
                ax,
                df,
                x_col=feature_col,
                y_col=y_col,
                color=color,
                x_label=str(feature_spec["x_label"]),
                y_label=y_label if panel_idx % 2 == 0 else "",
                panel_label=chr(ord("A") + panel_idx),
                export_config=export_config,
                y_lim=y_lim,
                inv_units=str(feature_spec["inv_units"]),
            )
            if summary_text:
                outside_box_specs.append((ax, summary_text))

        for ax in axes_flat[len(selected_features):]:
            ax.set_visible(False)

        fig.subplots_adjust(top=0.89, bottom=0.12, hspace=0.46, wspace=0.24)
        for ax, text in outside_box_specs:
            _add_outside_panel_box(fig, ax, text, export_config, y_pad=0.010)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def _plot_difficulty_categorical_grid(
    df: pd.DataFrame,
    save_dir: str | Path,
    *,
    y_col: str,
    y_label: str,
    color: str,
    y_lim: tuple[float, float],
    export_config: QAFigureExportConfig,
    file_stem: str,
) -> list[Path]:
    with _font_rc(export_config):
        fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2), dpi=export_config.dpi, sharey=True)
        axes_flat = list(np.ravel(axes))

        for panel_idx, (ax, feature_col) in enumerate(
            zip(axes_flat, COARSE_CATEGORY_SPECS.keys(), strict=False)
        ):
            feature_spec = COARSE_CATEGORY_SPECS[feature_col]
            _draw_feature_category_panel(
                ax,
                df,
                feature_col=feature_col,
                outcome_col=y_col,
                color=color,
                x_label=str(feature_spec["x_label"]),
                panel_label=chr(ord("A") + panel_idx),
                export_config=export_config,
                y_label=y_label if panel_idx % 2 == 0 else "",
                y_lim=y_lim,
                category_order=feature_spec["order"],
                display_map=feature_spec["display_map"],
                rotate_xticks=bool(feature_spec["rotate_xticks"]),
                jitter_seed_base=2400 + 101 * panel_idx,
            )

        fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.42, wspace=0.20)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def _plot_difficulty_mixed_summary(
    df: pd.DataFrame,
    save_dir: str | Path,
    *,
    y_col: str,
    y_label: str,
    color: str,
    y_lim: tuple[float, float],
    export_config: QAFigureExportConfig,
    file_stem: str,
) -> list[Path]:
    selected_continuous = _top_continuous_features(df, outcome_col=y_col, n_features=3)
    if not selected_continuous:
        return []

    with _font_rc(export_config):
        fig = plt.figure(figsize=(15.2, 9.2), dpi=export_config.dpi)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], hspace=0.42, wspace=0.30)
        top_axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
        bottom_ax = fig.add_subplot(gs[1, :])
        outside_box_specs: list[tuple[object, str]] = []

        for panel_idx, (ax, feature_col) in enumerate(zip(top_axes, selected_continuous, strict=False)):
            feature_spec = CONTINUOUS_FEATURE_SPECS[feature_col]
            summary_text = _draw_feature_regression_panel(
                fig,
                ax,
                df,
                x_col=feature_col,
                y_col=y_col,
                color=color,
                x_label=str(feature_spec["x_label"]),
                y_label=y_label,
                panel_label=chr(ord("A") + panel_idx),
                export_config=export_config,
                y_lim=y_lim,
                inv_units=str(feature_spec["inv_units"]),
            )
            if summary_text:
                outside_box_specs.append((ax, summary_text))

        _draw_feature_category_panel(
            bottom_ax,
            df,
            feature_col="DIL sextant zone 12",
            outcome_col=y_col,
            color=color,
            x_label="12-zone lesion location",
            panel_label="D",
            export_config=export_config,
            y_label=y_label,
            y_lim=y_lim,
            category_order=ALL_12_ZONE_ORDER,
            rotate_xticks=True,
            jitter_seed_base=1707,
        )

        fig.subplots_adjust(top=0.90, bottom=0.14)
        for ax, text in outside_box_specs:
            _add_outside_panel_box(fig, ax, text, export_config, y_pad=0.010)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)


def plot_optimizer_difficulty_summary(
    optimizer_difficulty_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_07_optimizer_difficulty_summary",
) -> list[Path]:
    if optimizer_difficulty_df.empty:
        return []

    df = optimizer_difficulty_df.copy()
    y_col = "optimizer_gain_mean__DIL Global Mean BE"
    y_label = r"Optimizer increment $\Delta^{(O-C)}\langle \mathcal{P}_{D} \rangle$"
    return _plot_difficulty_mixed_summary(
        df,
        save_dir,
        y_col=y_col,
        y_label=y_label,
        color=CONTRAST_COLOR_MAP["optimal_minus_centroid"],
        y_lim=_difficulty_y_limits(df, y_col, low_floor=-0.14, high_floor=0.10),
        export_config=export_config,
        file_stem=file_stem,
    )


def plot_targeting_difficulty_summary(
    family_difficulty_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_08_targeting_difficulty_summary",
) -> list[Path]:
    if family_difficulty_df.empty:
        return []

    df = family_difficulty_df.copy()
    y_col = "delta_centroid_minus_real_mean__DIL Global Mean BE"
    y_label = r"$\Delta^{(C-R)}\langle \mathcal{P}_{D} \rangle$"
    return _plot_difficulty_mixed_summary(
        df,
        save_dir,
        y_col=y_col,
        y_label=y_label,
        color=CONTRAST_COLOR_MAP["centroid_minus_real"],
        y_lim=_difficulty_y_limits(df, y_col, low_floor=-0.04, high_floor=0.74),
        export_config=export_config,
        file_stem=file_stem,
    )


def plot_optimizer_difficulty_continuous(
    optimizer_difficulty_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_09_optimizer_difficulty_continuous",
) -> list[Path]:
    if optimizer_difficulty_df.empty:
        return []

    df = optimizer_difficulty_df.copy()
    y_col = "optimizer_gain_mean__DIL Global Mean BE"
    return _plot_difficulty_continuous_grid(
        df,
        save_dir,
        y_col=y_col,
        y_label=r"Optimizer increment $\Delta^{(O-C)}\langle \mathcal{P}_{D} \rangle$",
        color=CONTRAST_COLOR_MAP["optimal_minus_centroid"],
        y_lim=_difficulty_y_limits(df, y_col, low_floor=-0.14, high_floor=0.10),
        export_config=export_config,
        file_stem=file_stem,
    )


def plot_targeting_difficulty_continuous(
    family_difficulty_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_10_targeting_difficulty_continuous",
) -> list[Path]:
    if family_difficulty_df.empty:
        return []

    df = family_difficulty_df.copy()
    y_col = "delta_centroid_minus_real_mean__DIL Global Mean BE"
    return _plot_difficulty_continuous_grid(
        df,
        save_dir,
        y_col=y_col,
        y_label=r"$\Delta^{(C-R)}\langle \mathcal{P}_{D} \rangle$",
        color=CONTRAST_COLOR_MAP["centroid_minus_real"],
        y_lim=_difficulty_y_limits(df, y_col, low_floor=-0.04, high_floor=0.74),
        export_config=export_config,
        file_stem=file_stem,
    )


def plot_optimizer_difficulty_categorical(
    optimizer_difficulty_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_11_optimizer_difficulty_categorical",
) -> list[Path]:
    if optimizer_difficulty_df.empty:
        return []

    df = optimizer_difficulty_df.copy()
    y_col = "optimizer_gain_mean__DIL Global Mean BE"
    return _plot_difficulty_categorical_grid(
        df,
        save_dir,
        y_col=y_col,
        y_label=r"Optimizer increment $\Delta^{(O-C)}\langle \mathcal{P}_{D} \rangle$",
        color=CONTRAST_COLOR_MAP["optimal_minus_centroid"],
        y_lim=_difficulty_y_limits(df, y_col, low_floor=-0.14, high_floor=0.10),
        export_config=export_config,
        file_stem=file_stem,
    )


def plot_targeting_difficulty_categorical(
    family_difficulty_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_12_targeting_difficulty_categorical",
) -> list[Path]:
    if family_difficulty_df.empty:
        return []

    df = family_difficulty_df.copy()
    y_col = "delta_centroid_minus_real_mean__DIL Global Mean BE"
    return _plot_difficulty_categorical_grid(
        df,
        save_dir,
        y_col=y_col,
        y_label=r"$\Delta^{(C-R)}\langle \mathcal{P}_{D} \rangle$",
        color=CONTRAST_COLOR_MAP["centroid_minus_real"],
        y_lim=_difficulty_y_limits(df, y_col, low_floor=-0.04, high_floor=0.74),
        export_config=export_config,
        file_stem=file_stem,
    )


def _localization_summary_text(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_symbol: str,
    y_symbol: str,
) -> str:
    return "\n".join(
        [
            rf"$\mu_{{{x_symbol}}} = {np.mean(x):+.2f}$ mm, $\sigma_{{{x_symbol}}} = {np.std(x, ddof=1):.2f}$ mm",
            rf"$\mu_{{{y_symbol}}} = {np.mean(y):+.2f}$ mm, $\sigma_{{{y_symbol}}} = {np.std(y, ddof=1):.2f}$ mm",
            rf"$n = {len(x)}$ real cores",
            "Dashed ellipse: 95% covariance contour",
        ]
    )


def _plot_localization_panel(
    fig: mpl.figure.Figure,
    spec,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    x_symbol: str,
    y_symbol: str,
    plane_label: str,
    panel_label: str,
    export_config: QAFigureExportConfig,
    norm,
    cmap,
) -> tuple[object, str | None, object]:
    gs = spec.subgridspec(
        2,
        2,
        height_ratios=[0.24, 1.0],
        width_ratios=[1.0, 0.24],
        hspace=0.05,
        wspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    c = df["DIL Global Mean BE"].to_numpy(dtype=float)
    scatter = ax_main.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        norm=norm,
        s=74,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.96,
        zorder=3,
    )
    _draw_covariance_ellipse(ax_main, x, y, edgecolor=HIGHLIGHT_COLOR, alpha=0.9)
    ax_main.plot(np.mean(x), np.mean(y), marker="+", color="black", markersize=12, markeredgewidth=1.8, zorder=4)
    ax_main.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax_main.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    max_abs = max(np.max(np.abs(x)), np.max(np.abs(y)))
    max_abs = float(np.ceil(max_abs + 1.0))
    ax_main.set_xlim(-max_abs, max_abs)
    ax_main.set_ylim(-max_abs, max_abs)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    _style_axes(ax_main, export_config, x_minor_ticks=2)
    _add_panel_label(ax_main, panel_label, export_config)
    ax_main.text(
        0.03,
        0.97,
        plane_label,
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=export_config.title_fontsize - 1,
        style="italic",
    )

    bins = np.linspace(-max_abs, max_abs, 15)
    ax_top.hist(x, bins=bins, color="#d4d4d4", edgecolor="white", linewidth=0.7)
    ax_top.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)))
    ax_right.hist(y, bins=bins, orientation="horizontal", color="#d4d4d4", edgecolor="white", linewidth=0.7)
    ax_right.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)))

    ax_top.tick_params(axis="x", which="both", labelbottom=False)
    ax_top.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_right.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", which="both", labelleft=False)
    for side in ["top", "right", "left"]:
        ax_top.spines[side].set_visible(False)
    for side in ["top", "right", "bottom"]:
        ax_right.spines[side].set_visible(False)
    ax_top.grid(visible=False)
    ax_right.grid(visible=False)

    labeled = df[df["Selected biopsy heading"].notna()].copy()
    if not labeled.empty:
        all_points = df[[x_col, y_col]].to_numpy(dtype=float)
        used_label_positions: list[np.ndarray] = []
        for _, row in labeled.iterrows():
            dx, dy = _choose_annotation_offset(
                ax_main,
                float(row[x_col]),
                float(row[y_col]),
                all_points=all_points,
                used_label_positions=used_label_positions,
            )
            txt = ax_main.annotate(
                str(row["Selected biopsy heading"]),
                xy=(float(row[x_col]), float(row[y_col])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=export_config.annotation_fontsize,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#4a4a4a",
                    linewidth=0.7,
                    shrinkA=3,
                    shrinkB=3,
                ),
                zorder=5,
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
            used_label_positions.append(
                ax_main.transData.transform((float(row[x_col]), float(row[y_col])))
                + np.array([dx, dy], dtype=float)
            )

    return ax_main, _localization_summary_text(x, y, x_symbol=x_symbol, y_symbol=y_symbol), scatter


def plot_localization_accuracy_centroids(
    localization_real_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    export_config: QAFigureExportConfig = QAFigureExportConfig(),
    file_stem: str = "Fig_QA_13_localization_accuracy_centroids",
) -> list[Path]:
    if localization_real_df.empty:
        return []

    df = localization_real_df.copy()
    df = df.dropna(
        subset=[
            "Bx (X, DIL centroid frame)",
            "Bx (Y, DIL centroid frame)",
            "Bx (Z, DIL centroid frame)",
            "DIL Global Mean BE",
        ]
    ).copy()
    if df.empty:
        return []

    cmap = mpl.cm.get_cmap("cividis")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    with _font_rc(export_config):
        fig = plt.figure(figsize=(15.6, 7.8), dpi=export_config.dpi)
        outer = fig.add_gridspec(1, 2, wspace=0.24)

        left_ax, left_text, scatter = _plot_localization_panel(
            fig,
            outer[0, 0],
            df,
            x_col="Bx (X, DIL centroid frame)",
            y_col="Bx (Y, DIL centroid frame)",
            x_label="L/R offset from DIL centroid (mm)\nL(+), R(-)",
            y_label="A/P offset from DIL centroid (mm)\nP(+), A(-)",
            x_symbol="x",
            y_symbol="y",
            plane_label="Transverse plane",
            panel_label="A",
            export_config=export_config,
            norm=norm,
            cmap=cmap,
        )
        right_ax, right_text, _ = _plot_localization_panel(
            fig,
            outer[0, 1],
            df,
            x_col="Bx (Z, DIL centroid frame)",
            y_col="Bx (Y, DIL centroid frame)",
            x_label="S/I offset from DIL centroid (mm)\nS(+), I(-)",
            y_label="A/P offset from DIL centroid (mm)\nP(+), A(-)",
            x_symbol="z",
            y_symbol="y",
            plane_label="Sagittal plane",
            panel_label="B",
            export_config=export_config,
            norm=norm,
            cmap=cmap,
        )

        fig.subplots_adjust(top=0.88, bottom=0.11, right=0.90)
        cax = fig.add_axes([0.915, 0.17, 0.018, 0.66])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(r"$\langle \mathcal{P}_{D} \rangle$")

        _add_inside_panel_box(left_ax, left_text, export_config, x=0.97, y=0.03)
        _add_inside_panel_box(right_ax, right_text, export_config, x=0.97, y=0.03)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)
