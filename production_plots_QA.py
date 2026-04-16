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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    "Real": "#0b3b8a",
    "Centroid": "#c75000",
    "Optimal": "#2a9d8f",
}

CONTRAST_COLOR_MAP = {
    "centroid_minus_real": "#c75000",
    "optimal_minus_real": "#2a9d8f",
    "optimal_minus_centroid": "#7a5195",
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
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax.tick_params(axis="both", which="major", length=5, width=1.0)
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

        annotation_offsets = [(-14, 12), (10, 14), (12, -14), (-16, -12), (18, 0), (-18, 0)]

        for panel_idx, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
            sub = reference_disagreement_df[reference_disagreement_df["metric"] == metric].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

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
                for idx, (_, row) in enumerate(highlight_df.iterrows()):
                    dx, dy = annotation_offsets[idx % len(annotation_offsets)]
                    txt = ax.annotate(
                        str(row["Family display label"]),
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

            mean_delta = float(sub["delta_value"].mean())
            median_delta = float(sub["delta_value"].median())
            max_abs_idx = sub["abs_delta_value"].idxmax()
            max_abs_row = sub.loc[max_abs_idx]
            stats_lines = [
                rf"$\overline{{\Delta}}^{{(O-C)}}$ = {mean_delta:+.3f}",
                rf"$\mathrm{{med}}(\Delta^{{(O-C)}})$ = {median_delta:+.3f}",
                rf"$\max |\Delta^{{(O-C)}}|$ = {max_abs_row['abs_delta_value']:.3f}",
                f"Top family: {max_abs_row['Family display label']}",
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
            ax.tick_params(axis="y", which="major", labelleft=True)
            _add_panel_label(ax, heading, export_config, y=1.12)
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
    optimizer_color = CONTRAST_COLOR_MAP["optimal_minus_centroid"]
    double_sextant_order = ["LP", "LA", "RP", "RA"]
    ap_order = ["Posterior", "Anterior"]
    si_order = ["Apex", "Mid", "Base"]

    with _font_rc(export_config):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), dpi=export_config.dpi, sharey=True)
        axes_flat = list(np.ravel(axes))
        y_vals = df[y_col].to_numpy(dtype=float)
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        y_span = max(y_max - y_min, 0.08)
        y_lim = (
            min(-0.14, y_min - 0.10 * y_span),
            max(0.10, y_max + 0.12 * y_span),
        )

        ax = axes_flat[0]
        scatter_df = df.dropna(subset=["DIL Maximum 3D diameter", y_col]).copy()
        x = scatter_df["DIL Maximum 3D diameter"].to_numpy(dtype=float)
        y = scatter_df[y_col].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            s=62,
            color=optimizer_color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )
        if len(x) >= 2:
            x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            slope, intercept = np.polyfit(x, y, deg=1)
            ax.plot(
                x_grid,
                intercept + slope * x_grid,
                color=optimizer_color,
                linewidth=1.8,
                linestyle=(0, (5, 3)),
                zorder=2,
            )
            rho_s = pd.Series(x).corr(pd.Series(y), method="spearman")
            ax.text(
                0.03,
                0.95,
                rf"$\rho_s = {rho_s:+.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=export_config.annotation_fontsize,
                bbox=ANNOT_BBOX,
            )
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title("Maximum DIL diameter", pad=18)
        ax.set_xlabel("DIL maximum 3D diameter (mm)")
        ax.set_ylabel(r"Optimizer increment $\Delta^{(O-C)}\langle \mathcal{P}_{D} \rangle$")
        ax.set_ylim(*y_lim)
        _style_axes(ax, export_config, x_minor_ticks=2)
        _add_panel_label(ax, "A", export_config)

        cat_specs = [
            (axes_flat[1], "DIL DIL prostate sextant (AP)", ap_order, "AP lesion location", "B"),
            (axes_flat[2], "DIL SI short", si_order, "SI lesion location", "C"),
            (axes_flat[3], "DIL double sextant zone", double_sextant_order, "Double sextant zone", "D"),
        ]
        for ax, category_col, category_order, title, panel_label in cat_specs:
            sub = df[df[category_col].isin(category_order)].copy()
            boxplot_data = []
            positions = np.arange(len(category_order), dtype=float)
            for idx, category in enumerate(category_order):
                cat_df = sub[sub[category_col] == category].copy()
                values = cat_df[y_col].to_numpy(dtype=float)
                boxplot_data.append(values)
                xs = _spread_positions(len(values), positions[idx], half_width=0.12)
                ax.scatter(
                    xs,
                    values,
                    s=58,
                    color=optimizer_color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.95,
                    zorder=3,
                )

            bp = ax.boxplot(
                boxplot_data,
                positions=positions,
                widths=0.50,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
                whiskerprops={"color": "#4a4a4a", "linewidth": 1.0},
                capprops={"color": "#4a4a4a", "linewidth": 1.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(optimizer_color)
                patch.set_alpha(0.15)
                patch.set_edgecolor(optimizer_color)
                patch.set_linewidth(1.2)

            ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.set_title(title, pad=18)
            ax.set_xticks(positions, _category_tick_labels(sub, category_col, category_order))
            if ax in {axes_flat[2]}:
                ax.set_ylabel(r"Optimizer increment $\Delta^{(O-C)}\langle \mathcal{P}_{D} \rangle$")
            ax.set_ylim(*y_lim)
            _style_axes(ax, export_config)
            _add_panel_label(ax, panel_label, export_config)

        fig.subplots_adjust(top=0.92, bottom=0.13, hspace=0.38, wspace=0.22)
        return _save_figure_multi(fig, save_dir, file_stem, export_config)
