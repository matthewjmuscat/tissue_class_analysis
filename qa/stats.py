from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qa.config import QAStudyConfig
from qa.endpoints import KEY_SECONDARY_ENDPOINT_COLUMNS, PRIMARY_ENDPOINT_COLUMNS
from qa.families import QAFamilyOutputs


HEADLINE_METRIC_COLUMNS = PRIMARY_ENDPOINT_COLUMNS + KEY_SECONDARY_ENDPOINT_COLUMNS
CLUSTER_COL = "Base patient ID"


@dataclass
class QAStatsOutputs:
    headline_delta_long_df: pd.DataFrame
    headline_bootstrap_summary_df: pd.DataFrame
    headline_bootstrap_samples_df: pd.DataFrame


def _build_headline_delta_long(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    real_pairs = family_outputs.qa_real_core_pairs.copy()
    for metric_col in HEADLINE_METRIC_COLUMNS:
        if metric_col not in real_pairs.columns:
            continue

        centroid_delta_col = f"delta_centroid_minus_real__{metric_col}"
        optimal_delta_col = f"delta_optimal_minus_real__{metric_col}"

        centroid_df = real_pairs[
            [CLUSTER_COL, "Relative DIL index", "Patient ID", "Bx ID", metric_col, f"centroid_{metric_col}", centroid_delta_col]
        ].copy()
        centroid_df.columns = [
            CLUSTER_COL,
            "Relative DIL index",
            "Observation patient ID",
            "Observation Bx ID",
            "group_b_value",
            "group_a_value",
            "delta_value",
        ]
        centroid_df["contrast_key"] = "centroid_minus_real"
        centroid_df["group_a_name"] = "Centroid"
        centroid_df["group_b_name"] = "Real"
        centroid_df["metric"] = metric_col
        rows.append(centroid_df)

        optimal_df = real_pairs[
            [CLUSTER_COL, "Relative DIL index", "Patient ID", "Bx ID", metric_col, f"optimal_{metric_col}", optimal_delta_col]
        ].copy()
        optimal_df.columns = [
            CLUSTER_COL,
            "Relative DIL index",
            "Observation patient ID",
            "Observation Bx ID",
            "group_b_value",
            "group_a_value",
            "delta_value",
        ]
        optimal_df["contrast_key"] = "optimal_minus_real"
        optimal_df["group_a_name"] = "Optimal"
        optimal_df["group_b_name"] = "Real"
        optimal_df["metric"] = metric_col
        rows.append(optimal_df)

    ref_pairs = family_outputs.qa_family_reference_pairs.copy()
    for metric_col in HEADLINE_METRIC_COLUMNS:
        if f"centroid_{metric_col}" not in ref_pairs.columns or f"optimal_{metric_col}" not in ref_pairs.columns:
            continue

        delta_col = f"delta_optimal_minus_centroid__{metric_col}"
        ref_df = ref_pairs[
            [CLUSTER_COL, "Relative DIL index", "centroid_Patient ID", "centroid_Bx ID", f"centroid_{metric_col}", f"optimal_{metric_col}", delta_col]
        ].copy()
        ref_df.columns = [
            CLUSTER_COL,
            "Relative DIL index",
            "Observation patient ID",
            "Observation Bx ID",
            "group_b_value",
            "group_a_value",
            "delta_value",
        ]
        ref_df["contrast_key"] = "optimal_minus_centroid"
        ref_df["group_a_name"] = "Optimal"
        ref_df["group_b_name"] = "Centroid"
        ref_df["metric"] = metric_col
        rows.append(ref_df)

    out = pd.concat(rows, ignore_index=True)
    out = out[
        [
            CLUSTER_COL,
            "Relative DIL index",
            "Observation patient ID",
            "Observation Bx ID",
            "contrast_key",
            "group_a_name",
            "group_b_name",
            "metric",
            "group_a_value",
            "group_b_value",
            "delta_value",
        ]
    ]
    return out.sort_values(["metric", "contrast_key", CLUSTER_COL, "Relative DIL index"]).reset_index(drop=True)


def _cluster_bootstrap_mean(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    cluster_stats = (
        df.groupby(cluster_col)[value_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "cluster_sum", "count": "cluster_count"})
        .reset_index()
    )
    cluster_ids = cluster_stats[cluster_col].to_numpy()
    cluster_sums = cluster_stats["cluster_sum"].to_numpy(dtype=float)
    cluster_counts = cluster_stats["cluster_count"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    sampled_positions = rng.integers(0, len(cluster_ids), size=(n_bootstrap, len(cluster_ids)))

    boot = np.empty(n_bootstrap, dtype=float)
    for idx, positions in enumerate(sampled_positions):
        sampled_sums = cluster_sums[positions].sum()
        sampled_counts = cluster_counts[positions].sum()
        boot[idx] = sampled_sums / sampled_counts
    return boot


def _format_significance_label(p_value: float) -> str:
    if np.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def build_headline_bootstrap_summary(
    headline_delta_long_df: pd.DataFrame,
    config: QAStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = 1.0 - config.bootstrap_confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    contrast_order = [
        "centroid_minus_real",
        "optimal_minus_real",
        "optimal_minus_centroid",
    ]

    for metric_col in HEADLINE_METRIC_COLUMNS:
        for contrast_key in contrast_order:
            sub = headline_delta_long_df[
                (headline_delta_long_df["metric"] == metric_col)
                & (headline_delta_long_df["contrast_key"] == contrast_key)
            ].copy()
            if sub.empty:
                continue

            observed_delta = float(sub["delta_value"].mean())
            observed_group_a = float(sub["group_a_value"].mean())
            observed_group_b = float(sub["group_b_value"].mean())
            delta_std = float(sub["delta_value"].std(ddof=1)) if len(sub) > 1 else float("nan")
            standardized_mean_delta = (
                observed_delta / delta_std if np.isfinite(delta_std) and delta_std > 0 else float("nan")
            )

            boot = _cluster_bootstrap_mean(
                sub,
                cluster_col=CLUSTER_COL,
                value_col="delta_value",
                n_bootstrap=config.bootstrap_iterations,
                seed=config.bootstrap_seed + abs(hash((metric_col, contrast_key))) % 100000,
            )

            ci_lower = float(np.percentile(boot, lower_q))
            ci_upper = float(np.percentile(boot, upper_q))
            p_two_sided = float(2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0)))
            p_two_sided = min(max(p_two_sided, 0.0), 1.0)
            supports_nonzero_ci = bool((ci_lower > 0.0) or (ci_upper < 0.0))

            summary_rows.append(
                {
                    "metric": metric_col,
                    "contrast_key": contrast_key,
                    "group_a_name": sub["group_a_name"].iloc[0],
                    "group_b_name": sub["group_b_name"].iloc[0],
                    "n_rows": int(len(sub)),
                    "n_clusters": int(sub[CLUSTER_COL].nunique()),
                    "group_a_mean": observed_group_a,
                    "group_b_mean": observed_group_b,
                    "observed_mean_delta": observed_delta,
                    "bootstrap_ci_lower": ci_lower,
                    "bootstrap_ci_upper": ci_upper,
                    "bootstrap_p_two_sided": p_two_sided,
                    "standardized_mean_delta": standardized_mean_delta,
                    "supports_nonzero_ci": supports_nonzero_ci,
                    "significance_label": _format_significance_label(p_two_sided),
                    "bootstrap_iterations": int(config.bootstrap_iterations),
                    "bootstrap_confidence_level": float(config.bootstrap_confidence_level),
                }
            )

            bootstrap_rows.extend(
                {
                    "metric": metric_col,
                    "contrast_key": contrast_key,
                    "bootstrap_index": int(i),
                    "bootstrap_mean_delta": float(value),
                }
                for i, value in enumerate(boot)
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["metric", "contrast_key"]).reset_index(drop=True)
    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(
        ["metric", "contrast_key", "bootstrap_index"]
    ).reset_index(drop=True)
    return summary_df, bootstrap_df


def build_qa_stats_outputs(
    family_outputs: QAFamilyOutputs,
    config: QAStudyConfig,
) -> QAStatsOutputs:
    headline_delta_long_df = _build_headline_delta_long(family_outputs)
    headline_bootstrap_summary_df, headline_bootstrap_samples_df = build_headline_bootstrap_summary(
        headline_delta_long_df,
        config,
    )
    return QAStatsOutputs(
        headline_delta_long_df=headline_delta_long_df,
        headline_bootstrap_summary_df=headline_bootstrap_summary_df,
        headline_bootstrap_samples_df=headline_bootstrap_samples_df,
    )
