from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from qa.config import QAStudyConfig
from qa.endpoints import (
    KEY_SECONDARY_ENDPOINT_COLUMNS,
    PRIMARY_ENDPOINT_COLUMNS,
    SAFETY_DISTANCE_ENDPOINT_COLUMNS,
)
from qa.families import QAFamilyOutputs

try:
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover - dependency is expected but keep graceful fallback
    smf = None
    ConvergenceWarning = Warning


HEADLINE_METRIC_COLUMNS = PRIMARY_ENDPOINT_COLUMNS + KEY_SECONDARY_ENDPOINT_COLUMNS
SAFETY_DISTANCE_METRIC_COLUMNS = SAFETY_DISTANCE_ENDPOINT_COLUMNS
METHOD_COMPARISON_METRIC_COLUMNS = HEADLINE_METRIC_COLUMNS + SAFETY_DISTANCE_METRIC_COLUMNS
CLUSTER_COL = "Base patient ID"
CONTRAST_ORDER = [
    "centroid_minus_real",
    "optimal_minus_real",
    "optimal_minus_centroid",
]
GROUP_NAME_ORDER = ["Real", "Centroid", "Optimal"]
METHOD_ORDER = [
    "patient_cluster_bootstrap",
    "family_mean_paired_t",
    "family_mean_wilcoxon",
    "mixedlm_patient_family",
]
SUMMARY_METHOD_COLUMNS = [
    "metric",
    "contrast_key",
    "group_a_name",
    "group_b_name",
    "method_key",
    "analysis_scale",
    "real_aggregation",
    "n_rows",
    "n_clusters",
    "n_families",
    "group_a_mean",
    "group_b_mean",
    "observed_mean_delta",
    "ci_lower_95",
    "ci_upper_95",
    "p_value",
    "test_statistic",
    "standardized_mean_delta",
    "ci_excludes_zero",
    "significance_label",
    "notes",
]


@dataclass
class QAStatsOutputs:
    headline_delta_long_df: pd.DataFrame
    headline_bootstrap_summary_df: pd.DataFrame
    headline_bootstrap_samples_df: pd.DataFrame
    safety_delta_long_df: pd.DataFrame
    safety_bootstrap_summary_df: pd.DataFrame
    safety_bootstrap_samples_df: pd.DataFrame
    group_bootstrap_summary_df: pd.DataFrame
    family_mean_delta_long_df: pd.DataFrame
    classical_paired_summary_df: pd.DataFrame
    mixedlm_contrast_summary_df: pd.DataFrame
    method_comparison_summary_df: pd.DataFrame


def _family_id_from_keys(df: pd.DataFrame) -> pd.Series:
    return (
        df[CLUSTER_COL].astype(str)
        + "::"
        + df["Relative DIL index"].astype("Int64").astype(str)
    )


def _deterministic_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


def _contrast_sort_key(contrast_key: str) -> int:
    try:
        return CONTRAST_ORDER.index(contrast_key)
    except ValueError:
        return len(CONTRAST_ORDER)


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


def _standardized_mean_delta(deltas: np.ndarray) -> float:
    if deltas.size <= 1:
        return float("nan")
    sd = float(np.std(deltas, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan")
    return float(np.mean(deltas) / sd)


def _build_delta_long_for_metrics(
    family_outputs: QAFamilyOutputs,
    metric_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    real_pairs = family_outputs.qa_real_core_pairs.copy()
    if "Family ID" not in real_pairs.columns:
        real_pairs["Family ID"] = _family_id_from_keys(real_pairs)
    for metric_col in metric_columns:
        if metric_col not in real_pairs.columns:
            continue

        centroid_delta_col = f"delta_centroid_minus_real__{metric_col}"
        optimal_delta_col = f"delta_optimal_minus_real__{metric_col}"

        centroid_df = real_pairs[
            [
                CLUSTER_COL,
                "Relative DIL index",
                "Family ID",
                "Patient ID",
                "Bx ID",
                metric_col,
                f"centroid_{metric_col}",
                centroid_delta_col,
            ]
        ].copy()
        centroid_df.columns = [
            CLUSTER_COL,
            "Relative DIL index",
            "Family ID",
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
            [
                CLUSTER_COL,
                "Relative DIL index",
                "Family ID",
                "Patient ID",
                "Bx ID",
                metric_col,
                f"optimal_{metric_col}",
                optimal_delta_col,
            ]
        ].copy()
        optimal_df.columns = [
            CLUSTER_COL,
            "Relative DIL index",
            "Family ID",
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
    if "Family ID" not in ref_pairs.columns:
        ref_pairs["Family ID"] = _family_id_from_keys(ref_pairs)
    for metric_col in metric_columns:
        if f"centroid_{metric_col}" not in ref_pairs.columns or f"optimal_{metric_col}" not in ref_pairs.columns:
            continue

        delta_col = f"delta_optimal_minus_centroid__{metric_col}"
        ref_df = ref_pairs[
            [
                CLUSTER_COL,
                "Relative DIL index",
                "Family ID",
                "centroid_Patient ID",
                "centroid_Bx ID",
                f"centroid_{metric_col}",
                f"optimal_{metric_col}",
                delta_col,
            ]
        ].copy()
        ref_df.columns = [
            CLUSTER_COL,
            "Relative DIL index",
            "Family ID",
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
            "Family ID",
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
    cluster_sums = cluster_stats["cluster_sum"].to_numpy(dtype=float)
    cluster_counts = cluster_stats["cluster_count"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    sampled_positions = rng.integers(0, len(cluster_stats), size=(n_bootstrap, len(cluster_stats)))

    boot = np.empty(n_bootstrap, dtype=float)
    for idx, positions in enumerate(sampled_positions):
        sampled_sums = cluster_sums[positions].sum()
        sampled_counts = cluster_counts[positions].sum()
        boot[idx] = sampled_sums / sampled_counts
    return boot


def _build_bootstrap_summary(
    delta_long_df: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...],
    config: QAStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = 1.0 - config.bootstrap_confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for metric_col in metric_columns:
        for contrast_key in CONTRAST_ORDER:
            sub = delta_long_df[
                (delta_long_df["metric"] == metric_col)
                & (delta_long_df["contrast_key"] == contrast_key)
            ].copy()
            if sub.empty:
                continue

            deltas = sub["delta_value"].to_numpy(dtype=float)
            observed_delta = float(np.mean(deltas))
            observed_group_a = float(sub["group_a_value"].mean())
            observed_group_b = float(sub["group_b_value"].mean())

            boot = _cluster_bootstrap_mean(
                sub,
                cluster_col=CLUSTER_COL,
                value_col="delta_value",
                n_bootstrap=config.bootstrap_iterations,
                seed=_deterministic_seed(config.bootstrap_seed, metric_col, contrast_key, "cluster_bootstrap"),
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
                    "n_families": int(sub["Family ID"].nunique()) if "Family ID" in sub.columns else np.nan,
                    "group_a_mean": observed_group_a,
                    "group_b_mean": observed_group_b,
                    "observed_mean_delta": observed_delta,
                    "bootstrap_ci_lower": ci_lower,
                    "bootstrap_ci_upper": ci_upper,
                    "bootstrap_p_two_sided": p_two_sided,
                    "standardized_mean_delta": _standardized_mean_delta(deltas),
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


def _build_group_bootstrap_summary(
    family_outputs: QAFamilyOutputs,
    metric_columns: list[str] | tuple[str, ...],
    config: QAStudyConfig,
) -> pd.DataFrame:
    qa_long = family_outputs.qa_family_members_long.copy()
    if "Family ID" not in qa_long.columns:
        qa_long["Family ID"] = _family_id_from_keys(qa_long)
    if "Family member label" not in qa_long.columns:
        qa_long["Family member label"] = qa_long["Simulated type"].replace(
            {"Centroid DIL": "Centroid", "Optimal DIL": "Optimal"}
        )
    alpha = 1.0 - config.bootstrap_confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    rows: list[dict[str, object]] = []
    for metric_col in metric_columns:
        if metric_col not in qa_long.columns:
            continue
        for group_name in GROUP_NAME_ORDER:
            sub = qa_long[qa_long["Family member label"].astype(str) == group_name].copy()
            if sub.empty:
                continue
            values = pd.to_numeric(sub[metric_col], errors="coerce").dropna()
            sub = sub.loc[values.index].copy()
            if sub.empty:
                continue
            boot = _cluster_bootstrap_mean(
                sub,
                cluster_col=CLUSTER_COL,
                value_col=metric_col,
                n_bootstrap=config.bootstrap_iterations,
                seed=_deterministic_seed(config.bootstrap_seed, metric_col, group_name, "group_bootstrap"),
            )
            rows.append(
                {
                    "metric": metric_col,
                    "group_name": group_name,
                    "n_rows": int(len(sub)),
                    "n_clusters": int(sub[CLUSTER_COL].nunique()),
                    "n_families": int(sub["Family ID"].nunique()) if "Family ID" in sub.columns else np.nan,
                    "observed_mean": float(sub[metric_col].mean()),
                    "bootstrap_ci_lower": float(np.percentile(boot, lower_q)),
                    "bootstrap_ci_upper": float(np.percentile(boot, upper_q)),
                    "bootstrap_iterations": int(config.bootstrap_iterations),
                    "bootstrap_confidence_level": float(config.bootstrap_confidence_level),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "metric",
                "group_name",
                "n_rows",
                "n_clusters",
                "n_families",
                "observed_mean",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "bootstrap_iterations",
                "bootstrap_confidence_level",
            ]
        )
    return pd.DataFrame(rows).sort_values(["metric", "group_name"]).reset_index(drop=True)


def _build_family_mean_delta_long_for_metrics(
    family_outputs: QAFamilyOutputs,
    metric_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    real_agg = family_outputs.qa_family_real_aggregated.copy()
    real_agg["Family ID"] = _family_id_from_keys(real_agg)
    real_agg["n_real_cores"] = pd.to_numeric(real_agg["Real_core_count"], errors="coerce")

    ref_pairs = family_outputs.qa_family_reference_pairs.copy()
    ref_pairs["Family ID"] = _family_id_from_keys(ref_pairs)

    ref_context_cols = [CLUSTER_COL, "Relative DIL index", "Family ID"]
    rows: list[pd.DataFrame] = []

    for metric_col in metric_columns:
        real_mean_col = f"real_mean__{metric_col}"
        centroid_col = f"centroid_{metric_col}"
        optimal_col = f"optimal_{metric_col}"
        delta_opt_cent_col = f"delta_optimal_minus_centroid__{metric_col}"

        if real_mean_col in real_agg.columns and centroid_col in ref_pairs.columns:
            centroid_df = real_agg[
                ref_context_cols + ["n_real_cores", real_mean_col]
            ].merge(
                ref_pairs[ref_context_cols + [centroid_col]],
                on=ref_context_cols,
                how="inner",
                validate="1:1",
            )
            centroid_df["metric"] = metric_col
            centroid_df["contrast_key"] = "centroid_minus_real"
            centroid_df["group_a_name"] = "Centroid"
            centroid_df["group_b_name"] = "Real"
            centroid_df["group_a_value"] = centroid_df[centroid_col]
            centroid_df["group_b_value"] = centroid_df[real_mean_col]
            centroid_df["delta_value"] = centroid_df["group_a_value"] - centroid_df["group_b_value"]
            centroid_df["real_aggregation"] = "family_mean"
            rows.append(
                centroid_df[
                    ref_context_cols
                    + [
                        "n_real_cores",
                        "metric",
                        "contrast_key",
                        "group_a_name",
                        "group_b_name",
                        "group_a_value",
                        "group_b_value",
                        "delta_value",
                        "real_aggregation",
                    ]
                ]
            )

        if real_mean_col in real_agg.columns and optimal_col in ref_pairs.columns:
            optimal_df = real_agg[
                ref_context_cols + ["n_real_cores", real_mean_col]
            ].merge(
                ref_pairs[ref_context_cols + [optimal_col]],
                on=ref_context_cols,
                how="inner",
                validate="1:1",
            )
            optimal_df["metric"] = metric_col
            optimal_df["contrast_key"] = "optimal_minus_real"
            optimal_df["group_a_name"] = "Optimal"
            optimal_df["group_b_name"] = "Real"
            optimal_df["group_a_value"] = optimal_df[optimal_col]
            optimal_df["group_b_value"] = optimal_df[real_mean_col]
            optimal_df["delta_value"] = optimal_df["group_a_value"] - optimal_df["group_b_value"]
            optimal_df["real_aggregation"] = "family_mean"
            rows.append(
                optimal_df[
                    ref_context_cols
                    + [
                        "n_real_cores",
                        "metric",
                        "contrast_key",
                        "group_a_name",
                        "group_b_name",
                        "group_a_value",
                        "group_b_value",
                        "delta_value",
                        "real_aggregation",
                    ]
                ]
            )

        if centroid_col in ref_pairs.columns and optimal_col in ref_pairs.columns:
            reference_df = ref_pairs[
                ref_context_cols + [centroid_col, optimal_col]
            ].copy()
            reference_df["n_real_cores"] = real_agg.set_index(ref_context_cols)["n_real_cores"].reindex(
                pd.MultiIndex.from_frame(reference_df[ref_context_cols])
            ).to_numpy()
            if delta_opt_cent_col in ref_pairs.columns:
                reference_df["delta_value"] = ref_pairs[delta_opt_cent_col].to_numpy(dtype=float)
            else:
                reference_df["delta_value"] = reference_df[optimal_col] - reference_df[centroid_col]
            reference_df["metric"] = metric_col
            reference_df["contrast_key"] = "optimal_minus_centroid"
            reference_df["group_a_name"] = "Optimal"
            reference_df["group_b_name"] = "Centroid"
            reference_df["group_a_value"] = reference_df[optimal_col]
            reference_df["group_b_value"] = reference_df[centroid_col]
            reference_df["real_aggregation"] = "not_applicable"
            rows.append(
                reference_df[
                    ref_context_cols
                    + [
                        "n_real_cores",
                        "metric",
                        "contrast_key",
                        "group_a_name",
                        "group_b_name",
                        "group_a_value",
                        "group_b_value",
                        "delta_value",
                        "real_aggregation",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                CLUSTER_COL,
                "Relative DIL index",
                "Family ID",
                "n_real_cores",
                "metric",
                "contrast_key",
                "group_a_name",
                "group_b_name",
                "group_a_value",
                "group_b_value",
                "delta_value",
                "real_aggregation",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["metric", "contrast_key", CLUSTER_COL, "Relative DIL index"]
    ).reset_index(drop=True)


def _paired_t_summary(
    family_mean_delta_long_df: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...],
    config: QAStudyConfig,
) -> pd.DataFrame:
    alpha = 1.0 - config.bootstrap_confidence_level
    rows: list[dict[str, object]] = []
    for metric_col in metric_columns:
        for contrast_key in CONTRAST_ORDER:
            sub = family_mean_delta_long_df[
                (family_mean_delta_long_df["metric"] == metric_col)
                & (family_mean_delta_long_df["contrast_key"] == contrast_key)
            ].copy()
            if sub.empty:
                continue
            deltas = sub["delta_value"].to_numpy(dtype=float)
            mean_delta = float(np.mean(deltas))
            if len(deltas) >= 2:
                t_res = stats.ttest_1samp(deltas, popmean=0.0, nan_policy="omit")
                sem = float(stats.sem(deltas, nan_policy="omit"))
                ci_low, ci_high = stats.t.interval(
                    config.bootstrap_confidence_level,
                    df=len(deltas) - 1,
                    loc=mean_delta,
                    scale=sem,
                )
                statistic = float(t_res.statistic)
                p_value = float(t_res.pvalue)
            else:
                ci_low = ci_high = statistic = p_value = float("nan")
            rows.append(
                {
                    "metric": metric_col,
                    "contrast_key": contrast_key,
                    "group_a_name": sub["group_a_name"].iloc[0],
                    "group_b_name": sub["group_b_name"].iloc[0],
                    "method_key": "family_mean_paired_t",
                    "analysis_scale": "family_aggregated_mean",
                    "real_aggregation": sub["real_aggregation"].iloc[0],
                    "n_rows": int(len(sub)),
                    "n_clusters": int(sub[CLUSTER_COL].nunique()),
                    "n_families": int(sub["Family ID"].nunique()),
                    "group_a_mean": float(sub["group_a_value"].mean()),
                    "group_b_mean": float(sub["group_b_value"].mean()),
                    "observed_mean_delta": mean_delta,
                    "ci_lower_95": float(ci_low) if np.isfinite(ci_low) else float("nan"),
                    "ci_upper_95": float(ci_high) if np.isfinite(ci_high) else float("nan"),
                    "p_value": p_value,
                    "test_statistic": statistic,
                    "standardized_mean_delta": _standardized_mean_delta(deltas),
                    "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)) if np.isfinite(ci_low) else False,
                    "significance_label": _format_significance_label(p_value),
                    "notes": "One-sample t-test on family-level paired deltas against zero.",
                }
            )
    if not rows:
        return pd.DataFrame(columns=SUMMARY_METHOD_COLUMNS)
    return pd.DataFrame(rows).sort_values(["metric", "contrast_key"]).reset_index(drop=True)


def _paired_wilcoxon_summary(
    family_mean_delta_long_df: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_col in metric_columns:
        for contrast_key in CONTRAST_ORDER:
            sub = family_mean_delta_long_df[
                (family_mean_delta_long_df["metric"] == metric_col)
                & (family_mean_delta_long_df["contrast_key"] == contrast_key)
            ].copy()
            if sub.empty:
                continue
            deltas = sub["delta_value"].to_numpy(dtype=float)
            non_zero = deltas[np.abs(deltas) > 0.0]
            if len(non_zero) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                wilcox = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox", mode="auto")
                statistic = float(wilcox.statistic)
                p_value = float(wilcox.pvalue)
            rows.append(
                {
                    "metric": metric_col,
                    "contrast_key": contrast_key,
                    "group_a_name": sub["group_a_name"].iloc[0],
                    "group_b_name": sub["group_b_name"].iloc[0],
                    "method_key": "family_mean_wilcoxon",
                    "analysis_scale": "family_aggregated_mean",
                    "real_aggregation": sub["real_aggregation"].iloc[0],
                    "n_rows": int(len(sub)),
                    "n_clusters": int(sub[CLUSTER_COL].nunique()),
                    "n_families": int(sub["Family ID"].nunique()),
                    "group_a_mean": float(sub["group_a_value"].mean()),
                    "group_b_mean": float(sub["group_b_value"].mean()),
                    "observed_mean_delta": float(np.mean(deltas)),
                    "ci_lower_95": float("nan"),
                    "ci_upper_95": float("nan"),
                    "p_value": p_value,
                    "test_statistic": statistic,
                    "standardized_mean_delta": _standardized_mean_delta(deltas),
                    "ci_excludes_zero": np.nan,
                    "significance_label": _format_significance_label(p_value),
                    "notes": "Wilcoxon signed-rank test on family-level paired deltas; no CI reported.",
                }
            )
    if not rows:
        return pd.DataFrame(columns=SUMMARY_METHOD_COLUMNS)
    return pd.DataFrame(rows).sort_values(["metric", "contrast_key"]).reset_index(drop=True)


def _mixedlm_contrast_summary(
    family_outputs: QAFamilyOutputs,
    metric_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    if smf is None:
        return pd.DataFrame(columns=SUMMARY_METHOD_COLUMNS)

    qa_long = family_outputs.qa_family_members_long.copy()
    if "Family ID" not in qa_long.columns:
        qa_long["Family ID"] = _family_id_from_keys(qa_long)
    qa_long["group"] = qa_long["Simulated type"].replace(
        {"Centroid DIL": "Centroid", "Optimal DIL": "Optimal"}
    )

    rows: list[dict[str, object]] = []
    for metric_col in metric_columns:
        if metric_col not in qa_long.columns:
            continue
        data = qa_long[[CLUSTER_COL, "Family ID", "group", metric_col]].dropna().copy()
        if data.empty or data["group"].nunique() < 3:
            continue
        data = data.rename(columns={CLUSTER_COL: "patient", "Family ID": "family", metric_col: "y"})

        fit = None
        fit_note = "MixedLM with patient random intercept and family variance component."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                model = smf.mixedlm(
                    "y ~ C(group, Treatment(reference='Real'))",
                    data=data,
                    groups="patient",
                    vc_formula={"family": "0 + C(family)"},
                )
                fit = model.fit(reml=False, method="lbfgs", disp=False)
            except Exception:
                try:
                    model = smf.mixedlm(
                        "y ~ C(group, Treatment(reference='Real'))",
                        data=data,
                        groups="patient",
                    )
                    fit = model.fit(reml=False, method="lbfgs", disp=False)
                    fit_note = "MixedLM fallback with patient random intercept only."
                except Exception:
                    fit = None
        if fit is None:
            continue

        mean_map = data.groupby("group")["y"].mean().to_dict()
        fe_param_index = list(fit.fe_params.index)
        coef_centroid = "C(group, Treatment(reference='Real'))[T.Centroid]"
        coef_optimal = "C(group, Treatment(reference='Real'))[T.Optimal]"
        conf = fit.conf_int()

        def _append_direct_contrast(contrast_key: str, coef_name: str, group_a: str, group_b: str) -> None:
            if coef_name not in fit.params.index:
                return
            ci_row = conf.loc[coef_name]
            p_value = float(fit.pvalues[coef_name])
            rows.append(
                {
                    "metric": metric_col,
                    "contrast_key": contrast_key,
                    "group_a_name": group_a,
                    "group_b_name": group_b,
                    "method_key": "mixedlm_patient_family",
                    "analysis_scale": "family_member_long",
                    "real_aggregation": "not_applicable",
                    "n_rows": int(len(data)),
                    "n_clusters": int(data["patient"].nunique()),
                    "n_families": int(data["family"].nunique()),
                    "group_a_mean": float(mean_map.get(group_a, np.nan)),
                    "group_b_mean": float(mean_map.get(group_b, np.nan)),
                    "observed_mean_delta": float(fit.params[coef_name]),
                    "ci_lower_95": float(ci_row.iloc[0]),
                    "ci_upper_95": float(ci_row.iloc[1]),
                    "p_value": p_value,
                    "test_statistic": float(fit.tvalues[coef_name]),
                    "standardized_mean_delta": np.nan,
                    "ci_excludes_zero": bool((ci_row.iloc[0] > 0.0) or (ci_row.iloc[1] < 0.0)),
                    "significance_label": _format_significance_label(p_value),
                    "notes": fit_note,
                }
            )

        _append_direct_contrast("centroid_minus_real", coef_centroid, "Centroid", "Real")
        _append_direct_contrast("optimal_minus_real", coef_optimal, "Optimal", "Real")

        if coef_centroid in fe_param_index and coef_optimal in fe_param_index:
            contrast_vec = np.zeros((1, len(fe_param_index)), dtype=float)
            contrast_vec[0, fe_param_index.index(coef_optimal)] = 1.0
            contrast_vec[0, fe_param_index.index(coef_centroid)] = -1.0
            contrast_res = fit.t_test(contrast_vec)
            ci = contrast_res.conf_int(alpha=0.05)
            p_value = float(np.squeeze(contrast_res.pvalue))
            estimate = float(np.squeeze(contrast_res.effect))
            stat = float(np.squeeze(contrast_res.tvalue))
            ci_low = float(np.squeeze(ci[:, 0]))
            ci_high = float(np.squeeze(ci[:, 1]))
            rows.append(
                {
                    "metric": metric_col,
                    "contrast_key": "optimal_minus_centroid",
                    "group_a_name": "Optimal",
                    "group_b_name": "Centroid",
                    "method_key": "mixedlm_patient_family",
                    "analysis_scale": "family_member_long",
                    "real_aggregation": "not_applicable",
                    "n_rows": int(len(data)),
                    "n_clusters": int(data["patient"].nunique()),
                    "n_families": int(data["family"].nunique()),
                    "group_a_mean": float(mean_map.get("Optimal", np.nan)),
                    "group_b_mean": float(mean_map.get("Centroid", np.nan)),
                    "observed_mean_delta": estimate,
                    "ci_lower_95": ci_low,
                    "ci_upper_95": ci_high,
                    "p_value": p_value,
                    "test_statistic": stat,
                    "standardized_mean_delta": np.nan,
                    "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
                    "significance_label": _format_significance_label(p_value),
                    "notes": fit_note,
                }
            )

    if not rows:
        return pd.DataFrame(columns=SUMMARY_METHOD_COLUMNS)
    return pd.DataFrame(rows).sort_values(["metric", "contrast_key"]).reset_index(drop=True)


def _method_comparison_from_bootstrap(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    out = summary_df.copy()
    out["method_key"] = "patient_cluster_bootstrap"
    out["analysis_scale"] = np.where(
        out["contrast_key"].astype(str) == "optimal_minus_centroid",
        "family_reference_pair",
        "real_core_pair",
    )
    out["real_aggregation"] = np.where(
        out["contrast_key"].astype(str) == "optimal_minus_centroid",
        "not_applicable",
        "core_level",
    )
    return out.rename(
        columns={
            "n_rows": "n_rows",
            "n_clusters": "n_clusters",
            "n_families": "n_families",
            "observed_mean_delta": "observed_mean_delta",
            "bootstrap_ci_lower": "ci_lower_95",
            "bootstrap_ci_upper": "ci_upper_95",
            "bootstrap_p_two_sided": "p_value",
            "standardized_mean_delta": "standardized_mean_delta",
            "supports_nonzero_ci": "ci_excludes_zero",
        }
    )[
        [
            "metric",
            "contrast_key",
            "group_a_name",
            "group_b_name",
            "method_key",
            "analysis_scale",
            "real_aggregation",
            "n_rows",
            "n_clusters",
            "n_families",
            "group_a_mean",
            "group_b_mean",
            "observed_mean_delta",
            "ci_lower_95",
            "ci_upper_95",
            "p_value",
            "standardized_mean_delta",
            "ci_excludes_zero",
            "significance_label",
        ]
    ].assign(test_statistic=np.nan, notes="Patient-clustered paired-delta percentile bootstrap.")


def _build_method_comparison_summary(
    bootstrap_summary_dfs: list[pd.DataFrame],
    classical_paired_summary_df: pd.DataFrame,
    mixedlm_contrast_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in bootstrap_summary_dfs:
        if frame is not None and not frame.empty:
            frames.append(_method_comparison_from_bootstrap(frame))
    if classical_paired_summary_df is not None and not classical_paired_summary_df.empty:
        frames.append(classical_paired_summary_df.copy())
    if mixedlm_contrast_summary_df is not None and not mixedlm_contrast_summary_df.empty:
        frames.append(mixedlm_contrast_summary_df.copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["metric_order"] = out["metric"].map(
        {metric: idx for idx, metric in enumerate(METHOD_COMPARISON_METRIC_COLUMNS)}
    )
    out["contrast_order"] = out["contrast_key"].map(_contrast_sort_key)
    out["method_order"] = out["method_key"].map({key: idx for idx, key in enumerate(METHOD_ORDER)})
    out = out.sort_values(["metric_order", "contrast_order", "method_order"]).reset_index(drop=True)
    return out.drop(columns=["metric_order", "contrast_order", "method_order"], errors="ignore")


def build_qa_stats_outputs(
    family_outputs: QAFamilyOutputs,
    config: QAStudyConfig,
) -> QAStatsOutputs:
    headline_delta_long_df = _build_delta_long_for_metrics(family_outputs, HEADLINE_METRIC_COLUMNS)
    headline_bootstrap_summary_df, headline_bootstrap_samples_df = _build_bootstrap_summary(
        headline_delta_long_df,
        HEADLINE_METRIC_COLUMNS,
        config,
    )

    safety_delta_long_df = _build_delta_long_for_metrics(family_outputs, SAFETY_DISTANCE_METRIC_COLUMNS)
    safety_bootstrap_summary_df, safety_bootstrap_samples_df = _build_bootstrap_summary(
        safety_delta_long_df,
        SAFETY_DISTANCE_METRIC_COLUMNS,
        config,
    )

    group_bootstrap_summary_df = _build_group_bootstrap_summary(
        family_outputs,
        METHOD_COMPARISON_METRIC_COLUMNS,
        config,
    )
    family_mean_delta_long_df = _build_family_mean_delta_long_for_metrics(
        family_outputs,
        METHOD_COMPARISON_METRIC_COLUMNS,
    )
    classical_paired_summary_df = pd.concat(
        [
            _paired_t_summary(family_mean_delta_long_df, METHOD_COMPARISON_METRIC_COLUMNS, config),
            _paired_wilcoxon_summary(family_mean_delta_long_df, METHOD_COMPARISON_METRIC_COLUMNS),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["metric", "contrast_key", "method_key"]).reset_index(drop=True)
    mixedlm_contrast_summary_df = _mixedlm_contrast_summary(
        family_outputs,
        METHOD_COMPARISON_METRIC_COLUMNS,
    )
    method_comparison_summary_df = _build_method_comparison_summary(
        [headline_bootstrap_summary_df, safety_bootstrap_summary_df],
        classical_paired_summary_df,
        mixedlm_contrast_summary_df,
    )

    return QAStatsOutputs(
        headline_delta_long_df=headline_delta_long_df,
        headline_bootstrap_summary_df=headline_bootstrap_summary_df,
        headline_bootstrap_samples_df=headline_bootstrap_samples_df,
        safety_delta_long_df=safety_delta_long_df,
        safety_bootstrap_summary_df=safety_bootstrap_summary_df,
        safety_bootstrap_samples_df=safety_bootstrap_samples_df,
        group_bootstrap_summary_df=group_bootstrap_summary_df,
        family_mean_delta_long_df=family_mean_delta_long_df,
        classical_paired_summary_df=classical_paired_summary_df,
        mixedlm_contrast_summary_df=mixedlm_contrast_summary_df,
        method_comparison_summary_df=method_comparison_summary_df,
    )
