from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qa.families import QAFamilyOutputs
from qa.stats import HEADLINE_METRIC_COLUMNS, QAStatsOutputs


HEADLINE_METRIC_LABEL_MAP = {
    "DIL Global Mean BE": r"Mean DIL support, $\langle P_D \rangle$",
    "DIL Global Max BE": r"Peak DIL support, $\max(P_D)$",
}

FAMILY_GROUP_ORDER = ("Real", "Centroid", "Optimal")
FAMILY_GROUP_LABEL_MAP = {
    "Real": "Real",
    "Centroid": "Centroid",
    "Optimal": "Optimal",
}

CONTRAST_ORDER = (
    "centroid_minus_real",
    "optimal_minus_real",
    "optimal_minus_centroid",
)

CONTRAST_LABEL_MAP = {
    "centroid_minus_real": "Centroid - Real",
    "optimal_minus_real": "Optimal - Real",
    "optimal_minus_centroid": "Optimal - Centroid",
}


@dataclass
class QAPlotDataOutputs:
    qa_plot_family_comparison_long: pd.DataFrame
    qa_plot_headroom_long: pd.DataFrame
    qa_plot_reference_disagreement: pd.DataFrame


def _assign_pair_offsets(real_pairs: pd.DataFrame) -> pd.Series:
    offsets = pd.Series(0.0, index=real_pairs.index, dtype=float)
    grouped = real_pairs.groupby(["Base patient ID", "Relative DIL index"], sort=True)
    for _, idx in grouped.groups.items():
        ordered_idx = sorted(
            idx,
            key=lambda i: (
                str(real_pairs.at[i, "Patient ID"]),
                str(real_pairs.at[i, "Bx ID"]),
            ),
        )
        if len(ordered_idx) == 1:
            offsets.loc[ordered_idx] = 0.0
            continue
        offsets.loc[ordered_idx] = np.linspace(-0.08, 0.08, len(ordered_idx))
    return offsets


def build_family_comparison_long_df(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    real_pairs = family_outputs.qa_real_core_pairs.copy().reset_index(drop=True)
    real_pairs["Observation ID"] = (
        real_pairs["Patient ID"].astype(str) + "::" + real_pairs["Bx ID"].astype(str)
    )
    real_pairs["pair_offset"] = _assign_pair_offsets(real_pairs)

    rows: list[dict[str, object]] = []
    group_specs = [
        ("Real", None, 0),
        ("Centroid", "centroid_", 1),
        ("Optimal", "optimal_", 2),
    ]

    for _, row in real_pairs.iterrows():
        for metric in HEADLINE_METRIC_COLUMNS:
            for group_key, prefix, order in group_specs:
                value_col = metric if prefix is None else f"{prefix}{metric}"
                if value_col not in row.index:
                    continue
                rows.append(
                    {
                        "Base patient ID": row["Base patient ID"],
                        "Relative DIL index": row["Relative DIL index"],
                        "Family ID": row.get(
                            "Family ID",
                            f"{row['Base patient ID']}::{row['Relative DIL index']}",
                        ),
                        "Observation patient ID": row["Patient ID"],
                        "Observation Bx ID": row["Bx ID"],
                        "Observation ID": row["Observation ID"],
                        "Family_real_core_count": row.get("Family_real_core_count"),
                        "metric": metric,
                        "metric_label": HEADLINE_METRIC_LABEL_MAP.get(metric, metric),
                        "group_key": group_key,
                        "group_label": FAMILY_GROUP_LABEL_MAP[group_key],
                        "group_order": order,
                        "pair_offset": row["pair_offset"],
                        "value": row[value_col],
                    }
                )

    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "metric",
            "Base patient ID",
            "Relative DIL index",
            "Observation patient ID",
            "Observation Bx ID",
            "group_order",
        ]
    ).reset_index(drop=True)


def build_headroom_long_df(stats_outputs: QAStatsOutputs) -> pd.DataFrame:
    out = stats_outputs.headline_delta_long_df.copy()
    out["metric_label"] = out["metric"].map(HEADLINE_METRIC_LABEL_MAP).fillna(out["metric"])
    out["contrast_label"] = out["contrast_key"].map(CONTRAST_LABEL_MAP).fillna(out["contrast_key"])
    out["contrast_order"] = out["contrast_key"].map(
        {contrast: idx for idx, contrast in enumerate(CONTRAST_ORDER)}
    )
    out["Observation ID"] = (
        out["Observation patient ID"].astype(str) + "::" + out["Observation Bx ID"].astype(str)
    )
    return out.sort_values(
        [
            "metric",
            "contrast_order",
            "Base patient ID",
            "Relative DIL index",
            "Observation patient ID",
            "Observation Bx ID",
        ]
    ).reset_index(drop=True)


def build_reference_disagreement_df(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    ref_pairs = family_outputs.qa_family_reference_pairs.copy()
    family_meta = family_outputs.qa_family_audit[
        ["Base patient ID", "Relative DIL index", "Family_real_core_count", "Relative_DIL_IDs"]
    ].copy()
    ref_pairs = ref_pairs.merge(
        family_meta,
        on=["Base patient ID", "Relative DIL index"],
        how="left",
        validate="1:1",
    )

    rows: list[dict[str, object]] = []
    for _, row in ref_pairs.iterrows():
        family_id = f"{row['Base patient ID']}::{row['Relative DIL index']}"
        family_display_label = family_id
        for metric in HEADLINE_METRIC_COLUMNS:
            centroid_col = f"centroid_{metric}"
            optimal_col = f"optimal_{metric}"
            delta_col = f"delta_optimal_minus_centroid__{metric}"
            if centroid_col not in row.index or optimal_col not in row.index or delta_col not in row.index:
                continue
            rows.append(
                {
                    "Base patient ID": row["Base patient ID"],
                    "Relative DIL index": row["Relative DIL index"],
                    "Family ID": family_id,
                    "Family display label": family_display_label,
                    "Relative DIL IDs": row.get("Relative_DIL_IDs"),
                    "Family_real_core_count": row.get("Family_real_core_count"),
                    "metric": metric,
                    "metric_label": HEADLINE_METRIC_LABEL_MAP.get(metric, metric),
                    "centroid_value": row[centroid_col],
                    "optimal_value": row[optimal_col],
                    "delta_value": row[delta_col],
                    "abs_delta_value": abs(float(row[delta_col])),
                    "centroid_BX to DIL centroid distance": row.get(
                        "centroid_BX to DIL centroid distance"
                    ),
                    "optimal_BX to DIL centroid distance": row.get(
                        "optimal_BX to DIL centroid distance"
                    ),
                    "centroid_Urethra NN dist mean": row.get("centroid_Urethra NN dist mean"),
                    "optimal_Urethra NN dist mean": row.get("optimal_Urethra NN dist mean"),
                    "centroid_Rectum NN dist mean": row.get("centroid_Rectum NN dist mean"),
                    "optimal_Rectum NN dist mean": row.get("optimal_Rectum NN dist mean"),
                }
            )

    out = pd.DataFrame(rows)
    out["abs_delta_rank"] = (
        out.groupby("metric")["abs_delta_value"].rank(method="first", ascending=False).astype(int)
    )
    return out.sort_values(["metric", "abs_delta_rank", "Family ID"]).reset_index(drop=True)


def build_qa_plot_data_outputs(
    family_outputs: QAFamilyOutputs,
    stats_outputs: QAStatsOutputs,
) -> QAPlotDataOutputs:
    return QAPlotDataOutputs(
        qa_plot_family_comparison_long=build_family_comparison_long_df(family_outputs),
        qa_plot_headroom_long=build_headroom_long_df(stats_outputs),
        qa_plot_reference_disagreement=build_reference_disagreement_df(family_outputs),
    )
