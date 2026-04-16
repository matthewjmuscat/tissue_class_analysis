from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qa.families import QAFamilyOutputs
from qa.load import QASourceTables
from qa.notation import metric_math
from qa.stats import (
    HEADLINE_METRIC_COLUMNS,
    SAFETY_DISTANCE_METRIC_COLUMNS,
    QAStatsOutputs,
)


HEADLINE_METRIC_LABEL_MAP = {
    metric: metric_math(metric)
    for metric in HEADLINE_METRIC_COLUMNS
}
SAFETY_DISTANCE_LABEL_MAP = {
    metric: metric_math(metric)
    for metric in SAFETY_DISTANCE_METRIC_COLUMNS
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

PROFILE_SELECTION_REASON_MAP = {
    "largest_positive_optimal_minus_centroid": r"Positive $\Delta^{(O-C)}$ case",
    "largest_negative_optimal_minus_centroid": r"Negative $\Delta^{(O-C)}$ case",
    "largest_total_headroom": "Largest total headroom",
    "representative_small_optimal_minus_centroid": "Near-centroid case",
}

PROFILE_SELECTION_ORDER = (
    "largest_positive_optimal_minus_centroid",
    "largest_negative_optimal_minus_centroid",
    "largest_total_headroom",
    "representative_small_optimal_minus_centroid",
)


@dataclass
class QAPlotDataOutputs:
    qa_plot_family_comparison_long: pd.DataFrame
    qa_plot_headroom_long: pd.DataFrame
    qa_plot_reference_disagreement: pd.DataFrame
    qa_plot_safety_distance_long: pd.DataFrame
    qa_plot_selected_profile_long: pd.DataFrame
    qa_selected_profile_cases: pd.DataFrame
    qa_family_optimizer_difficulty: pd.DataFrame


def _alpha_code(index: int) -> str:
    if index < 0:
        raise ValueError("index must be >= 0")
    chars: list[str] = []
    n = int(index)
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


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


def _build_family_comparison_long_from_real_pairs(
    real_pairs: pd.DataFrame,
    metric_columns: tuple[str, ...] | list[str],
    metric_label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    work = real_pairs.copy().reset_index(drop=True)
    work["Observation ID"] = work["Patient ID"].astype(str) + "::" + work["Bx ID"].astype(str)
    work["pair_offset"] = _assign_pair_offsets(work)

    rows: list[dict[str, object]] = []
    group_specs = [
        ("Real", None, 0),
        ("Centroid", "centroid_", 1),
        ("Optimal", "optimal_", 2),
    ]

    for _, row in work.iterrows():
        for metric in metric_columns:
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
                        "metric_label": (metric_label_map or {}).get(metric, metric_math(metric)),
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


def build_family_comparison_long_df(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    return _build_family_comparison_long_from_real_pairs(
        family_outputs.qa_real_core_pairs,
        HEADLINE_METRIC_COLUMNS,
        HEADLINE_METRIC_LABEL_MAP,
    )


def build_safety_distance_long_df(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    return _build_family_comparison_long_from_real_pairs(
        family_outputs.qa_real_core_pairs,
        SAFETY_DISTANCE_METRIC_COLUMNS,
        SAFETY_DISTANCE_LABEL_MAP,
    )


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
                    "Family display label": family_id,
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


def _pick_real_core_for_family(
    real_pairs: pd.DataFrame,
    *,
    base_patient_id: str | int,
    relative_dil_index: str | int,
    prefer_same_fraction: bool = True,
) -> pd.Series:
    sub = real_pairs[
        (real_pairs["Base patient ID"].astype(str) == str(base_patient_id))
        & (
            pd.to_numeric(real_pairs["Relative DIL index"], errors="coerce")
            == pd.to_numeric(pd.Series([relative_dil_index]), errors="coerce").iloc[0]
        )
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No real-core rows found for family {base_patient_id}::{relative_dil_index}"
        )
    if prefer_same_fraction and "same_fraction_with_references" in sub.columns:
        same_fraction_sub = sub[sub["same_fraction_with_references"]].copy()
        if not same_fraction_sub.empty:
            sub = same_fraction_sub
    sort_cols = [
        "delta_optimal_minus_real__DIL Global Mean BE",
        "delta_optimal_minus_real__DIL Global Max BE",
        "Patient ID",
        "Bx ID",
    ]
    ascending = [False, False, True, True]
    return sub.sort_values(sort_cols, ascending=ascending).iloc[0]


def build_selected_profile_cases_df(
    family_outputs: QAFamilyOutputs,
    reference_disagreement_df: pd.DataFrame,
) -> pd.DataFrame:
    real_pairs = family_outputs.qa_real_core_pairs.copy()
    real_pairs["same_fraction_with_references"] = (
        real_pairs["Patient ID"].astype(str) == real_pairs["centroid_Patient ID"].astype(str)
    ) & (
        real_pairs["Patient ID"].astype(str) == real_pairs["optimal_Patient ID"].astype(str)
    )
    ref_sub = reference_disagreement_df[
        reference_disagreement_df["metric"] == "DIL Global Mean BE"
    ].copy()
    if ref_sub.empty:
        return pd.DataFrame()

    selected_rows: list[dict[str, object]] = []
    used_families: set[str] = set()

    def take_family_from_ref(sort_cols, ascending, selection_key: str) -> None:
        candidates = ref_sub.loc[~ref_sub["Family ID"].isin(used_families)].sort_values(
            sort_cols,
            ascending=ascending,
        )
        if candidates.empty:
            return
        family_row = candidates.iloc[0]
        real_row = _pick_real_core_for_family(
            real_pairs,
            base_patient_id=family_row["Base patient ID"],
            relative_dil_index=family_row["Relative DIL index"],
            prefer_same_fraction=True,
        )
        used_families.add(str(family_row["Family ID"]))
        selected_rows.append(
            {
                "Selection key": selection_key,
                "Selection reason": PROFILE_SELECTION_REASON_MAP[selection_key],
                "Base patient ID": real_row["Base patient ID"],
                "Relative DIL index": real_row["Relative DIL index"],
                "Family ID": real_row["Family ID"],
                "Patient ID": real_row["Patient ID"],
                "Bx ID": real_row["Bx ID"],
                "Bx index": int(real_row["Bx index"]),
                "centroid_Patient ID": real_row["centroid_Patient ID"],
                "centroid_Bx ID": real_row["centroid_Bx ID"],
                "centroid_Bx index": int(real_row["centroid_Bx index"]),
                "optimal_Patient ID": real_row["optimal_Patient ID"],
                "optimal_Bx ID": real_row["optimal_Bx ID"],
                "optimal_Bx index": int(real_row["optimal_Bx index"]),
                "real_DIL Global Mean BE": real_row["DIL Global Mean BE"],
                "centroid_DIL Global Mean BE": real_row["centroid_DIL Global Mean BE"],
                "optimal_DIL Global Mean BE": real_row["optimal_DIL Global Mean BE"],
                "delta_centroid_minus_real__DIL Global Mean BE": real_row[
                    "delta_centroid_minus_real__DIL Global Mean BE"
                ],
                "delta_optimal_minus_real__DIL Global Mean BE": real_row[
                    "delta_optimal_minus_real__DIL Global Mean BE"
                ],
                "delta_optimal_minus_centroid__DIL Global Mean BE": (
                    float(real_row["optimal_DIL Global Mean BE"])
                    - float(real_row["centroid_DIL Global Mean BE"])
                ),
                "Family_real_core_count": real_row["Family_real_core_count"],
            }
        )

    take_family_from_ref(
        ["delta_value", "abs_delta_value", "Family ID"],
        [False, False, True],
        "largest_positive_optimal_minus_centroid",
    )
    take_family_from_ref(
        ["delta_value", "abs_delta_value", "Family ID"],
        [True, False, True],
        "largest_negative_optimal_minus_centroid",
    )

    headroom_candidate_pool = real_pairs.loc[~real_pairs["Family ID"].isin(used_families)].copy()
    same_fraction_pool = headroom_candidate_pool[headroom_candidate_pool["same_fraction_with_references"]].copy()
    if not same_fraction_pool.empty:
        headroom_candidate_pool = same_fraction_pool
    headroom_candidates = headroom_candidate_pool.sort_values(
        [
            "delta_optimal_minus_real__DIL Global Mean BE",
            "delta_optimal_minus_real__DIL Global Max BE",
            "Family ID",
        ],
        ascending=[False, False, True],
    )
    if not headroom_candidates.empty:
        real_row = headroom_candidates.iloc[0]
        used_families.add(str(real_row["Family ID"]))
        selected_rows.append(
            {
                "Selection key": "largest_total_headroom",
                "Selection reason": PROFILE_SELECTION_REASON_MAP["largest_total_headroom"],
                "Base patient ID": real_row["Base patient ID"],
                "Relative DIL index": real_row["Relative DIL index"],
                "Family ID": real_row["Family ID"],
                "Patient ID": real_row["Patient ID"],
                "Bx ID": real_row["Bx ID"],
                "Bx index": int(real_row["Bx index"]),
                "centroid_Patient ID": real_row["centroid_Patient ID"],
                "centroid_Bx ID": real_row["centroid_Bx ID"],
                "centroid_Bx index": int(real_row["centroid_Bx index"]),
                "optimal_Patient ID": real_row["optimal_Patient ID"],
                "optimal_Bx ID": real_row["optimal_Bx ID"],
                "optimal_Bx index": int(real_row["optimal_Bx index"]),
                "real_DIL Global Mean BE": real_row["DIL Global Mean BE"],
                "centroid_DIL Global Mean BE": real_row["centroid_DIL Global Mean BE"],
                "optimal_DIL Global Mean BE": real_row["optimal_DIL Global Mean BE"],
                "delta_centroid_minus_real__DIL Global Mean BE": real_row[
                    "delta_centroid_minus_real__DIL Global Mean BE"
                ],
                "delta_optimal_minus_real__DIL Global Mean BE": real_row[
                    "delta_optimal_minus_real__DIL Global Mean BE"
                ],
                "delta_optimal_minus_centroid__DIL Global Mean BE": (
                    float(real_row["optimal_DIL Global Mean BE"])
                    - float(real_row["centroid_DIL Global Mean BE"])
                ),
                "Family_real_core_count": real_row["Family_real_core_count"],
            }
        )

    take_family_from_ref(
        ["abs_delta_value", "Family ID"],
        [True, True],
        "representative_small_optimal_minus_centroid",
    )

    out = pd.DataFrame(selected_rows)
    if out.empty:
        return out

    selection_order_map = {key: idx for idx, key in enumerate(PROFILE_SELECTION_ORDER)}
    out["Selection order"] = out["Selection key"].map(selection_order_map)
    out = out.sort_values(["Selection order", "Family ID", "Patient ID", "Bx ID"]).reset_index(drop=True)
    out["Biopsy heading"] = [f"Biopsy {_alpha_code(i)}" for i in range(len(out))]
    return out


def build_selected_profile_long_df(
    source_tables: QASourceTables,
    selected_cases_df: pd.DataFrame,
) -> pd.DataFrame:
    if selected_cases_df.empty:
        return pd.DataFrame()

    voxel_df = source_tables.cohort_sum_to_one_mc_results_df.copy()
    voxel_df = voxel_df[voxel_df["Tissue class"].astype(str) == "DIL"].copy()
    if voxel_df.empty:
        return pd.DataFrame()

    for col in [
        "Bx index",
        "Voxel index",
        "Voxel begin (Z)",
        "Voxel end (Z)",
        "Nominal",
        "Binomial estimator",
        "CI lower vals",
        "CI upper vals",
    ]:
        if col in voxel_df.columns:
            voxel_df[col] = pd.to_numeric(voxel_df[col], errors="coerce")

    voxel_df["Voxel mid Z"] = 0.5 * (voxel_df["Voxel begin (Z)"] + voxel_df["Voxel end (Z)"])

    rows: list[pd.DataFrame] = []
    family_specs = [
        ("Real", "Patient ID", "Bx ID", "Bx index", 0),
        ("Centroid", "centroid_Patient ID", "centroid_Bx ID", "centroid_Bx index", 1),
        ("Optimal", "optimal_Patient ID", "optimal_Bx ID", "optimal_Bx index", 2),
    ]

    for _, case in selected_cases_df.iterrows():
        for family_name, patient_col, bx_id_col, bx_index_col, family_order in family_specs:
            sub = voxel_df[
                (voxel_df["Patient ID"].astype(str) == str(case[patient_col]))
                & (pd.to_numeric(voxel_df["Bx index"], errors="coerce") == int(case[bx_index_col]))
                & (voxel_df["Bx ID"].astype(str) == str(case[bx_id_col]))
            ].copy()
            if sub.empty:
                continue
            sub["Selection key"] = case["Selection key"]
            sub["Selection reason"] = case["Selection reason"]
            sub["Selection order"] = case["Selection order"]
            sub["Biopsy heading"] = case["Biopsy heading"]
            sub["Family ID"] = case["Family ID"]
            sub["Base patient ID"] = case["Base patient ID"]
            sub["Relative DIL index"] = case["Relative DIL index"]
            sub["Selected real Patient ID"] = case["Patient ID"]
            sub["Selected real Bx ID"] = case["Bx ID"]
            sub["Selected real Bx index"] = case["Bx index"]
            sub["Family group"] = family_name
            sub["Family order"] = family_order
            sub["Panel delta_centroid_minus_real__DIL Global Mean BE"] = case[
                "delta_centroid_minus_real__DIL Global Mean BE"
            ]
            sub["Panel delta_optimal_minus_real__DIL Global Mean BE"] = case[
                "delta_optimal_minus_real__DIL Global Mean BE"
            ]
            sub["Panel delta_optimal_minus_centroid__DIL Global Mean BE"] = case[
                "delta_optimal_minus_centroid__DIL Global Mean BE"
            ]
            rows.append(sub)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(
        ["Selection order", "Family order", "Voxel index"]
    ).reset_index(drop=True)


def build_family_optimizer_difficulty_df(
    family_outputs: QAFamilyOutputs,
) -> pd.DataFrame:
    ref_pairs = family_outputs.qa_family_reference_pairs.copy()
    real_agg = family_outputs.qa_family_real_aggregated.copy()

    family_context_source = family_outputs.qa_family_members_long.copy()
    centroid_context = family_context_source[
        family_context_source["Simulated type"].astype(str) == "Centroid DIL"
    ].copy()
    if centroid_context.empty:
        centroid_context = family_context_source.drop_duplicates(
            subset=["Base patient ID", "Relative DIL index"],
            keep="first",
        ).copy()

    context_cols = [
        "Base patient ID",
        "Relative DIL index",
        "Family ID",
        "Family_real_core_count",
        "DIL Volume",
        "DIL Maximum 3D diameter",
        "DIL Elongation",
        "DIL Flatness",
        "Prostate Volume",
        "DIL DIL prostate sextant (LR)",
        "DIL DIL prostate sextant (AP)",
        "DIL DIL prostate sextant (SI)",
        "DIL DIL centroid (X, prostate frame)",
        "DIL DIL centroid (Y, prostate frame)",
        "DIL DIL centroid (Z, prostate frame)",
    ]
    context_cols = [col for col in context_cols if col in centroid_context.columns]
    family_context = centroid_context[context_cols].drop_duplicates(
        subset=["Base patient ID", "Relative DIL index"],
        keep="first",
    )

    out = ref_pairs.merge(
        real_agg,
        on=["Base patient ID", "Relative DIL index"],
        how="left",
        validate="1:1",
    ).merge(
        family_context,
        on=["Base patient ID", "Relative DIL index"],
        how="left",
        validate="1:1",
    )

    out["delta_centroid_minus_real_mean__DIL Global Mean BE"] = (
        out["centroid_DIL Global Mean BE"] - out["real_mean__DIL Global Mean BE"]
    )
    out["delta_optimal_minus_real_mean__DIL Global Mean BE"] = (
        out["optimal_DIL Global Mean BE"] - out["real_mean__DIL Global Mean BE"]
    )
    out["optimizer_gain_mean__DIL Global Mean BE"] = (
        out["delta_optimal_minus_centroid__DIL Global Mean BE"]
    )
    out["optimizer_gain_abs_mean__DIL Global Mean BE"] = (
        out["optimizer_gain_mean__DIL Global Mean BE"].abs()
    )

    lr_map = {"Left": "L", "Right": "R"}
    ap_map = {"Posterior": "P", "Anterior": "A"}
    si_map = {
        "Apex (Inferior)": "Apex",
        "Mid": "Mid",
        "Base (Superior)": "Base",
    }
    out["DIL LR short"] = out.get("DIL DIL prostate sextant (LR)", pd.Series(index=out.index)).map(lr_map)
    out["DIL AP short"] = out.get("DIL DIL prostate sextant (AP)", pd.Series(index=out.index)).map(ap_map)
    out["DIL SI short"] = out.get("DIL DIL prostate sextant (SI)", pd.Series(index=out.index)).map(si_map)
    out["DIL double sextant zone"] = (
        out["DIL LR short"].fillna("?") + out["DIL AP short"].fillna("?")
    )

    if "Family ID" not in out.columns:
        out["Family ID"] = (
            out["Base patient ID"].astype(str)
            + "::"
            + out["Relative DIL index"].astype("Int64").astype(str)
        )

    keep_order = [
        "Base patient ID",
        "Relative DIL index",
        "Family ID",
        "Family_real_core_count",
        "centroid_DIL Global Mean BE",
        "optimal_DIL Global Mean BE",
        "real_mean__DIL Global Mean BE",
        "delta_centroid_minus_real_mean__DIL Global Mean BE",
        "delta_optimal_minus_real_mean__DIL Global Mean BE",
        "optimizer_gain_mean__DIL Global Mean BE",
        "optimizer_gain_abs_mean__DIL Global Mean BE",
        "DIL Volume",
        "DIL Maximum 3D diameter",
        "DIL Elongation",
        "DIL Flatness",
        "Prostate Volume",
        "DIL DIL prostate sextant (LR)",
        "DIL DIL prostate sextant (AP)",
        "DIL DIL prostate sextant (SI)",
        "DIL LR short",
        "DIL AP short",
        "DIL SI short",
        "DIL double sextant zone",
        "DIL DIL centroid (X, prostate frame)",
        "DIL DIL centroid (Y, prostate frame)",
        "DIL DIL centroid (Z, prostate frame)",
        "centroid_BX to DIL centroid distance",
        "optimal_BX to DIL centroid distance",
    ]
    keep_order = [col for col in keep_order if col in out.columns]
    return out[keep_order].sort_values(
        ["optimizer_gain_abs_mean__DIL Global Mean BE", "Family ID"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_qa_plot_data_outputs(
    source_tables: QASourceTables,
    family_outputs: QAFamilyOutputs,
    stats_outputs: QAStatsOutputs,
) -> QAPlotDataOutputs:
    reference_disagreement_df = build_reference_disagreement_df(family_outputs)
    selected_profile_cases_df = build_selected_profile_cases_df(
        family_outputs,
        reference_disagreement_df,
    )
    return QAPlotDataOutputs(
        qa_plot_family_comparison_long=build_family_comparison_long_df(family_outputs),
        qa_plot_headroom_long=build_headroom_long_df(stats_outputs),
        qa_plot_reference_disagreement=reference_disagreement_df,
        qa_plot_safety_distance_long=build_safety_distance_long_df(family_outputs),
        qa_plot_selected_profile_long=build_selected_profile_long_df(
            source_tables,
            selected_profile_cases_df,
        ),
        qa_selected_profile_cases=selected_profile_cases_df,
        qa_family_optimizer_difficulty=build_family_optimizer_difficulty_df(family_outputs),
    )
