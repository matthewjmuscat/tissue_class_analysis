from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qa.families import QAFamilyOutputs
from qa.load import QASourceTables
from qa.notation import metric_math
from qa.plot_data import CONTRAST_LABEL_MAP, QAPlotDataOutputs
from qa.stats import QAStatsOutputs


PRIMARY_METRIC_ORDER = [
    "DIL Global Mean BE",
    "DIL Global Max BE",
]

SAFETY_METRIC_ORDER = [
    "Urethra NN dist mean",
    "Rectum NN dist mean",
]

CONTRAST_ORDER = [
    "centroid_minus_real",
    "optimal_minus_real",
    "optimal_minus_centroid",
]


@dataclass
class QADeliverableOutputs:
    cohort_overview_table: pd.DataFrame
    primary_headroom_table: pd.DataFrame
    safety_distance_table: pd.DataFrame
    targeting_feature_ranking_table: pd.DataFrame
    targeting_location_summary_table: pd.DataFrame
    biopsy_case_catalog_table: pd.DataFrame
    geometric_biopsy_level_table: pd.DataFrame
    geometric_biopsy_level_summary_table: pd.DataFrame
    geometric_voxelwise_table: pd.DataFrame
    geometric_voxelwise_group_summary_table: pd.DataFrame


def _rename_for_manuscript_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["outcome_symbol"] = out["metric"].map(metric_math).fillna(out["metric"])
    out["contrast_label"] = out["contrast_key"].map(CONTRAST_LABEL_MAP).fillna(out["contrast_key"])
    out["outcome_order"] = out["metric"].map({m: i for i, m in enumerate(PRIMARY_METRIC_ORDER)})
    out["contrast_order"] = out["contrast_key"].map(
        {contrast: i for i, contrast in enumerate(CONTRAST_ORDER)}
    )
    out = out.rename(
        columns={
            "metric": "outcome_name",
            "contrast_key": "contrast_key",
            "group_a_name": "group_a_name",
            "group_b_name": "group_b_name",
            "n_rows": "n_observations",
            "n_clusters": "n_patients",
            "group_a_mean": "group_a_mean",
            "group_b_mean": "group_b_mean",
            "observed_mean_delta": "mean_delta",
            "bootstrap_ci_lower": "ci_lower_95",
            "bootstrap_ci_upper": "ci_upper_95",
            "bootstrap_p_two_sided": "bootstrap_p_value",
            "standardized_mean_delta": "standardized_mean_delta",
            "supports_nonzero_ci": "ci_excludes_zero",
            "significance_label": "significance_label",
        }
    )
    keep_cols = [
        "outcome_name",
        "outcome_symbol",
        "contrast_key",
        "contrast_label",
        "group_a_name",
        "group_b_name",
        "n_observations",
        "n_patients",
        "group_a_mean",
        "group_b_mean",
        "mean_delta",
        "ci_lower_95",
        "ci_upper_95",
        "bootstrap_p_value",
        "standardized_mean_delta",
        "ci_excludes_zero",
        "significance_label",
    ]
    return out.sort_values(["outcome_order", "contrast_order"])[keep_cols].reset_index(drop=True)


def build_cohort_overview_table(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    qa_long = family_outputs.qa_family_members_long
    audit = family_outputs.qa_family_audit
    real_counts = audit["Family_real_core_count"].value_counts(dropna=False).to_dict()

    rows = [
        ("n_base_patients", int(qa_long["Base patient ID"].nunique())),
        ("n_fraction_labeled_patient_ids", int(qa_long["Patient ID"].nunique())),
        ("n_lesion_families", int(len(audit))),
        ("n_complete_families", int(audit["Family_complete"].sum())),
        ("n_total_family_members", int(len(qa_long))),
        ("n_real_cores", int((qa_long["Simulated type"] == "Real").sum())),
        ("n_centroid_reference_cores", int((qa_long["Simulated type"] == "Centroid DIL").sum())),
        ("n_optimal_reference_cores", int((qa_long["Simulated type"] == "Optimal DIL").sum())),
        ("n_families_with_1_real_core", int(real_counts.get(1, 0))),
        ("n_families_with_2_real_cores", int(real_counts.get(2, 0))),
        ("n_families_with_3_real_cores", int(real_counts.get(3, 0))),
    ]
    return pd.DataFrame(rows, columns=["summary_item", "value"])


def build_primary_headroom_table(stats_outputs: QAStatsOutputs) -> pd.DataFrame:
    summary = stats_outputs.headline_bootstrap_summary_df.copy()
    summary = summary[summary["metric"].isin(PRIMARY_METRIC_ORDER)].copy()
    return _rename_for_manuscript_summary(summary)


def build_safety_distance_table(stats_outputs: QAStatsOutputs) -> pd.DataFrame:
    summary = stats_outputs.safety_bootstrap_summary_df.copy()
    summary["outcome_order"] = summary["metric"].map({m: i for i, m in enumerate(SAFETY_METRIC_ORDER)})
    summary["contrast_order"] = summary["contrast_key"].map(
        {contrast: i for i, contrast in enumerate(CONTRAST_ORDER)}
    )
    out = summary.rename(
        columns={
            "metric": "outcome_name",
            "contrast_key": "contrast_key",
            "group_a_name": "group_a_name",
            "group_b_name": "group_b_name",
            "n_rows": "n_observations",
            "n_clusters": "n_patients",
            "group_a_mean": "group_a_mean",
            "group_b_mean": "group_b_mean",
            "observed_mean_delta": "mean_delta",
            "bootstrap_ci_lower": "ci_lower_95",
            "bootstrap_ci_upper": "ci_upper_95",
            "bootstrap_p_two_sided": "bootstrap_p_value",
            "standardized_mean_delta": "standardized_mean_delta",
            "supports_nonzero_ci": "ci_excludes_zero",
            "significance_label": "significance_label",
        }
    )
    out["outcome_symbol"] = out["outcome_name"].map(metric_math).fillna(out["outcome_name"])
    out["contrast_label"] = out["contrast_key"].map(CONTRAST_LABEL_MAP).fillna(out["contrast_key"])
    keep_cols = [
        "outcome_name",
        "outcome_symbol",
        "contrast_key",
        "contrast_label",
        "group_a_name",
        "group_b_name",
        "n_observations",
        "n_patients",
        "group_a_mean",
        "group_b_mean",
        "mean_delta",
        "ci_lower_95",
        "ci_upper_95",
        "bootstrap_p_value",
        "standardized_mean_delta",
        "ci_excludes_zero",
        "significance_label",
    ]
    return out.sort_values(["outcome_order", "contrast_order"])[keep_cols].reset_index(drop=True)


def build_targeting_feature_ranking_table(plot_data_outputs: QAPlotDataOutputs) -> pd.DataFrame:
    out = plot_data_outputs.qa_targeting_difficulty_correlations.copy()
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    out["rank_within_outcome"] = (
        out.groupby("outcome_key")["abs_spearman_rho"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out.sort_values(
        ["outcome_key", "rank_within_outcome", "feature_col"]
    ).reset_index(drop=True)


def build_targeting_location_summary_table(plot_data_outputs: QAPlotDataOutputs) -> pd.DataFrame:
    out = plot_data_outputs.qa_targeting_difficulty_group_summary.copy()
    return out.sort_values(["outcome_key", "group_col", "category"]).reset_index(drop=True)


def build_biopsy_case_catalog_table(
    family_outputs: QAFamilyOutputs,
    plot_data_outputs: QAPlotDataOutputs,
) -> pd.DataFrame:
    disagreement = plot_data_outputs.qa_plot_reference_disagreement.copy()
    disagreement = disagreement[
        (disagreement["metric"] == "DIL Global Mean BE")
        & disagreement["Family figure label"].astype(str).str.startswith("Biopsy ")
    ].copy()
    disagreement = disagreement.sort_values(["abs_delta_rank", "Family ID"]).drop_duplicates(
        subset=["Family ID"],
        keep="first",
    )

    context_cols = [
        "Base patient ID",
        "Relative DIL index",
        "Family ID",
        "DIL Volume",
        "DIL Maximum 3D diameter",
        "DIL DIL centroid (Y, prostate frame)",
        "DIL DIL centroid (Z, prostate frame)",
        "DIL DIL prostate sextant (LR)",
        "DIL DIL prostate sextant (AP)",
        "DIL DIL prostate sextant (SI)",
        "DIL double sextant zone",
        "DIL sextant zone 12",
        "Prostate Volume",
        "delta_centroid_minus_real_mean__DIL Global Mean BE",
        "delta_optimal_minus_real_mean__DIL Global Mean BE",
        "optimizer_gain_mean__DIL Global Mean BE",
    ]
    family_context = plot_data_outputs.qa_family_optimizer_difficulty[
        [col for col in context_cols if col in plot_data_outputs.qa_family_optimizer_difficulty.columns]
    ].drop_duplicates(subset=["Family ID"], keep="first")

    selected_cols = [
        "Family ID",
        "Selection reason",
        "Patient ID",
        "Bx ID",
        "real_DIL Global Mean BE",
        "delta_centroid_minus_real__DIL Global Mean BE",
        "delta_optimal_minus_real__DIL Global Mean BE",
    ]
    selected_cases = plot_data_outputs.qa_selected_profile_cases[
        [col for col in selected_cols if col in plot_data_outputs.qa_selected_profile_cases.columns]
    ].drop_duplicates(subset=["Family ID"], keep="first")

    out = disagreement.merge(family_context, on="Family ID", how="left", validate="1:1")
    out = out.merge(selected_cases, on="Family ID", how="left", validate="1:1")
    out = out.rename(
        columns={
            "Family figure label": "biopsy_label",
            "Family figure label source": "label_source",
            "Relative_DIL_IDs": "relative_dil_ids",
            "Family_real_core_count": "family_real_core_count",
            "centroid_value": "centroid_mean_be",
            "optimal_value": "optimal_mean_be",
            "delta_value": "delta_optimal_minus_centroid_mean_be",
            "centroid_BX to DIL centroid distance": "centroid_bx_to_dil_centroid_distance_mm",
            "optimal_BX to DIL centroid distance": "optimal_bx_to_dil_centroid_distance_mm",
            "centroid_Urethra NN dist mean": "centroid_urethra_nn_dist_mean_mm",
            "optimal_Urethra NN dist mean": "optimal_urethra_nn_dist_mean_mm",
            "centroid_Rectum NN dist mean": "centroid_rectum_nn_dist_mean_mm",
            "optimal_Rectum NN dist mean": "optimal_rectum_nn_dist_mean_mm",
            "DIL Volume": "dil_volume_mm3",
            "DIL Maximum 3D diameter": "dil_maximum_3d_diameter_mm",
            "DIL DIL centroid (Y, prostate frame)": "dil_centroid_y_prostate_frame_mm",
            "DIL DIL centroid (Z, prostate frame)": "dil_centroid_z_prostate_frame_mm",
            "DIL DIL prostate sextant (LR)": "dil_lr_sextant",
            "DIL DIL prostate sextant (AP)": "dil_ap_sextant",
            "DIL DIL prostate sextant (SI)": "dil_si_sextant",
            "DIL double sextant zone": "dil_double_sextant_zone",
            "DIL sextant zone 12": "dil_sextant_zone_12",
            "Prostate Volume": "prostate_volume_mm3",
            "delta_centroid_minus_real_mean__DIL Global Mean BE": "delta_centroid_minus_real_mean_be",
            "delta_optimal_minus_real_mean__DIL Global Mean BE": "delta_optimal_minus_real_mean_be",
            "optimizer_gain_mean__DIL Global Mean BE": "optimizer_gain_mean_be",
            "Selection reason": "selected_profile_reason",
            "Patient ID": "selected_real_patient_id",
            "Bx ID": "selected_real_bx_id",
            "real_DIL Global Mean BE": "selected_real_mean_be",
            "delta_centroid_minus_real__DIL Global Mean BE": "selected_delta_centroid_minus_real_mean_be",
            "delta_optimal_minus_real__DIL Global Mean BE": "selected_delta_optimal_minus_real_mean_be",
        }
    )
    keep_cols = [
        "biopsy_label",
        "label_source",
        "selected_profile_reason",
        "Family ID",
        "relative_dil_ids",
        "family_real_core_count",
        "dil_volume_mm3",
        "dil_maximum_3d_diameter_mm",
        "dil_centroid_y_prostate_frame_mm",
        "dil_centroid_z_prostate_frame_mm",
        "dil_lr_sextant",
        "dil_ap_sextant",
        "dil_si_sextant",
        "dil_double_sextant_zone",
        "dil_sextant_zone_12",
        "prostate_volume_mm3",
        "selected_real_patient_id",
        "selected_real_bx_id",
        "selected_real_mean_be",
        "centroid_mean_be",
        "optimal_mean_be",
        "selected_delta_centroid_minus_real_mean_be",
        "selected_delta_optimal_minus_real_mean_be",
        "delta_optimal_minus_centroid_mean_be",
        "centroid_bx_to_dil_centroid_distance_mm",
        "optimal_bx_to_dil_centroid_distance_mm",
        "centroid_urethra_nn_dist_mean_mm",
        "optimal_urethra_nn_dist_mean_mm",
        "centroid_rectum_nn_dist_mean_mm",
        "optimal_rectum_nn_dist_mean_mm",
        "delta_centroid_minus_real_mean_be",
        "delta_optimal_minus_real_mean_be",
        "optimizer_gain_mean_be",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    return out[keep_cols].sort_values("biopsy_label").reset_index(drop=True)


def build_geometric_biopsy_level_table(family_outputs: QAFamilyOutputs) -> pd.DataFrame:
    out = family_outputs.qa_family_members_long.copy()
    rename_map = {
        "Base patient ID": "base_patient_id",
        "Patient ID": "patient_id",
        "Family ID": "family_id",
        "Family member label": "family_member_label",
        "Family_real_core_count": "family_real_core_count",
        "Family_complete": "family_complete",
        "Simulated bool": "simulated_bool",
        "Simulated type": "simulated_type",
        "Bx ID": "bx_id",
        "Bx index": "bx_index",
        "Relative DIL ID": "relative_dil_id",
        "Relative DIL index": "relative_dil_index",
        "Length (mm)": "length_mm",
        "Volume (mm3)": "biopsy_volume_mm3",
        "Voxel side length (mm)": "voxel_side_length_mm",
        "BX to DIL centroid (X)": "bx_to_dil_centroid_x_mm",
        "BX to DIL centroid (Y)": "bx_to_dil_centroid_y_mm",
        "BX to DIL centroid (Z)": "bx_to_dil_centroid_z_mm",
        "BX to DIL centroid distance": "bx_to_dil_centroid_distance_mm",
        "NN surface-surface distance": "bx_to_dil_surface_nn_mm",
        "DIL NN dist mean": "dil_boundary_nn_mean_mm",
        "DIL centroid dist mean": "dil_centroid_distance_mean_mm",
        "Prostate NN dist mean": "prostate_boundary_nn_mean_mm",
        "Prostate centroid dist mean": "prostate_centroid_distance_mean_mm",
        "Urethra NN dist mean": "urethra_boundary_nn_mean_mm",
        "Urethra centroid dist mean": "urethra_centroid_distance_mean_mm",
        "Rectum NN dist mean": "rectum_boundary_nn_mean_mm",
        "Rectum centroid dist mean": "rectum_centroid_distance_mean_mm",
        "BX_to_prostate_centroid_distance_norm_mean_dim": "bx_to_prostate_centroid_distance_norm_mean_dim",
        "Bx position in prostate LR": "bx_position_in_prostate_lr",
        "Bx position in prostate AP": "bx_position_in_prostate_ap",
        "Bx position in prostate SI": "bx_position_in_prostate_si",
        "DIL Global Mean BE": "dil_mean_be",
        "DIL Global Max BE": "dil_max_be",
        "DIL Global Q50 BE": "dil_q50_be",
        "Prostatic Global Mean BE": "prostatic_mean_be",
        "Periprostatic Global Mean BE": "periprostatic_mean_be",
        "Urethral Global Mean BE": "urethral_mean_be",
        "Rectal Global Mean BE": "rectal_mean_be",
        "DIL Volume": "dil_volume_mm3",
        "DIL Maximum 3D diameter": "dil_maximum_3d_diameter_mm",
        "DIL DIL centroid (X, prostate frame)": "dil_centroid_x_prostate_frame_mm",
        "DIL DIL centroid (Y, prostate frame)": "dil_centroid_y_prostate_frame_mm",
        "DIL DIL centroid (Z, prostate frame)": "dil_centroid_z_prostate_frame_mm",
        "DIL DIL centroid distance (prostate frame)": "dil_centroid_distance_prostate_frame_mm",
        "DIL DIL prostate sextant (LR)": "dil_lr_sextant",
        "DIL DIL prostate sextant (AP)": "dil_ap_sextant",
        "DIL DIL prostate sextant (SI)": "dil_si_sextant",
        "Prostate Volume": "prostate_volume_mm3",
    }
    keep_cols = [col for col in rename_map if col in out.columns]
    out = out[keep_cols].rename(columns={col: rename_map[col] for col in keep_cols})
    sort_cols = [col for col in ["base_patient_id", "relative_dil_index", "simulated_type", "patient_id", "bx_index"] if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_geometric_biopsy_level_summary_table(geometric_biopsy_level_table: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "bx_to_dil_centroid_distance_mm",
        "bx_to_dil_surface_nn_mm",
        "dil_boundary_nn_mean_mm",
        "urethra_boundary_nn_mean_mm",
        "rectum_boundary_nn_mean_mm",
        "dil_mean_be",
        "dil_max_be",
    ]
    rows: list[dict[str, object]] = []
    for family_member_label, sub in geometric_biopsy_level_table.groupby("family_member_label", dropna=False):
        for metric in metrics:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "family_member_label": family_member_label,
                    "metric": metric,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "family_member_label"]).reset_index(drop=True)


def _clean_distance_export_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Patient ID"].notna()].copy()

    rename_map = {
        "Patient ID": "patient_id",
        "Bx ID": "bx_id",
        "Bx index": "bx_index",
        "Simulated bool": "simulated_bool",
        "Simulated type": "simulated_type",
        "Relative structure ROI": "relative_structure_roi",
        "Relative structure type": "relative_structure_type",
        "Relative structure index": "relative_structure_index",
        "Voxel index": "voxel_index",
        "Voxel begin (Z)": "voxel_begin_z_mm",
        "Voxel end (Z)": "voxel_end_z_mm",
    }

    stat_suffix_map = {
        "": "count",
        ".1": "mean",
        ".2": "std",
        ".3": "min",
        ".4": "q05",
        ".5": "q25",
        ".6": "q50",
        ".7": "q75",
        ".8": "q95",
        ".9": "max",
    }
    for base_col, stem in {
        "Struct. boundary NN dist.": "boundary_nn_dist",
        "Dist. from struct. centroid": "centroid_dist",
    }.items():
        for suffix, stat in stat_suffix_map.items():
            source_col = f"{base_col}{suffix}"
            if source_col in out.columns:
                rename_map[source_col] = f"{stem}_{stat}_mm"

    keep_cols = [col for col in rename_map if col in out.columns]
    out = out[keep_cols].rename(columns={col: rename_map[col] for col in keep_cols})

    numeric_cols = [col for col in out.columns if col not in {"patient_id", "bx_id", "simulated_bool", "simulated_type", "relative_structure_roi", "relative_structure_type"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["voxel_mid_z_mm"] = 0.5 * (out["voxel_begin_z_mm"] + out["voxel_end_z_mm"])
    sort_cols = [
        "patient_id",
        "bx_index",
        "relative_structure_type",
        "relative_structure_index",
        "voxel_index",
    ]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_geometric_voxelwise_table(source_tables: QASourceTables) -> pd.DataFrame:
    return _clean_distance_export_table(source_tables.cohort_tissue_class_distances_voxelwise_df)


def build_geometric_voxelwise_group_summary_table(
    geometric_voxelwise_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["simulated_type", "relative_structure_type"]
    metric_cols = [
        "boundary_nn_dist_mean_mm",
        "centroid_dist_mean_mm",
    ]
    for group_key, sub in geometric_voxelwise_table.groupby(group_cols, dropna=False):
        simulated_type, relative_structure_type = group_key
        for metric in metric_cols:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "simulated_type": simulated_type,
                    "relative_structure_type": relative_structure_type,
                    "metric": metric,
                    "n_rows": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "q95": float(values.quantile(0.95)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["relative_structure_type", "metric", "simulated_type"]
    ).reset_index(drop=True)


def build_qa_deliverable_outputs(
    source_tables: QASourceTables,
    family_outputs: QAFamilyOutputs,
    stats_outputs: QAStatsOutputs,
    plot_data_outputs: QAPlotDataOutputs,
) -> QADeliverableOutputs:
    geometric_biopsy_level_table = build_geometric_biopsy_level_table(family_outputs)
    geometric_voxelwise_table = build_geometric_voxelwise_table(source_tables)
    return QADeliverableOutputs(
        cohort_overview_table=build_cohort_overview_table(family_outputs),
        primary_headroom_table=build_primary_headroom_table(stats_outputs),
        safety_distance_table=build_safety_distance_table(stats_outputs),
        targeting_feature_ranking_table=build_targeting_feature_ranking_table(plot_data_outputs),
        targeting_location_summary_table=build_targeting_location_summary_table(plot_data_outputs),
        biopsy_case_catalog_table=build_biopsy_case_catalog_table(family_outputs, plot_data_outputs),
        geometric_biopsy_level_table=geometric_biopsy_level_table,
        geometric_biopsy_level_summary_table=build_geometric_biopsy_level_summary_table(
            geometric_biopsy_level_table
        ),
        geometric_voxelwise_table=geometric_voxelwise_table,
        geometric_voxelwise_group_summary_table=build_geometric_voxelwise_group_summary_table(
            geometric_voxelwise_table
        ),
    )
