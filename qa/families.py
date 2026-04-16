from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qa.endpoints import PAIR_METRIC_COLUMNS
from qa.load import QASourceTables


KEY_COLS = ["Patient ID", "Bx ID", "Bx index"]
FAMILY_KEY_COLS = ["Base patient ID", "Relative DIL index"]


@dataclass
class QAFamilyOutputs:
    qa_family_members_long: pd.DataFrame
    qa_family_audit: pd.DataFrame
    qa_real_core_pairs: pd.DataFrame
    qa_family_reference_pairs: pd.DataFrame
    qa_family_real_aggregated: pd.DataFrame


def _extract_base_patient_id(patient_id: pd.Series) -> pd.Series:
    extracted = patient_id.astype(str).str.extract(r"^(\d+)")[0]
    return extracted.fillna(patient_id.astype(str))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()

    flat_cols: list[str] = []
    for col in df.columns:
        parts = [str(part).strip() for part in col if str(part).strip() and str(part).lower() != "nan"]
        flat_cols.append(" ".join(parts))
    out = df.copy()
    out.columns = flat_cols
    return out


def _build_global_tissue_wide(global_sum_to_one_df: pd.DataFrame) -> pd.DataFrame:
    candidate_be_cols = [
        "Global Mean BE",
        "Global Min BE",
        "Global Max BE",
        "Global STD BE",
        "Global SEM BE",
        "Global Q05 BE",
        "Global Q25 BE",
        "Global Q50 BE",
        "Global Q75 BE",
        "Global Q95 BE",
        "Global CI 95 BE (lower)",
        "Global CI 95 BE (upper)",
    ]
    be_cols = [c for c in candidate_be_cols if c in global_sum_to_one_df.columns]
    subset = global_sum_to_one_df[KEY_COLS + ["Tissue class"] + be_cols].copy()
    subset = subset.groupby(KEY_COLS + ["Tissue class"], as_index=False)[be_cols].mean()
    global_wide = subset.set_index(KEY_COLS + ["Tissue class"])[be_cols].unstack("Tissue class")
    global_wide.columns = [f"{tissue} {metric}" for metric, tissue in global_wide.columns]
    return global_wide.reset_index()


def _merge_radiomics(base_df: pd.DataFrame, radiomics_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["Patient ID", "Structure ID", "Structure type", "Structure refnum"]
    base_rad_cols = [c for c in radiomics_df.columns if c not in id_cols]

    dil_rad = (
        radiomics_df[radiomics_df["Structure type"] == "DIL ref"]
        [["Patient ID", "Structure ID"] + base_rad_cols]
        .drop_duplicates(subset=["Patient ID", "Structure ID"])
        .rename(columns={c: f"DIL {c}" for c in base_rad_cols})
    )

    merged = base_df.merge(
        dil_rad,
        left_on=["Patient ID", "Relative DIL ID"],
        right_on=["Patient ID", "Structure ID"],
        how="left",
        validate="m:1",
    ).drop(columns=["Structure ID"], errors="ignore")

    prostate_rad = (
        radiomics_df[radiomics_df["Structure type"] == "OAR ref"]
        [["Patient ID", "Structure ID"] + base_rad_cols]
        .drop_duplicates(subset=["Patient ID", "Structure ID"])
        .rename(columns={c: f"Prostate {c}" for c in base_rad_cols})
    )

    merged = merged.merge(
        prostate_rad,
        left_on=["Patient ID", "Relative prostate ID"],
        right_on=["Patient ID", "Structure ID"],
        how="left",
        validate="m:1",
    ).drop(columns=["Structure ID"], errors="ignore")

    return merged


def _merge_distances(base_df: pd.DataFrame, distances_df: pd.DataFrame) -> pd.DataFrame:
    distances_flat = _flatten_columns(distances_df)

    dil_dist = (
        distances_flat[distances_flat["Relative structure type"] == "DIL ref"]
        [
            [
                "Patient ID",
                "Bx ID",
                "Bx index",
                "Relative structure index",
                "Struct. boundary NN dist. mean",
                "Dist. from struct. centroid mean",
            ]
        ]
        .drop_duplicates(subset=["Patient ID", "Bx ID", "Bx index", "Relative structure index"])
        .rename(
            columns={
                "Struct. boundary NN dist. mean": "DIL NN dist mean",
                "Dist. from struct. centroid mean": "DIL centroid dist mean",
            }
        )
    )

    merged = base_df.merge(
        dil_dist,
        left_on=["Patient ID", "Bx ID", "Bx index", "Relative DIL index"],
        right_on=["Patient ID", "Bx ID", "Bx index", "Relative structure index"],
        how="left",
        validate="m:1",
    ).drop(columns=["Relative structure index"], errors="ignore")

    for struct_type, prefix in [
        ("OAR ref", "Prostate"),
        ("Rectum ref", "Rectum"),
        ("Urethra ref", "Urethra"),
    ]:
        sub = (
            distances_flat[distances_flat["Relative structure type"] == struct_type]
            [
                [
                    "Patient ID",
                    "Bx ID",
                    "Bx index",
                    "Struct. boundary NN dist. mean",
                    "Dist. from struct. centroid mean",
                ]
            ]
            .drop_duplicates(subset=["Patient ID", "Bx ID", "Bx index"])
            .rename(
                columns={
                    "Struct. boundary NN dist. mean": f"{prefix} NN dist mean",
                    "Dist. from struct. centroid mean": f"{prefix} centroid dist mean",
                }
            )
        )
        merged = merged.merge(
            sub,
            on=["Patient ID", "Bx ID", "Bx index"],
            how="left",
            validate="m:1",
        )

    dim_cols = [
        "Prostate L/R dimension at centroid",
        "Prostate A/P dimension at centroid",
        "Prostate S/I dimension at centroid",
    ]
    if all(c in merged.columns for c in dim_cols):
        merged["Prostate mean dimension at centroid"] = merged[dim_cols].mean(axis=1)
        denom = merged["Prostate mean dimension at centroid"].replace(0, pd.NA)
        if "Prostate centroid dist mean" in merged.columns:
            merged["BX_to_prostate_centroid_distance_norm_mean_dim"] = (
                merged["Prostate centroid dist mean"] / denom
            )

    return merged


def build_qa_family_members_long(source_tables: QASourceTables) -> pd.DataFrame:
    biopsy_basic_df = (
        source_tables.cohort_biopsy_basic_spatial_features_df[[
            c
            for c in source_tables.cohort_biopsy_basic_spatial_features_df.columns
            if c != "Bx refnum"
        ]]
        .drop_duplicates(subset=KEY_COLS)
        .copy()
    )

    biopsy_basic_df["Base patient ID"] = _extract_base_patient_id(biopsy_basic_df["Patient ID"])
    biopsy_basic_df["Family ID"] = (
        biopsy_basic_df["Base patient ID"].astype(str)
        + "::"
        + biopsy_basic_df["Relative DIL index"].astype("Int64").astype(str)
    )
    biopsy_basic_df["Family member label"] = biopsy_basic_df["Simulated type"].replace(
        {"Centroid DIL": "Centroid", "Optimal DIL": "Optimal"}
    )

    global_wide = _build_global_tissue_wide(source_tables.cohort_global_sum_to_one_tissue_df)
    merged = biopsy_basic_df.merge(global_wide, on=KEY_COLS, how="left", validate="1:1")
    merged = _merge_radiomics(merged, source_tables.cohort_3d_radiomic_features_all_oar_dil_df)
    merged = _merge_distances(merged, source_tables.cohort_tissue_class_distances_global_df)

    family_counts = (
        merged.groupby(FAMILY_KEY_COLS, as_index=False)
        .agg(
            Family_member_count=("Bx index", "size"),
            Family_real_core_count=("Simulated type", lambda s: int((s == "Real").sum())),
            Family_centroid_count=("Simulated type", lambda s: int((s == "Centroid DIL").sum())),
            Family_optimal_count=("Simulated type", lambda s: int((s == "Optimal DIL").sum())),
        )
    )
    family_counts["Family_complete"] = (
        (family_counts["Family_real_core_count"] >= 1)
        & (family_counts["Family_centroid_count"] == 1)
        & (family_counts["Family_optimal_count"] == 1)
    )

    merged = merged.merge(family_counts, on=FAMILY_KEY_COLS, how="left", validate="m:1")

    sort_cols = [
        "Base patient ID",
        "Relative DIL index",
        "Simulated bool",
        "Simulated type",
        "Patient ID",
        "Bx index",
    ]
    return merged.sort_values(sort_cols).reset_index(drop=True)


def build_family_audit_table(qa_family_members_long: pd.DataFrame) -> pd.DataFrame:
    audit = (
        qa_family_members_long.groupby(FAMILY_KEY_COLS, as_index=False)
        .agg(
            Family_ID=("Family ID", "first"),
            Family_member_count=("Bx index", "size"),
            Family_real_core_count=("Simulated type", lambda s: int((s == "Real").sum())),
            Family_centroid_count=("Simulated type", lambda s: int((s == "Centroid DIL").sum())),
            Family_optimal_count=("Simulated type", lambda s: int((s == "Optimal DIL").sum())),
            Patient_ID_count=("Patient ID", "nunique"),
            Patient_IDs=("Patient ID", lambda s: " | ".join(sorted({str(v) for v in s}))),
            Relative_DIL_IDs=("Relative DIL ID", lambda s: " | ".join(sorted({str(v) for v in s}))),
            Real_Bx_IDs=(
                "Bx ID",
                lambda s: " | ".join(
                    sorted(
                        {
                            str(bx_id)
                            for bx_id, sim_type in zip(
                                qa_family_members_long.loc[s.index, "Bx ID"],
                                qa_family_members_long.loc[s.index, "Simulated type"],
                            )
                            if sim_type == "Real"
                        }
                    )
                ),
            ),
        )
    )
    audit["Family_complete"] = (
        (audit["Family_real_core_count"] >= 1)
        & (audit["Family_centroid_count"] == 1)
        & (audit["Family_optimal_count"] == 1)
    )
    return audit.sort_values(FAMILY_KEY_COLS).reset_index(drop=True)


def _available_pair_metric_columns(qa_family_members_long: pd.DataFrame) -> list[str]:
    return [col for col in PAIR_METRIC_COLUMNS if col in qa_family_members_long.columns]


def build_qa_real_core_pairs(qa_family_members_long: pd.DataFrame) -> pd.DataFrame:
    metric_cols = _available_pair_metric_columns(qa_family_members_long)
    ref_meta_cols = ["Patient ID", "Bx ID", "Bx index", "Simulated type", "Simulated bool"]

    real_df = qa_family_members_long[qa_family_members_long["Simulated type"] == "Real"].copy()
    centroid_df = qa_family_members_long[qa_family_members_long["Simulated type"] == "Centroid DIL"].copy()
    optimal_df = qa_family_members_long[qa_family_members_long["Simulated type"] == "Optimal DIL"].copy()

    centroid_df = centroid_df[FAMILY_KEY_COLS + ref_meta_cols + metric_cols].rename(
        columns={col: f"centroid_{col}" for col in ref_meta_cols + metric_cols}
    )
    optimal_df = optimal_df[FAMILY_KEY_COLS + ref_meta_cols + metric_cols].rename(
        columns={col: f"optimal_{col}" for col in ref_meta_cols + metric_cols}
    )

    out = real_df.merge(centroid_df, on=FAMILY_KEY_COLS, how="left", validate="m:1")
    out = out.merge(optimal_df, on=FAMILY_KEY_COLS, how="left", validate="m:1")

    for metric_col in metric_cols:
        out[f"delta_centroid_minus_real__{metric_col}"] = out[f"centroid_{metric_col}"] - out[metric_col]
        out[f"delta_optimal_minus_real__{metric_col}"] = out[f"optimal_{metric_col}"] - out[metric_col]

    return out.sort_values(FAMILY_KEY_COLS + ["Patient ID", "Bx index"]).reset_index(drop=True)


def build_qa_family_reference_pairs(qa_family_members_long: pd.DataFrame) -> pd.DataFrame:
    metric_cols = _available_pair_metric_columns(qa_family_members_long)
    ref_meta_cols = ["Patient ID", "Bx ID", "Bx index", "Simulated type", "Simulated bool"]

    centroid_df = qa_family_members_long[qa_family_members_long["Simulated type"] == "Centroid DIL"].copy()
    optimal_df = qa_family_members_long[qa_family_members_long["Simulated type"] == "Optimal DIL"].copy()

    out = centroid_df[FAMILY_KEY_COLS + ref_meta_cols + metric_cols].rename(
        columns={col: f"centroid_{col}" for col in ref_meta_cols + metric_cols}
    )
    optimal_renamed = optimal_df[FAMILY_KEY_COLS + ref_meta_cols + metric_cols].rename(
        columns={col: f"optimal_{col}" for col in ref_meta_cols + metric_cols}
    )
    out = out.merge(optimal_renamed, on=FAMILY_KEY_COLS, how="left", validate="1:1")

    for metric_col in metric_cols:
        out[f"delta_optimal_minus_centroid__{metric_col}"] = (
            out[f"optimal_{metric_col}"] - out[f"centroid_{metric_col}"]
        )

    return out.sort_values(FAMILY_KEY_COLS).reset_index(drop=True)


def build_qa_family_real_aggregated(qa_family_members_long: pd.DataFrame) -> pd.DataFrame:
    metric_cols = _available_pair_metric_columns(qa_family_members_long)
    real_df = qa_family_members_long[qa_family_members_long["Simulated type"] == "Real"].copy()

    agg_spec: dict[str, tuple[str, str]] = {
        "Real_patient_IDs": ("Patient ID", lambda s: " | ".join(sorted({str(v) for v in s}))),
        "Real_Bx_IDs": ("Bx ID", lambda s: " | ".join(sorted({str(v) for v in s}))),
        "Real_core_count": ("Bx index", "size"),
    }
    for metric_col in metric_cols:
        agg_spec[f"real_mean__{metric_col}"] = (metric_col, "mean")
        agg_spec[f"real_best__{metric_col}"] = (metric_col, "max")
        agg_spec[f"real_worst__{metric_col}"] = (metric_col, "min")

    out = real_df.groupby(FAMILY_KEY_COLS, as_index=False).agg(**agg_spec)
    return out.sort_values(FAMILY_KEY_COLS).reset_index(drop=True)


def build_qa_family_outputs(source_tables: QASourceTables) -> QAFamilyOutputs:
    qa_family_members_long = build_qa_family_members_long(source_tables)
    qa_family_audit = build_family_audit_table(qa_family_members_long)
    qa_real_core_pairs = build_qa_real_core_pairs(qa_family_members_long)
    qa_family_reference_pairs = build_qa_family_reference_pairs(qa_family_members_long)
    qa_family_real_aggregated = build_qa_family_real_aggregated(qa_family_members_long)
    return QAFamilyOutputs(
        qa_family_members_long=qa_family_members_long,
        qa_family_audit=qa_family_audit,
        qa_real_core_pairs=qa_real_core_pairs,
        qa_family_reference_pairs=qa_family_reference_pairs,
        qa_family_real_aggregated=qa_family_real_aggregated,
    )
