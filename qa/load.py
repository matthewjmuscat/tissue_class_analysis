from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import load_files
from qa.config import QAStudyConfig, resolve_existing_output_dir


@dataclass(frozen=True)
class QADataPaths:
    main_output_path: Path
    csv_directory: Path
    cohort_csvs_directory: Path


@dataclass
class QASourceTables:
    config: QAStudyConfig
    paths: QADataPaths
    cohort_biopsy_basic_spatial_features_df: pd.DataFrame
    cohort_nearest_dils_df: pd.DataFrame
    cohort_global_sum_to_one_tissue_df: pd.DataFrame
    cohort_sum_to_one_mc_results_df: pd.DataFrame
    cohort_3d_radiomic_features_all_oar_dil_df: pd.DataFrame
    cohort_tissue_class_distances_global_df: pd.DataFrame
    cohort_tissue_class_distances_ptwise_df: pd.DataFrame
    cohort_tissue_class_distances_voxelwise_df: pd.DataFrame
    cohort_per_voxel_prostate_double_sextant_df: pd.DataFrame


def resolve_qa_paths(config: QAStudyConfig) -> QADataPaths:
    resolved_main_output_path = resolve_existing_output_dir(config.main_output_path)
    csv_directory = resolved_main_output_path / "Output CSVs"
    return QADataPaths(
        main_output_path=resolved_main_output_path,
        csv_directory=csv_directory,
        cohort_csvs_directory=csv_directory / "Cohort",
    )


def load_qa_source_tables(config: QAStudyConfig | None = None) -> QASourceTables:
    config = QAStudyConfig() if config is None else config
    paths = resolve_qa_paths(config)

    cohort_biopsy_basic_spatial_features_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: Biopsy basic spatial features dataframe.csv"
    )
    cohort_nearest_dils_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: Nearest DILs to each biopsy.csv"
    )
    cohort_global_sum_to_one_tissue_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: global sum-to-one mc results.csv"
    )
    cohort_sum_to_one_mc_results_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: sum-to-one mc results.csv"
    )
    cohort_3d_radiomic_features_all_oar_dil_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: 3D radiomic features all OAR and DIL structures.csv"
    )
    cohort_tissue_class_distances_global_df = load_files.load_multiindex_csv(
        paths.cohort_csvs_directory / "Cohort: Tissue class - distances global results.csv",
        header_rows=[0, 1],
    )
    cohort_tissue_class_distances_ptwise_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: Tissue class - distances pt-wise results.csv"
    )
    cohort_tissue_class_distances_voxelwise_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: Tissue class - distances voxel-wise results.csv"
    )
    cohort_per_voxel_prostate_double_sextant_df = load_files.load_csv_as_dataframe(
        paths.cohort_csvs_directory / "Cohort: Per voxel prostate double sextant classification.csv"
    )

    return QASourceTables(
        config=config,
        paths=paths,
        cohort_biopsy_basic_spatial_features_df=cohort_biopsy_basic_spatial_features_df,
        cohort_nearest_dils_df=cohort_nearest_dils_df,
        cohort_global_sum_to_one_tissue_df=cohort_global_sum_to_one_tissue_df,
        cohort_sum_to_one_mc_results_df=cohort_sum_to_one_mc_results_df,
        cohort_3d_radiomic_features_all_oar_dil_df=cohort_3d_radiomic_features_all_oar_dil_df,
        cohort_tissue_class_distances_global_df=cohort_tissue_class_distances_global_df,
        cohort_tissue_class_distances_ptwise_df=cohort_tissue_class_distances_ptwise_df,
        cohort_tissue_class_distances_voxelwise_df=cohort_tissue_class_distances_voxelwise_df,
        cohort_per_voxel_prostate_double_sextant_df=cohort_per_voxel_prostate_double_sextant_df,
    )
