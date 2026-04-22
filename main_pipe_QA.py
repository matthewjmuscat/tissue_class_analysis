from __future__ import annotations

from pathlib import Path

import pandas as pd

import production_plots_QA
from qa.config import QAFigureExportConfig, QAStudyConfig
from qa.families import build_qa_family_outputs
from qa.load import load_qa_source_tables
from qa.plot_data import build_qa_plot_data_outputs
from qa.stats import build_qa_stats_outputs


def _ensure_dirs(config: QAStudyConfig) -> dict[str, Path]:
    dirs = {
        "root": config.output_root,
        "figures": config.output_root / config.figures_subdir,
        "csv": config.output_root / config.csv_subdir,
        "manifests": config.output_root / config.manifest_subdir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_csv(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def _build_inventory(extra_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in extra_frames.items():
        rows.append(
            {
                "table_name": name,
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
            }
        )
    return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)


def _build_figure_manifest(figure_paths: dict[str, list[Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for figure_key, paths in figure_paths.items():
        for path in paths:
            rows.append(
                {
                    "figure_key": figure_key,
                    "path": str(path),
                    "exists": path.exists(),
                }
            )
    return pd.DataFrame(rows).sort_values(["figure_key", "path"]).reset_index(drop=True)


def main() -> None:
    config = QAStudyConfig()
    export_config = QAFigureExportConfig()
    dirs = _ensure_dirs(config)

    source_tables = load_qa_source_tables(config)
    family_outputs = build_qa_family_outputs(source_tables)
    stats_outputs = build_qa_stats_outputs(family_outputs, config)
    plot_data_outputs = build_qa_plot_data_outputs(source_tables, family_outputs, stats_outputs)

    qa_csv_dir = dirs["csv"] / "qa"
    qa_fig_dir = dirs["figures"] / "qa"
    manifests_dir = dirs["manifests"]

    _write_csv(qa_csv_dir / "qa_family_members_long.csv", family_outputs.qa_family_members_long)
    _write_csv(qa_csv_dir / "qa_family_audit.csv", family_outputs.qa_family_audit)
    _write_csv(qa_csv_dir / "qa_real_core_pairs.csv", family_outputs.qa_real_core_pairs)
    _write_csv(qa_csv_dir / "qa_family_reference_pairs.csv", family_outputs.qa_family_reference_pairs)
    _write_csv(qa_csv_dir / "qa_family_real_aggregated.csv", family_outputs.qa_family_real_aggregated)
    _write_csv(qa_csv_dir / "qa_headline_delta_long.csv", stats_outputs.headline_delta_long_df)
    _write_csv(qa_csv_dir / "qa_headline_bootstrap_summary.csv", stats_outputs.headline_bootstrap_summary_df)
    _write_csv(qa_csv_dir / "qa_headline_bootstrap_samples.csv", stats_outputs.headline_bootstrap_samples_df)
    _write_csv(qa_csv_dir / "qa_safety_delta_long.csv", stats_outputs.safety_delta_long_df)
    _write_csv(qa_csv_dir / "qa_safety_bootstrap_summary.csv", stats_outputs.safety_bootstrap_summary_df)
    _write_csv(qa_csv_dir / "qa_safety_bootstrap_samples.csv", stats_outputs.safety_bootstrap_samples_df)
    _write_csv(
        qa_csv_dir / "qa_plot_family_comparison_long.csv",
        plot_data_outputs.qa_plot_family_comparison_long,
    )
    _write_csv(
        qa_csv_dir / "qa_plot_headroom_long.csv",
        plot_data_outputs.qa_plot_headroom_long,
    )
    _write_csv(
        qa_csv_dir / "qa_plot_reference_disagreement.csv",
        plot_data_outputs.qa_plot_reference_disagreement,
    )
    _write_csv(
        qa_csv_dir / "qa_plot_safety_distance_long.csv",
        plot_data_outputs.qa_plot_safety_distance_long,
    )
    _write_csv(
        qa_csv_dir / "qa_plot_selected_profile_long.csv",
        plot_data_outputs.qa_plot_selected_profile_long,
    )
    _write_csv(
        qa_csv_dir / "qa_selected_profile_cases.csv",
        plot_data_outputs.qa_selected_profile_cases,
    )
    _write_csv(
        qa_csv_dir / "qa_plot_localization_real.csv",
        plot_data_outputs.qa_plot_localization_real,
    )
    _write_csv(
        qa_csv_dir / "qa_family_optimizer_difficulty.csv",
        plot_data_outputs.qa_family_optimizer_difficulty,
    )
    _write_csv(
        qa_csv_dir / "qa_targeting_difficulty_correlations.csv",
        plot_data_outputs.qa_targeting_difficulty_correlations,
    )
    _write_csv(
        qa_csv_dir / "qa_targeting_difficulty_group_summary.csv",
        plot_data_outputs.qa_targeting_difficulty_group_summary,
    )

    figure_paths = {
        "headline_family_comparison": production_plots_QA.plot_headline_family_comparison(
            plot_data_outputs.qa_plot_family_comparison_long,
            stats_outputs.headline_bootstrap_summary_df,
            qa_fig_dir,
            export_config=export_config,
        ),
        "headline_headroom": production_plots_QA.plot_headline_headroom(
            plot_data_outputs.qa_plot_headroom_long,
            stats_outputs.headline_bootstrap_summary_df,
            qa_fig_dir,
            export_config=export_config,
        ),
        "reference_disagreement": production_plots_QA.plot_reference_disagreement(
            plot_data_outputs.qa_plot_reference_disagreement,
            qa_fig_dir,
            export_config=export_config,
        ),
        "safety_distance_family_comparison": production_plots_QA.plot_safety_distance_family_comparison(
            plot_data_outputs.qa_plot_safety_distance_long,
            stats_outputs.safety_bootstrap_summary_df,
            qa_fig_dir,
            export_config=export_config,
        ),
        "selected_dil_profiles": production_plots_QA.plot_selected_dil_profiles(
            plot_data_outputs.qa_plot_selected_profile_long,
            plot_data_outputs.qa_selected_profile_cases,
            qa_fig_dir,
            export_config=export_config,
        ),
        "selected_dil_profiles_step": production_plots_QA.plot_selected_dil_profiles_step(
            plot_data_outputs.qa_plot_selected_profile_long,
            plot_data_outputs.qa_selected_profile_cases,
            qa_fig_dir,
            export_config=export_config,
        ),
        "localization_accuracy_centroids": production_plots_QA.plot_localization_accuracy_centroids(
            plot_data_outputs.qa_plot_localization_real,
            qa_fig_dir,
            export_config=export_config,
        ),
        "optimizer_difficulty_summary": production_plots_QA.plot_optimizer_difficulty_summary(
            plot_data_outputs.qa_family_optimizer_difficulty,
            qa_fig_dir,
            export_config=export_config,
        ),
        "targeting_difficulty_summary": production_plots_QA.plot_targeting_difficulty_summary(
            plot_data_outputs.qa_family_optimizer_difficulty,
            qa_fig_dir,
            export_config=export_config,
        ),
        "optimizer_difficulty_continuous": production_plots_QA.plot_optimizer_difficulty_continuous(
            plot_data_outputs.qa_family_optimizer_difficulty,
            qa_fig_dir,
            export_config=export_config,
        ),
        "targeting_difficulty_continuous": production_plots_QA.plot_targeting_difficulty_continuous(
            plot_data_outputs.qa_family_optimizer_difficulty,
            qa_fig_dir,
            export_config=export_config,
        ),
        "optimizer_difficulty_categorical": production_plots_QA.plot_optimizer_difficulty_categorical(
            plot_data_outputs.qa_family_optimizer_difficulty,
            qa_fig_dir,
            export_config=export_config,
        ),
        "targeting_difficulty_categorical": production_plots_QA.plot_targeting_difficulty_categorical(
            plot_data_outputs.qa_family_optimizer_difficulty,
            qa_fig_dir,
            export_config=export_config,
        ),
    }

    inventory = _build_inventory(
        {
            "qa_family_members_long": family_outputs.qa_family_members_long,
            "qa_family_audit": family_outputs.qa_family_audit,
            "qa_real_core_pairs": family_outputs.qa_real_core_pairs,
            "qa_family_reference_pairs": family_outputs.qa_family_reference_pairs,
            "qa_family_real_aggregated": family_outputs.qa_family_real_aggregated,
            "qa_headline_delta_long": stats_outputs.headline_delta_long_df,
            "qa_headline_bootstrap_summary": stats_outputs.headline_bootstrap_summary_df,
            "qa_headline_bootstrap_samples": stats_outputs.headline_bootstrap_samples_df,
            "qa_safety_delta_long": stats_outputs.safety_delta_long_df,
            "qa_safety_bootstrap_summary": stats_outputs.safety_bootstrap_summary_df,
            "qa_safety_bootstrap_samples": stats_outputs.safety_bootstrap_samples_df,
            "qa_plot_family_comparison_long": plot_data_outputs.qa_plot_family_comparison_long,
            "qa_plot_headroom_long": plot_data_outputs.qa_plot_headroom_long,
            "qa_plot_reference_disagreement": plot_data_outputs.qa_plot_reference_disagreement,
            "qa_plot_safety_distance_long": plot_data_outputs.qa_plot_safety_distance_long,
            "qa_plot_selected_profile_long": plot_data_outputs.qa_plot_selected_profile_long,
            "qa_selected_profile_cases": plot_data_outputs.qa_selected_profile_cases,
            "qa_plot_localization_real": plot_data_outputs.qa_plot_localization_real,
            "qa_family_optimizer_difficulty": plot_data_outputs.qa_family_optimizer_difficulty,
            "qa_targeting_difficulty_correlations": plot_data_outputs.qa_targeting_difficulty_correlations,
            "qa_targeting_difficulty_group_summary": plot_data_outputs.qa_targeting_difficulty_group_summary,
        }
    )
    _write_csv(manifests_dir / "qa_table_inventory.csv", inventory)
    _write_csv(manifests_dir / "qa_figure_manifest.csv", _build_figure_manifest(figure_paths))

    complete_families = int(family_outputs.qa_family_audit["Family_complete"].sum())
    total_families = int(len(family_outputs.qa_family_audit))
    print(f"[QA] wrote outputs to: {config.output_root}")
    print(f"[QA] complete families: {complete_families}/{total_families}")
    print(f"[QA] family members rows: {len(family_outputs.qa_family_members_long)}")
    print(f"[QA] real-core pairs rows: {len(family_outputs.qa_real_core_pairs)}")
    print(f"[QA] headline bootstrap rows: {len(stats_outputs.headline_bootstrap_summary_df)}")
    print(f"[QA] figures written: {sum(len(paths) for paths in figure_paths.values())}")


if __name__ == "__main__":
    main()
