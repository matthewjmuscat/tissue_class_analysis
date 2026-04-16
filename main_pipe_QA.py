from __future__ import annotations

from pathlib import Path

import pandas as pd

from qa.config import QAStudyConfig
from qa.families import build_qa_family_outputs
from qa.load import load_qa_source_tables
from qa.stats import build_qa_stats_outputs


def _ensure_dirs(config: QAStudyConfig) -> dict[str, Path]:
    dirs = {
        "root": config.output_root,
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


def main() -> None:
    config = QAStudyConfig()
    dirs = _ensure_dirs(config)

    source_tables = load_qa_source_tables(config)
    family_outputs = build_qa_family_outputs(source_tables)
    stats_outputs = build_qa_stats_outputs(family_outputs, config)

    qa_csv_dir = dirs["csv"] / "qa"
    manifests_dir = dirs["manifests"]

    _write_csv(qa_csv_dir / "qa_family_members_long.csv", family_outputs.qa_family_members_long)
    _write_csv(qa_csv_dir / "qa_family_audit.csv", family_outputs.qa_family_audit)
    _write_csv(qa_csv_dir / "qa_real_core_pairs.csv", family_outputs.qa_real_core_pairs)
    _write_csv(qa_csv_dir / "qa_family_reference_pairs.csv", family_outputs.qa_family_reference_pairs)
    _write_csv(qa_csv_dir / "qa_family_real_aggregated.csv", family_outputs.qa_family_real_aggregated)
    _write_csv(qa_csv_dir / "qa_headline_delta_long.csv", stats_outputs.headline_delta_long_df)
    _write_csv(qa_csv_dir / "qa_headline_bootstrap_summary.csv", stats_outputs.headline_bootstrap_summary_df)
    _write_csv(qa_csv_dir / "qa_headline_bootstrap_samples.csv", stats_outputs.headline_bootstrap_samples_df)

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
        }
    )
    _write_csv(manifests_dir / "qa_table_inventory.csv", inventory)

    complete_families = int(family_outputs.qa_family_audit["Family_complete"].sum())
    total_families = int(len(family_outputs.qa_family_audit))
    print(f"[QA] wrote outputs to: {config.output_root}")
    print(f"[QA] complete families: {complete_families}/{total_families}")
    print(f"[QA] family members rows: {len(family_outputs.qa_family_members_long)}")
    print(f"[QA] real-core pairs rows: {len(family_outputs.qa_real_core_pairs)}")
    print(f"[QA] headline bootstrap rows: {len(stats_outputs.headline_bootstrap_summary_df)}")


if __name__ == "__main__":
    main()
