from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAIN_OUTPUT_PATH = Path(
    "/home/matthew-muscat/Documents/UBC/Research/Data/Output data/"
    "MC_sim_out- Date-Mar-03-2026 Time-15,34,07"
)


def resolve_existing_output_dir(requested_path: Path) -> Path:
    if requested_path.is_dir():
        return requested_path

    parent = requested_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    candidates = sorted(
        path
        for path in parent.glob(f"{requested_path.name}*")
        if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve output directory '{requested_path}'. No matching folder prefix was found."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            "Ambiguous output directory prefix "
            f"'{requested_path}'. Matching candidates: {[str(path) for path in candidates]}"
        )
    return candidates[0]


@dataclass(frozen=True)
class QAStudyConfig:
    main_output_path: Path = DEFAULT_MAIN_OUTPUT_PATH
    output_root: Path = REPO_ROOT / "output_data_QA"
    csv_subdir: str = "csv"
    manifest_subdir: str = "manifests"
    bootstrap_iterations: int = 10000
    bootstrap_seed: int = 20260415
    bootstrap_confidence_level: float = 0.95
