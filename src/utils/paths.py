"""
Shared path configuration for the Module 1 / Module 2 extraction pipeline.

`mode` keeps genuine (training) data and infer (evaluation) data fully
separate across every stage of the pipeline:

  mode="genuine" -> data/raw/genuine,  data/interim/genuine,  data/processed/genuine,
                     final_reports_genuine  (Module 3 training input)
  mode="infer"   -> data/raw/infer,    data/interim/infer,    data/processed/infer,
                     final_reports_infer    (Module 3 inference input)
"""

from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_MODES = ("genuine", "infer")


def get_pipeline_paths(mode: str, project_root: Path = PROJECT_ROOT) -> Dict[str, Path]:
    """Return all mode-dependent pipeline paths for the given mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    project_root = Path(project_root)
    data_root = project_root / "data"

    return {
        "project_root": project_root,
        "raw_dir": data_root / "raw" / mode,
        "interim_dir": data_root / "interim" / mode,
        "processed_dir": data_root / "processed" / mode,
        "final_reports_dir": project_root / f"final_reports_{mode}",
    }
