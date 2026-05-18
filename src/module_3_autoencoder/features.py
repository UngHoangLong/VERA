import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import FEATURE_SPECS, FEATURE_NAMES


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_nested_value(data: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def transform_value(value: Optional[float], transform: str) -> float:
    if value is None:
        return np.nan
    if transform == "identity":
        return value
    if transform == "one_minus":
        return 1.0 - value
    raise ValueError(f"Unsupported transform: {transform}")


def extract_chunk_features(chunk: Dict[str, Any]) -> Tuple[Dict[str, float], np.ndarray]:
    feature_dict: Dict[str, float] = {}
    for spec in FEATURE_SPECS:
        raw_value = get_nested_value(chunk, spec["path"])
        transformed = transform_value(raw_value, spec["transform"])
        feature_dict[spec["name"]] = float(transformed) if not np.isnan(transformed) else np.nan
    vector = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
    return feature_dict, vector


def extract_text_evidence(chunk: Dict[str, Any]) -> Dict[str, Any]:
    av = chunk.get("audio_visual_consistency", {})
    transcripts = av.get("transcripts", {})
    semantic = av.get("semantic_consistency", {})
    temporal = av.get("temporal_sync", {})
    return {
        "asr_text_audio": transcripts.get("asr_text_audio"),
        "vsr_text_lips": transcripts.get("vsr_text_lips"),
        "wer_score": transcripts.get("wer_score"),
        "percentile_3rd_cosine": semantic.get("percentile_3rd_cosine"),
        "sync_score": temporal.get("sync_score"),
        "min_sync_score": temporal.get("min_sync_score"),
    }


def iter_report_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*_report.json"))


def parse_report(path: Path) -> List[Dict[str, Any]]:
    report = load_json(path)
    video_id = report.get("video_metadata", {}).get("video_id", path.stem.replace("_report", ""))
    chunks = report.get("chunks", {})
    rows: List[Dict[str, Any]] = []
    for chunk_id, chunk in chunks.items():
        feature_dict, vector = extract_chunk_features(chunk)
        rows.append({
            "source_report": str(path),
            "video_id": video_id,
            "chunk_id": chunk_id,
            "time_metadata": chunk.get("time_metadata", {}),
            "frames_analyzed": chunk.get("frames_analyzed"),
            "feature_dict": feature_dict,
            "vector": vector,
            "raw_text_evidence": extract_text_evidence(chunk),
        })
    return rows


def parse_reports(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for path in paths:
        all_rows.extend(parse_report(path))
    return all_rows


def rows_to_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([row["vector"] for row in rows]).astype(np.float32)


def split_report_files_by_video(
    report_files: List[Path],
    val_ratio: float,
    seed: int
) -> Tuple[List[Path], List[Path]]:
    """
    Split reports by video file, but balance by number of chunks.

    Why:
    - We must split by video to avoid leakage between overlapping chunks.
    - But report files can contain very different numbers of chunks.
    - Therefore, validation selection should target val_ratio of total chunks,
      not val_ratio of number of files.
    """
    files = list(report_files)

    if not files:
        return [], []

    if val_ratio <= 0 or len(files) < 2:
        return files, []

    rng = np.random.default_rng(seed)

    file_stats = []
    for path in files:
        try:
            num_chunks = len(parse_report(path))
        except Exception:
            num_chunks = 0
        file_stats.append({
            "path": path,
            "num_chunks": num_chunks,
            "tie_breaker": float(rng.random())
        })

    total_chunks = sum(item["num_chunks"] for item in file_stats)

    if total_chunks <= 0:
        return files, []

    target_val_chunks = max(1, int(round(total_chunks * val_ratio)))

    # Prefer smaller videos for validation first.
    # This prevents one long video from dominating validation.
    file_stats = sorted(
        file_stats,
        key=lambda item: (item["num_chunks"], item["tie_breaker"])
    )

    val_files = []
    val_chunks = 0

    for item in file_stats:
        if len(val_files) >= len(file_stats) - 1:
            break

        if val_chunks >= target_val_chunks:
            break

        val_files.append(item["path"])
        val_chunks += item["num_chunks"]

    val_set = set(val_files)
    train_files = [item["path"] for item in file_stats if item["path"] not in val_set]

    if not train_files:
        return files, []

    return train_files, val_files