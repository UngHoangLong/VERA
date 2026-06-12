"""
dataset.py — Data loading, preprocessing, and PyTorch Dataset for Module 3.

Responsibilities:
  - Parse Module 2 JSON reports into feature rows
  - Split the flat 15-dim feature vector into visual / audio modalities
  - Scale features with NaN-aware RobustScaler (no imputation)
  - Expose a ModalityDataset for DataLoader usage
  - Train/val split by video (no chunk leakage)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    AUDIO_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    VISUAL_FEATURE_NAMES,
)

VISUAL_INDICES = [FEATURE_NAMES.index(n) for n in VISUAL_FEATURE_NAMES]
AUDIO_INDICES = [FEATURE_NAMES.index(n) for n in AUDIO_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Feature extraction from Module 2 JSON
# ---------------------------------------------------------------------------

def _get_nested(data: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    try:
        v = float(cur)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def _apply_transform(value: Optional[float], transform: str) -> float:
    if value is None:
        return np.nan
    if transform == "identity":
        return value
    if transform == "one_minus":
        return 1.0 - value
    raise ValueError(f"Unsupported transform: {transform}")


def extract_chunk_features(chunk: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    feature_dict: Dict[str, Any] = {}
    for spec in FEATURE_SPECS:
        raw = _get_nested(chunk, spec["path"])
        t = _apply_transform(raw, spec["transform"])
        feature_dict[spec["name"]] = float(t) if not np.isnan(t) else None
    vector = np.array(
        [feature_dict[n] if feature_dict[n] is not None else np.nan for n in FEATURE_NAMES],
        dtype=np.float32,
    )
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
    rows: List[Dict[str, Any]] = []
    for chunk_id, chunk in report.get("chunks", {}).items():
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
    rows: List[Dict[str, Any]] = []
    for p in paths:
        rows.extend(parse_report(p))
    return rows


# ---------------------------------------------------------------------------
# Modality split
# ---------------------------------------------------------------------------

def split_to_modalities(
    vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a flat [15] feature vector into visual and audio sub-vectors.

    Returns:
        x_visual  : [visual_dim] float, NaN where unobserved
        x_audio   : [audio_dim]  float, NaN where unobserved
        mask_visual: [visual_dim] bool — True = feature observed
        mask_audio : [audio_dim]  bool
    """
    x_v = vector[VISUAL_INDICES]
    x_a = vector[AUDIO_INDICES]
    return x_v, x_a, ~np.isnan(x_v), ~np.isnan(x_a)


def rows_to_modality_matrices(
    rows: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert parsed rows into zero-filled modality matrices + observation masks.

    NaN values are NOT imputed — they are tracked via masks and zero-filled
    only for encoder input. The masks drive masked reconstruction loss.

    Returns:
        x_visual     : [N, visual_dim] float32, zero-filled
        x_audio      : [N, audio_dim]  float32, zero-filled
        mask_visual  : [N, visual_dim] bool
        mask_audio   : [N, audio_dim]  bool
        avail_visual : [N] bool — True if ≥1 visual feature observed
        avail_audio  : [N] bool — True if ≥1 audio feature observed
    """
    n = len(rows)
    vd, ad = len(VISUAL_FEATURE_NAMES), len(AUDIO_FEATURE_NAMES)
    x_v = np.full((n, vd), np.nan, dtype=np.float32)
    x_a = np.full((n, ad), np.nan, dtype=np.float32)
    mask_v = np.zeros((n, vd), dtype=bool)
    mask_a = np.zeros((n, ad), dtype=bool)

    for i, row in enumerate(rows):
        xv, xa, mv, ma = split_to_modalities(row["vector"])
        x_v[i], x_a[i], mask_v[i], mask_a[i] = xv, xa, mv, ma

    avail_v = mask_v.any(axis=1)
    avail_a = mask_a.any(axis=1)

    x_v = np.where(mask_v, x_v, 0.0).astype(np.float32)
    x_a = np.where(mask_a, x_a, 0.0).astype(np.float32)
    return x_v, x_a, mask_v, mask_a, avail_v, avail_a


# ---------------------------------------------------------------------------
# NaN-aware robust scaler
# ---------------------------------------------------------------------------

class ModalityScaler:
    """
    Per-feature RobustScaler using nanmedian + nanIQR.

    Differences from sklearn RobustScaler:
      - Fits on NaN-containing arrays (ignores NaN via nanmedian/nanpercentile)
      - Transform preserves NaN (no imputation)
      - Zero-fill is done downstream, not here
    """

    def __init__(self) -> None:
        self.center_: np.ndarray = np.array([])
        self.scale_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "ModalityScaler":
        self.center_ = np.nanmedian(X, axis=0)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr < 1e-8] = 1.0
        self.scale_ = iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.center_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ModalityDataset(Dataset):
    """
    Dataset that yields (x_visual, x_audio, mask_visual, mask_audio,
    avail_visual, avail_audio) per sample.

    x_* are zero-filled scaled inputs.
    mask_* are float tensors (1.0 = observed) used in masked reconstruction loss.
    avail_* are bool tensors used in PoE fusion.
    """

    def __init__(
        self,
        x_v: np.ndarray,
        x_a: np.ndarray,
        mask_v: np.ndarray,
        mask_a: np.ndarray,
        avail_v: np.ndarray,
        avail_a: np.ndarray,
    ) -> None:
        self.x_v = torch.tensor(x_v, dtype=torch.float32)
        self.x_a = torch.tensor(x_a, dtype=torch.float32)
        self.mask_v = torch.tensor(mask_v, dtype=torch.float32)
        self.mask_a = torch.tensor(mask_a, dtype=torch.float32)
        self.avail_v = torch.tensor(avail_v, dtype=torch.bool)
        self.avail_a = torch.tensor(avail_a, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.x_v)

    def __getitem__(self, idx: int) -> Tuple:
        return (
            self.x_v[idx], self.x_a[idx],
            self.mask_v[idx], self.mask_a[idx],
            self.avail_v[idx], self.avail_a[idx],
        )


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_report_files_by_video(
    report_files: List[Path],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """
    Split report files into train/val by video (not by chunk) to prevent
    data leakage between overlapping chunks of the same video.
    Balances by number of chunks rather than number of files.
    """
    files = list(report_files)
    if not files:
        return [], []
    if val_ratio <= 0 or len(files) < 2:
        return files, []

    rng = np.random.default_rng(seed)
    stats = []
    for p in files:
        try:
            n = len(parse_report(p))
        except Exception:
            n = 0
        stats.append({"path": p, "n": n, "tie": float(rng.random())})

    total = sum(s["n"] for s in stats)
    if total <= 0:
        return files, []

    target_val = max(1, int(round(total * val_ratio)))
    stats = sorted(stats, key=lambda s: (s["n"], s["tie"]))

    val_files: List[Path] = []
    val_n = 0
    for s in stats:
        if len(val_files) >= len(stats) - 1 or val_n >= target_val:
            break
        val_files.append(s["path"])
        val_n += s["n"]

    val_set = set(val_files)
    train_files = [s["path"] for s in stats if s["path"] not in val_set]
    return (train_files, val_files) if train_files else (files, [])
