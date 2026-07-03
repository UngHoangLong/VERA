"""
ranker.py — Module 4 step 1: rank chunks by anomaly score.

Reads a Module 3 *_evidence.json and returns the top-K most anomalous chunks,
ranked by joint_anomaly_score. We rank by joint_anomaly_score (uncapped), NOT
normalized_anomaly_score (capped at 1.0) — otherwise every chunk above the
threshold would tie at 1.0 and become unrankable.

Also computes a video-level summary (score distribution + temporal pattern)
consumed by Module 5's Block A.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_evidence(evidence_path) -> Dict[str, Any]:
    with open(evidence_path, encoding="utf-8") as f:
        return json.load(f)


def _chunk_score(chunk: Dict[str, Any]) -> float:
    return float(chunk.get("anomaly", {}).get("joint_anomaly_score", 0.0))


def rank_chunks(evidence: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(chunk_id, chunk_data), ...] sorted by joint_anomaly_score desc."""
    chunks = evidence.get("chunks", {})
    ranked = sorted(chunks.items(), key=lambda kv: _chunk_score(kv[1]), reverse=True)
    return ranked[:top_k]


def _classify_temporal(positions: List[float], duration: float) -> str:
    """
    Heuristic temporal pattern from the start times of above-threshold chunks.
      ISOLATED     - a single anomalous chunk
      CONCENTRATED - anomalies clustered in a small time window (<=30% of video)
      SCATTERED    - anomalies span most of the video (>=60%)
      MIXED        - in between
    """
    if not positions or duration <= 0:
        return "NONE"
    if len(positions) == 1:
        return "ISOLATED"
    spread = (max(positions) - min(positions)) / duration
    if spread >= 0.6:
        return "SCATTERED"
    if spread <= 0.3:
        return "CONCENTRATED"
    return "MIXED"


def compute_video_summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Video-level score distribution + temporal pattern for Module 5 Block A."""
    chunks = evidence.get("chunks", {})
    scores = [_chunk_score(c) for c in chunks.values()]

    threshold = None
    duration = 0.0
    above_positions: List[float] = []

    for c in chunks.values():
        anomaly = c.get("anomaly", {})
        if threshold is None and anomaly.get("threshold") is not None:
            threshold = float(anomaly["threshold"])
        tm = c.get("time_metadata", {})
        end = tm.get("end_sec") or 0.0
        duration = max(duration, float(end))
        if threshold is not None and _chunk_score(c) >= threshold:
            above_positions.append(float(tm.get("start_sec") or 0.0))

    n = len(scores)
    return {
        "total_chunks": n,
        "threshold": threshold,
        "mean_score": round(sum(scores) / n, 6) if n else 0.0,
        "max_score": round(max(scores), 6) if scores else 0.0,
        "chunks_above_threshold": len(above_positions),
        "video_duration_sec": round(duration, 2),
        "temporal_pattern": _classify_temporal(above_positions, duration),
    }
