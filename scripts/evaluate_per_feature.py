#!/usr/bin/env python3
"""
Per-feature discriminative power analysis.

For each video, takes the chunk with highest joint_anomaly_score,
extracts all 21 feature values, computes AUC per feature vs real/fake labels.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


VISUAL_FEATURES = [
    "max_blur_flicker", "blur_flicker_variance", "max_texture_flicker",
    "asymmetry_max", "mean_landmark_jitter", "max_kinematic_flicker",
    "max_rigid_violation", "blinking_variance", "mouth_movement_variance",
    "gaze_anomaly", "iris_jitter_variance", "max_blending_flicker",
    "blending_variance",
]

AUDIO_VISUAL_FEATURES = [
    "wer_score", "semantic_anomaly", "min_cosine_anomaly",
    "temporal_anomaly", "min_temporal_anomaly", "temporal_sync_variance",
    "vocal_jitter_relative", "vocal_shimmer_relative",
]

# Features where LOW value = suspicious (from memory)
LOW_IS_SUSPICIOUS = {
    "blinking_variance", "mouth_movement_variance", "iris_jitter_variance",
    "vocal_jitter_relative", "vocal_shimmer_relative",
}

# Evidence JSON uses different names for audio features
EVIDENCE_AUDIO_KEY_MAP = {
    "wer_score": ("audio_visual", "wer_score"),
    "semantic_anomaly": ("audio_visual", "semantic_anomaly"),
    "min_cosine_anomaly": ("audio_visual", "min_cosine_anomaly"),
    "temporal_anomaly": ("audio_visual", "temporal_anomaly"),
    "min_temporal_anomaly": ("audio_visual", "min_temporal_anomaly"),
    "temporal_sync_variance": ("audio_visual", "temporal_sync_variance"),
    "vocal_jitter_relative": ("audio_visual", "vocal_jitter_relative"),
    "vocal_shimmer_relative": ("audio_visual", "vocal_shimmer_relative"),
}


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    labels = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_id = Path(row.get("local_path", "")).stem
            if video_id:
                labels[video_id] = row.get("label", "").strip().lower()
    return labels


def extract_features_from_evidence(evidence_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """For each video, get features from the chunk with highest anomaly score."""
    results = {}

    for p in sorted(evidence_dir.glob("*_evidence.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_id = data.get("video_metadata", {}).get("video_id", p.stem.replace("_evidence", ""))
        if data.get("video_metadata", {}).get("status") != "analyzed":
            continue

        chunks = data.get("chunks", {})
        if not chunks:
            continue

        best_chunk = None
        best_score = -1
        for chunk_id, chunk_info in chunks.items():
            score = chunk_info.get("anomaly", {}).get("joint_anomaly_score")
            if score is not None and score > best_score:
                best_score = score
                best_chunk = chunk_info

        if best_chunk is None:
            continue

        features = best_chunk.get("features", {})
        visual = features.get("visual", {})
        audio_visual = features.get("audio_visual", {})

        feat_values = {}
        for name in VISUAL_FEATURES:
            entry = visual.get(name, {})
            feat_values[name] = entry.get("value") if isinstance(entry, dict) else None

        for name in AUDIO_VISUAL_FEATURES:
            group, key = EVIDENCE_AUDIO_KEY_MAP.get(name, ("audio_visual", name))
            entry = audio_visual.get(key, {}) if group == "audio_visual" else visual.get(key, {})
            feat_values[name] = entry.get("value") if isinstance(entry, dict) else None

        results[video_id] = feat_values

    return results


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = 0.0
    for ps in pos:
        total += np.sum(neg < ps) + 0.5 * np.sum(neg == ps)
    return float(total / (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser(description="Per-feature AUC analysis")
    parser.add_argument("--evidence-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Loading manifest...", flush=True)
    labels = load_manifest(Path(args.manifest))

    print("Extracting features from evidence...", flush=True)
    video_features = extract_features_from_evidence(Path(args.evidence_dir))

    matched = sorted(set(labels.keys()) & set(video_features.keys()))
    print(f"Matched: {len(matched)} videos\n", flush=True)

    if not matched:
        print("ERROR: No matched videos.")
        return

    all_features = VISUAL_FEATURES + AUDIO_VISUAL_FEATURES
    results = []

    for feat_name in all_features:
        valid_ids = [vid for vid in matched if video_features[vid].get(feat_name) is not None]
        if len(valid_ids) < 100:
            results.append({
                "feature": feat_name,
                "auc": None,
                "auc_directed": None,
                "direction": "HIGH" if feat_name not in LOW_IS_SUSPICIOUS else "LOW",
                "n_valid": len(valid_ids),
                "note": "too_few_valid",
            })
            continue

        y = np.array([1 if labels[vid] == "fake" else 0 for vid in valid_ids])
        vals = np.array([video_features[vid][feat_name] for vid in valid_ids])

        auc_raw = compute_auc(y, vals)

        # For LOW_IS_SUSPICIOUS: negate so AUC > 0.5 means feature works
        if feat_name in LOW_IS_SUSPICIOUS:
            auc_directed = compute_auc(y, -vals)
            direction = "LOW=suspicious"
        else:
            auc_directed = auc_raw
            direction = "HIGH=suspicious"

        real_vals = vals[y == 0]
        fake_vals = vals[y == 1]

        results.append({
            "feature": feat_name,
            "group": "visual" if feat_name in VISUAL_FEATURES else "audio_visual",
            "direction": direction,
            "auc_raw": round(auc_raw, 4),
            "auc_directed": round(auc_directed, 4),
            "n_valid": len(valid_ids),
            "n_real": int(np.sum(y == 0)),
            "n_fake": int(np.sum(y == 1)),
            "real_median": round(float(np.median(real_vals)), 6),
            "fake_median": round(float(np.median(fake_vals)), 6),
            "real_mean": round(float(np.mean(real_vals)), 6),
            "fake_mean": round(float(np.mean(fake_vals)), 6),
        })

    # Sort by AUC directed descending
    results.sort(key=lambda x: x.get("auc_directed") or 0, reverse=True)

    # Print table
    print(f"{'Feature':<28} {'Direction':<18} {'AUC':>6} {'N':>6}  {'Real median':>12} {'Fake median':>12}")
    print("-" * 90)
    for r in results:
        auc_str = f"{r['auc_directed']:.4f}" if r.get("auc_directed") is not None else "N/A"
        real_med = f"{r.get('real_median', 0):.6f}" if r.get("real_median") is not None else "N/A"
        fake_med = f"{r.get('fake_median', 0):.6f}" if r.get("fake_median") is not None else "N/A"
        print(f"{r['feature']:<28} {r['direction']:<18} {auc_str:>6} {r['n_valid']:>6}  {real_med:>12} {fake_med:>12}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
