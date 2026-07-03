#!/usr/bin/env python3
"""
Level 1 Evaluation: Module 3 anomaly scores vs ground truth labels.

Aggregation: max chunk joint_anomaly_score per video.
Labels: from manifest.csv (video_path → label, generative_method, etc.)

Usage:
    python scripts/evaluate_level1.py \
      --evidence-dir src/module_3_autoencoder/evidence_reports/infer \
      --manifest data/external/mavos_dd_en/manifest.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    """Parse manifest.csv → dict keyed by video_id."""
    labels = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = row.get("local_path", "")
            video_id = Path(local_path).stem
            if not video_id:
                continue

            video_fake = row.get("video_fake", "").strip().lower() == "true"
            audio_fake = row.get("audio_fake", "").strip().lower() == "true"

            if video_fake and audio_fake:
                fake_type = "both"
            elif video_fake:
                fake_type = "video_only"
            elif audio_fake:
                fake_type = "audio_only"
            else:
                fake_type = "none"

            labels[video_id] = {
                "label": row.get("label", "").strip().lower(),
                "video_fake": video_fake,
                "audio_fake": audio_fake,
                "fake_type": fake_type,
                "generative_method": row.get("generative_method", "").strip(),
                "audio_generative_method": row.get("audio_generative_method", "").strip(),
            }
    return labels


def load_evidence_scores(evidence_dir: Path) -> Dict[str, float]:
    """Read evidence JSONs → dict of video_id → max chunk joint_anomaly_score."""
    scores = {}
    for p in sorted(evidence_dir.glob("*_evidence.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_id = data.get("video_metadata", {}).get("video_id", p.stem.replace("_evidence", ""))
        status = data.get("video_metadata", {}).get("status", "")
        if status != "analyzed":
            continue

        chunks = data.get("chunks", {})
        if not chunks:
            continue

        chunk_scores = []
        for chunk_info in chunks.values():
            anomaly = chunk_info.get("anomaly", {})
            js = anomaly.get("joint_anomaly_score")
            if js is not None:
                chunk_scores.append(float(js))

        if chunk_scores:
            scores[video_id] = max(chunk_scores)

    return scores


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn dependency."""
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    total = 0.0
    for ps in pos_scores:
        total += np.sum(neg_scores < ps) + 0.5 * np.sum(neg_scores == ps)

    auc = total / (len(pos_scores) * len(neg_scores))
    return round(float(auc), 4)


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """Find threshold that maximizes F1."""
    thresholds = np.percentile(scores, np.arange(1, 100))
    thresholds = np.unique(thresholds)

    best_f1 = -1.0
    best_threshold = 0.0
    best_metrics = {}

    for t in thresholds:
        y_pred = (scores > t).astype(int)
        m = compute_binary_metrics(y_true, y_pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = float(t)
            best_metrics = m

    return best_threshold, best_metrics


def evaluate(
    labels: Dict[str, Dict],
    scores: Dict[str, float],
    threshold: float,
) -> Dict:
    matched_ids = sorted(set(labels.keys()) & set(scores.keys()))
    unmatched_labels = sorted(set(labels.keys()) - set(scores.keys()))
    unmatched_scores = sorted(set(scores.keys()) - set(labels.keys()))

    if not matched_ids:
        return {"error": "No matched video_ids between manifest and evidence."}

    y_true = np.array([1 if labels[vid]["label"] == "fake" else 0 for vid in matched_ids])
    y_scores = np.array([scores[vid] for vid in matched_ids])

    # --- Overall binary ---
    auc = compute_auc_roc(y_true, y_scores)
    y_pred_fixed = (y_scores > threshold).astype(int)
    metrics_fixed = compute_binary_metrics(y_true, y_pred_fixed)

    optimal_threshold, metrics_optimal = find_optimal_threshold(y_true, y_scores)

    # --- Per fake_type ---
    fake_type_results = {}
    for vid in matched_ids:
        ft = labels[vid]["fake_type"]
        if ft == "none":
            ft = "real"
        if ft not in fake_type_results:
            fake_type_results[ft] = {"y_true": [], "y_scores": []}
        fake_type_results[ft]["y_true"].append(1 if labels[vid]["label"] == "fake" else 0)
        fake_type_results[ft]["y_scores"].append(scores[vid])

    fake_type_summary = {}
    for ft, data in sorted(fake_type_results.items()):
        yt = np.array(data["y_true"])
        ys = np.array(data["y_scores"])
        yp = (ys > threshold).astype(int)
        fake_type_summary[ft] = {
            "count": len(yt),
            "mean_score": round(float(np.mean(ys)), 4),
            "median_score": round(float(np.median(ys)), 4),
            "std_score": round(float(np.std(ys)), 4),
            "pct_above_threshold": round(float(np.mean(ys > threshold) * 100), 2),
        }

    # --- Per generative_method ---
    # generative_method only tracks VIDEO manipulation; audio-only fakes
    # (video untouched, audio cloned) also have generative_method == "real".
    # Bucket those separately so "real" only contains truly genuine videos.
    method_results = defaultdict(list)
    for vid in matched_ids:
        method = labels[vid]["generative_method"] or "real"
        if method == "real" and labels[vid]["label"] == "fake":
            method = "real_video_audio_only_fake"
        method_results[method].append(scores[vid])

    method_summary = {}
    for method, method_scores in sorted(method_results.items()):
        ms = np.array(method_scores)
        method_summary[method] = {
            "count": len(ms),
            "mean_score": round(float(np.mean(ms)), 4),
            "median_score": round(float(np.median(ms)), 4),
            "std_score": round(float(np.std(ms)), 4),
            "min_score": round(float(np.min(ms)), 4),
            "max_score": round(float(np.max(ms)), 4),
            "pct_above_threshold": round(float(np.mean(ms > threshold) * 100), 2),
        }

    # --- Score distribution ---
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]

    return {
        "summary": {
            "total_matched": len(matched_ids),
            "total_real": int(np.sum(y_true == 0)),
            "total_fake": int(np.sum(y_true == 1)),
            "unmatched_in_manifest": len(unmatched_labels),
            "unmatched_in_evidence": len(unmatched_scores),
        },
        "overall_binary": {
            "auc_roc": auc,
            f"metrics_at_threshold_{threshold}": metrics_fixed,
            "optimal_threshold": round(optimal_threshold, 4),
            "metrics_at_optimal_threshold": metrics_optimal,
        },
        "score_distribution": {
            "real": {
                "count": len(real_scores),
                "mean": round(float(np.mean(real_scores)), 4) if len(real_scores) > 0 else None,
                "median": round(float(np.median(real_scores)), 4) if len(real_scores) > 0 else None,
                "std": round(float(np.std(real_scores)), 4) if len(real_scores) > 0 else None,
                "p95": round(float(np.percentile(real_scores, 95)), 4) if len(real_scores) > 0 else None,
            },
            "fake": {
                "count": len(fake_scores),
                "mean": round(float(np.mean(fake_scores)), 4) if len(fake_scores) > 0 else None,
                "median": round(float(np.median(fake_scores)), 4) if len(fake_scores) > 0 else None,
                "std": round(float(np.std(fake_scores)), 4) if len(fake_scores) > 0 else None,
                "p5": round(float(np.percentile(fake_scores, 5)), 4) if len(fake_scores) > 0 else None,
            },
        },
        "per_fake_type": fake_type_summary,
        "per_method": method_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Level 1 Evaluation: Module 3 anomaly scores vs labels")
    parser.add_argument("--evidence-dir", type=str, required=True,
                        help="Directory containing *_evidence.json from Module 3 infer")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.csv with ground truth labels")
    parser.add_argument("--threshold", type=float, default=15.54,
                        help="Anomaly score threshold (default: 15.54 from 95th percentile genuine val)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("Loading manifest...", flush=True)
    labels = load_manifest(Path(args.manifest))
    print(f"  {len(labels)} videos in manifest", flush=True)

    print("Loading evidence scores...", flush=True)
    scores = load_evidence_scores(Path(args.evidence_dir))
    print(f"  {len(scores)} videos with scores", flush=True)

    print(f"Evaluating (threshold={args.threshold})...", flush=True)
    results = evaluate(labels, scores, args.threshold)

    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    print("\n" + output_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
