#!/usr/bin/env python3
"""
Level 2 Evaluation: Module 5 MLLM verdicts vs ground truth labels.

Compares verdict JSON (video_fake, audio_fake, label) against manifest.csv labels.

Usage:
    python scripts/evaluate_level2.py \
      --verdicts-dir src/module_5_agent/verdicts_qwen \
      --manifest data/external/mavos_dd_en/manifest.csv \
      --backend qwen
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    labels = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_id = Path(row.get("local_path", "")).stem
            if not video_id:
                continue
            video_fake = row.get("video_fake", "").strip().lower() == "true"
            audio_fake = row.get("audio_fake", "").strip().lower() == "true"
            labels[video_id] = {
                "label": row.get("label", "").strip().lower(),
                "video_fake": video_fake,
                "audio_fake": audio_fake,
                "generative_method": row.get("generative_method", "").strip(),
                "audio_generative_method": row.get("audio_generative_method", "").strip(),
            }
    return labels


def load_verdicts(verdicts_dir: Path) -> Dict[str, Dict]:
    verdicts = {}
    for p in sorted(verdicts_dir.glob("*_verdict.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        video_id = data.get("video_id", p.stem.replace("_verdict", ""))
        verdict = data.get("verdict")
        if not isinstance(verdict, dict):
            continue
        verdicts[video_id] = verdict
    return verdicts


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(labels: Dict, verdicts: Dict, backend: str) -> Dict:
    matched = sorted(set(labels.keys()) & set(verdicts.keys()))

    if not matched:
        return {"error": "No matched video_ids."}

    # Filter out UNCERTAIN verdicts for main metrics
    certain = [vid for vid in matched if verdicts[vid].get("label") in ("FAKE", "GENUINE")]
    uncertain = [vid for vid in matched if verdicts[vid].get("label") not in ("FAKE", "GENUINE")]

    # --- Overall binary (label: FAKE vs GENUINE) ---
    gt_label = np.array([1 if labels[vid]["label"] == "fake" else 0 for vid in certain])
    pred_label = np.array([1 if verdicts[vid].get("label") == "FAKE" else 0 for vid in certain])
    overall = compute_metrics(gt_label, pred_label)

    # --- Video fake ---
    gt_vf = np.array([1 if labels[vid]["video_fake"] else 0 for vid in certain])
    pred_vf = np.array([1 if verdicts[vid].get("video_fake") else 0 for vid in certain])
    video_fake_metrics = compute_metrics(gt_vf, pred_vf)

    # --- Audio fake ---
    gt_af = np.array([1 if labels[vid]["audio_fake"] else 0 for vid in certain])
    pred_af = np.array([1 if verdicts[vid].get("audio_fake") else 0 for vid in certain])
    audio_fake_metrics = compute_metrics(gt_af, pred_af)

    # --- Per generative_method ---
    # generative_method only tracks VIDEO manipulation; audio-only fakes
    # (video untouched, audio cloned) also have generative_method == "real".
    # Bucket those separately so "real" only contains truly genuine videos.
    method_results = defaultdict(lambda: {"gt": [], "pred": []})
    for vid in certain:
        method = labels[vid]["generative_method"] or "real"
        if method == "real" and labels[vid]["label"] == "fake":
            method = "real_video_audio_only_fake"
        gt = 1 if labels[vid]["label"] == "fake" else 0
        pred = 1 if verdicts[vid].get("label") == "FAKE" else 0
        method_results[method]["gt"].append(gt)
        method_results[method]["pred"].append(pred)

    method_summary = {}
    for method, data in sorted(method_results.items()):
        gt = np.array(data["gt"])
        pred = np.array(data["pred"])
        m = compute_metrics(gt, pred)
        method_summary[method] = m

    # --- Per confidence level ---
    confidence_dist = defaultdict(int)
    for vid in matched:
        conf = verdicts[vid].get("confidence", "UNKNOWN")
        confidence_dist[conf] += 1

    # --- Verdict distribution ---
    verdict_dist = defaultdict(int)
    for vid in matched:
        lbl = verdicts[vid].get("label", "UNKNOWN")
        verdict_dist[lbl] += 1

    # --- Per fake type (video_only, audio_only, both, real) ---
    fake_type_results = defaultdict(lambda: {"gt": [], "pred": []})
    for vid in certain:
        vf = labels[vid]["video_fake"]
        af = labels[vid]["audio_fake"]
        if vf and af:
            ft = "both"
        elif vf:
            ft = "video_only"
        elif af:
            ft = "audio_only"
        else:
            ft = "real"
        gt = 1 if labels[vid]["label"] == "fake" else 0
        pred = 1 if verdicts[vid].get("label") == "FAKE" else 0
        fake_type_results[ft]["gt"].append(gt)
        fake_type_results[ft]["pred"].append(pred)

    fake_type_summary = {}
    for ft, data in sorted(fake_type_results.items()):
        gt = np.array(data["gt"])
        pred = np.array(data["pred"])
        fake_type_summary[ft] = compute_metrics(gt, pred)

    return {
        "backend": backend,
        "summary": {
            "total_matched": len(matched),
            "total_certain": len(certain),
            "total_uncertain": len(uncertain),
            "total_real_gt": int(np.sum(gt_label == 0)),
            "total_fake_gt": int(np.sum(gt_label == 1)),
        },
        "verdict_distribution": dict(verdict_dist),
        "confidence_distribution": dict(confidence_dist),
        "overall_binary": overall,
        "video_fake_detection": video_fake_metrics,
        "audio_fake_detection": audio_fake_metrics,
        "per_fake_type": fake_type_summary,
        "per_method": method_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Level 2 Evaluation: MLLM verdicts vs labels")
    parser.add_argument("--verdicts-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--backend", type=str, default="qwen")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Loading manifest...", flush=True)
    labels = load_manifest(Path(args.manifest))
    print(f"  {len(labels)} videos in manifest", flush=True)

    print("Loading verdicts...", flush=True)
    verdicts = load_verdicts(Path(args.verdicts_dir))
    print(f"  {len(verdicts)} verdicts loaded", flush=True)

    results = evaluate(labels, verdicts, args.backend)

    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    print("\n" + output_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
