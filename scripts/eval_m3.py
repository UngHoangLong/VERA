"""
Level-1 evaluation of Module 3: video score = max joint_anomaly_score over chunks.

Reports AUC and F1/recall/specificity at the model's own threshold, overall and
per generative method (each fake method vs. all genuine "real" videos).

    python scripts/eval_m3.py --evidence_dir <dir of *_evidence.json> \
        --manifest data/external/mavos_dd_en/manifest.csv \
        --threshold_json src/track_a/module_3_autoencoder/module3_models/threshold.json
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from sklearn.metrics import roc_auc_score


def load_labels(manifest_path):
    """(method, stem) -> set of labels. 'real' folder holds real + audio-only fakes."""
    lab = defaultdict(set)
    for r in csv.DictReader(open(manifest_path, encoding="utf-8")):
        if not r["usage"].startswith("eval"):
            continue
        parts = r["local_path"].split("/")
        stem = os.path.splitext(parts[-1])[0]
        lab[(parts[1], stem)].add(r["label"])
    return lab


def video_key(video_id):
    if "__" in video_id:
        m, s = video_id.split("__", 1)
        return m, s
    return "real", video_id


def metrics(rows, thr):
    y = [r["y"] for r in rows]
    s = [r["score"] for r in rows]
    pred = [int(x >= thr) for x in s]
    tp = sum(1 for a, b in zip(y, pred) if a and b)
    fn = sum(1 for a, b in zip(y, pred) if a and not b)
    fp = sum(1 for a, b in zip(y, pred) if not a and b)
    tn = sum(1 for a, b in zip(y, pred) if not a and not b)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
    return dict(n=len(y), pos=sum(y), auc=auc, f1=f1, recall=rec, specificity=spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--threshold_json", required=True)
    a = ap.parse_args()

    thr = json.load(open(a.threshold_json))["threshold"]
    labels = load_labels(a.manifest)

    rows, skipped = [], Counter()
    for fn in os.listdir(a.evidence_dir):
        if not fn.endswith("_evidence.json"):
            continue
        j = json.load(open(os.path.join(a.evidence_dir, fn)))
        vid = j["video_metadata"]["video_id"]
        method, stem = video_key(vid)
        chunks = j.get("chunks") or {}
        if not chunks:
            skipped[f"no_chunks:{method}"] += 1
            continue
        ls = labels.get((method, stem), set())
        if len(ls) != 1:
            skipped["ambiguous_or_unknown_label"] += 1
            continue
        vals = [c["anomaly"]["joint_anomaly_score"] for c in chunks.values()]
        vals = [v for v in vals if v is not None and v == v]  # drop null/NaN chunk scores
        if not vals:
            skipped["all_chunk_scores_nan"] += 1
            continue
        score = max(vals)
        rows.append(dict(method=method, y=int(next(iter(ls)) == "fake"), score=score))

    print(f"threshold = {thr:.4f}; videos scored = {len(rows)}")
    print("skipped:", dict(skipped))
    print(f"\n{'subset':<28}{'n':>6}{'pos':>6}{'AUC':>8}{'F1':>8}{'recall':>8}{'spec':>8}")

    def show(name, rs):
        m = metrics(rs, thr)
        print(f"{name:<28}{m['n']:>6}{m['pos']:>6}{m['auc']:>8.3f}{m['f1']:>8.3f}"
              f"{m['recall']:>8.3f}{m['specificity']:>8.3f}")

    show("ALL (real folder + fakes)", rows)
    real = [r for r in rows if r["method"] == "real" and r["y"] == 0]
    audio_fake = [r for r in rows if r["method"] == "real" and r["y"] == 1]
    show("real folder (real vs audio-fake)", [r for r in rows if r["method"] == "real"])
    for m in sorted({r["method"] for r in rows} - {"real"}):
        show(f"{m} vs real", [r for r in rows if r["method"] == m] + real)
    print(f"\ngenuine videos scored: {len(real)}, audio-only fakes (in 'real' folder): {len(audio_fake)}")


if __name__ == "__main__":
    main()
