#!/usr/bin/env python3
"""
Select genuine_val videos with the most chunks to balance the Level 2 test set.

Reads manifest.csv for usage=="genuine_val" rows, cross-references with
final_reports_genuine/ to count valid chunks per video, and outputs the
top-N video_ids sorted by chunk count descending.

Usage:
    python scripts/select_genuine_val_for_balance.py \
      --manifest data/external/mavos_dd_en/manifest.csv \
      --reports-dir final_reports_genuine \
      --n 712 \
      --output results/genuine_val_selected.txt
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--reports-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=712)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    val_video_ids = set()
    with open(args.manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("usage") == "genuine_val":
                vid = Path(row["local_path"]).stem
                val_video_ids.add(vid)

    print(f"genuine_val videos in manifest: {len(val_video_ids)}")

    chunk_counts = []
    for vid in val_video_ids:
        report_path = reports_dir / f"{vid}_report.json"
        if not report_path.exists():
            continue
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunks = data.get("chunks", {})
        if chunks:
            chunk_counts.append((vid, len(chunks)))

    print(f"genuine_val videos with valid chunks: {len(chunk_counts)}")

    chunk_counts.sort(key=lambda x: x[1], reverse=True)
    selected = chunk_counts[: args.n]

    print(f"Selected: {len(selected)} (requested {args.n})")
    if len(selected) < args.n:
        print(f"WARNING: only {len(selected)} available, less than requested {args.n}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for vid, n_chunks in selected:
            f.write(f"{vid}\n")

    print(f"Saved video_ids to {out_path}")
    if selected:
        print(f"Chunk count range: {selected[0][1]} (max) to {selected[-1][1]} (min)")


if __name__ == "__main__":
    main()
