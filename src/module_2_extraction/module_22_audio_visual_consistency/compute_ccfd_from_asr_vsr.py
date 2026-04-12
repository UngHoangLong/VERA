#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


_CHUNK_JSON_RE = re.compile(r"chunk_(\d+)\.json$")


def chunk_sort_key(path: Path) -> Tuple[int, str]:
    m = _CHUNK_JSON_RE.match(path.name)
    idx = int(m.group(1)) if m else 10**9
    return idx, path.name


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = " ".join(text.split())
    return text


def levenshtein_words(ref_words: List[str], hyp_words: List[str]) -> int:
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        rw = ref_words[i - 1]
        for j in range(1, m + 1):
            cost = 0 if rw == hyp_words[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[m]


def compute_wer(reference: str, hypothesis: str) -> Dict:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    edit_distance = levenshtein_words(ref_words, hyp_words)

    if len(ref_words) == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer = edit_distance / float(len(ref_words))

    ccfd_score = 1.0 - min(wer, 1.0)

    return {
        "reference_text_raw": reference,
        "hypothesis_text_raw": hypothesis,
        "reference_text_norm": ref_norm,
        "hypothesis_text_norm": hyp_norm,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "edit_distance": int(edit_distance),
        "wer": float(round(wer, 6)),
        "ccfd_score": float(round(ccfd_score, 6)),
    }


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_chunk_paths(video_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for p in sorted(video_dir.glob("chunk_*.json"), key=chunk_sort_key):
        paths[p.stem] = p
    return paths


def build_video_summary(video_id: str, rows: List[Dict]) -> Dict:
    valid = [r for r in rows if r.get("ok")]
    failed = [r for r in rows if not r.get("ok")]

    if valid:
        wers = [r["wer"] for r in valid]
        scores = [r["ccfd_score"] for r in valid]
        worst = max(valid, key=lambda x: x["wer"])
        best = max(valid, key=lambda x: x["ccfd_score"])
        summary = {
            "video_id": video_id,
            "num_chunks": len(rows),
            "num_ok": len(valid),
            "num_failed": len(failed),
            "mean_wer": round(float(sum(wers) / len(wers)), 6),
            "median_wer": round(float(statistics.median(wers)), 6),
            "mean_ccfd_score": round(float(sum(scores) / len(scores)), 6),
            "median_ccfd_score": round(float(statistics.median(scores)), 6),
            "best_chunk": best["chunk"],
            "best_chunk_score": best["ccfd_score"],
            "worst_chunk": worst["chunk"],
            "worst_chunk_wer": worst["wer"],
        }
    else:
        summary = {
            "video_id": video_id,
            "num_chunks": len(rows),
            "num_ok": 0,
            "num_failed": len(rows),
            "mean_wer": None,
            "median_wer": None,
            "mean_ccfd_score": None,
            "median_ccfd_score": None,
            "best_chunk": None,
            "best_chunk_score": None,
            "worst_chunk": None,
            "worst_chunk_wer": None,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete CCFD by using ASR text as reference, VSR text as hypothesis, and computing WER per chunk."
    )
    parser.add_argument("--asr-root", type=str, required=True, help="Example: ./src/module_2_extraction/output/asr_output")
    parser.add_argument("--vsr-root", type=str, required=True, help="Example: ./src/module_2_extraction/output/vsr_output")
    parser.add_argument("--output-root", type=str, required=True, help="Example: ./src/module_2_extraction/output/ccfd_output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asr_root = Path(args.asr_root)
    vsr_root = Path(args.vsr_root)
    output_root = Path(args.output_root)

    if not asr_root.exists():
        raise FileNotFoundError(f"Khong ton tai asr_root: {asr_root}")
    if not vsr_root.exists():
        raise FileNotFoundError(f"Khong ton tai vsr_root: {vsr_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(
        {p.name for p in asr_root.iterdir() if p.is_dir()} |
        {p.name for p in vsr_root.iterdir() if p.is_dir()}
    )

    all_rows: List[Dict] = []
    video_summaries: List[Dict] = []

    for video_id in video_ids:
        asr_video_dir = asr_root / video_id
        vsr_video_dir = vsr_root / video_id
        out_video_dir = output_root / video_id
        out_video_dir.mkdir(parents=True, exist_ok=True)

        asr_chunks = collect_chunk_paths(asr_video_dir) if asr_video_dir.exists() else {}
        vsr_chunks = collect_chunk_paths(vsr_video_dir) if vsr_video_dir.exists() else {}

        chunk_names = sorted(set(asr_chunks) | set(vsr_chunks), key=lambda x: chunk_sort_key(Path(f"{x}.json")))

        video_rows: List[Dict] = []
        for chunk_name in chunk_names:
            out_json = out_video_dir / f"{chunk_name}.json"
            if out_json.exists() and not args.overwrite:
                row = load_json(out_json)
                video_rows.append(row)
                all_rows.append(row)
                continue

            row = {
                "video_id": video_id,
                "chunk": chunk_name,
                "asr_json": str(asr_chunks[chunk_name]) if chunk_name in asr_chunks else None,
                "vsr_json": str(vsr_chunks[chunk_name]) if chunk_name in vsr_chunks else None,
                "output_json": str(out_json),
                "ok": False,
                "reason": None,
            }

            if chunk_name not in asr_chunks:
                row["reason"] = "missing_asr_chunk"
            elif chunk_name not in vsr_chunks:
                row["reason"] = "missing_vsr_chunk"
            else:
                asr_data = load_json(asr_chunks[chunk_name])
                vsr_data = load_json(vsr_chunks[chunk_name])

                if not asr_data.get("ok", False):
                    row["reason"] = f"asr_not_ok: {asr_data.get('reason')}"
                elif not vsr_data.get("ok", False):
                    row["reason"] = f"vsr_not_ok: {vsr_data.get('reason')}"
                else:
                    reference_text = asr_data.get("text") or ""
                    hypothesis_text = vsr_data.get("hypothesis") or ""
                    row.update(compute_wer(reference_text, hypothesis_text))
                    row["ok"] = True
                    row["reason"] = "success"

            out_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            video_rows.append(row)
            all_rows.append(row)

        summary = build_video_summary(video_id, video_rows)
        summary_path = out_video_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        video_summaries.append(summary)

    manifest = {
        "asr_root": str(asr_root),
        "vsr_root": str(vsr_root),
        "output_root": str(output_root),
        "num_videos": len(video_ids),
        "num_chunks": len(all_rows),
        "num_ok": sum(1 for r in all_rows if r.get("ok")),
        "num_failed": sum(1 for r in all_rows if not r.get("ok")),
        "video_summaries": video_summaries,
        "results": all_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "output_root": str(output_root),
            "manifest": str(manifest_path),
            "num_videos": manifest["num_videos"],
            "num_chunks": manifest["num_chunks"],
            "num_ok": manifest["num_ok"],
            "num_failed": manifest["num_failed"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
