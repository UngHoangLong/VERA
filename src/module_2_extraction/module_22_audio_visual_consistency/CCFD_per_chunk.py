#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from num2words import num2words


_CHUNK_JSON_RE = re.compile(r"chunk_(\d+)\.json$")


# Tạo khóa để sắp xếp file chunk_*.json đúng thứ tự chunk.
def chunk_sort_key(path: Path) -> Tuple[int, str]:
    m = _CHUNK_JSON_RE.match(path.name)
    idx = int(m.group(1)) if m else 10**9
    return idx, path.name


# Chuyển đổi số thành chữ. Vì với model ASR nó đủ tốt để chuyển thành số, nhưng với VSR thường nó sẽ ghi full chữ.
def replace_numbers_with_words(text: str) -> str:
    def _replace(match):
        try:
            return num2words(int(match.group())).replace("-", " ")
        except Exception:
            return match.group()

    return re.sub(r"\b\d+\b", _replace, text)


CONTRACTIONS = {
    "it's": "it is", "don't": "do not", "doesn't": "does not", "can't": "cannot",
    "i'm": "i am", "that's": "that is", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "you're": "you are", "we're": "we are",
    "they're": "they are", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "won't": "will not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "he's": "he is", "she's": "she is", "there's": "there is", "let's": "let us",
    "what's": "what is", "who's": "who is", "where's": "where is", "how's": "how is",
    "y'all": "you all",
}

CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b")


def expand_contractions(text: str) -> str:
    def replace(match):
        return CONTRACTIONS[match.group(1)]

    return CONTRACTION_RE.sub(replace, text)


# Chuẩn hóa văn bản trước khi so sánh.
def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'")
    text = expand_contractions(text)
    text = replace_numbers_with_words(text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()
    fillers = {"um", "uh", "ah", "er", "hmm", "mhmm", "huh"}
    words = [w for w in words if w not in fillers]
    return " ".join(words)


# Tính số phép chỉnh sửa khác nhau giữa 2 câu ở mức từ.
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


# Tính WER và suy ra ccfd_score từ text ASR và VSR.
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


# Đọc file JSON.
def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Gom các file chunk của một video thành danh sách/ánh xạ để xử lý.
def collect_chunk_paths(video_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for p in sorted(video_dir.glob("chunk_*.json"), key=chunk_sort_key):
        paths[p.stem] = p
    return paths


def parse_cuda_visible_devices() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def auto_num_workers(num_tasks: int) -> int:
    visible_gpus = parse_cuda_visible_devices()
    if visible_gpus:
        return max(1, min(len(visible_gpus), num_tasks))
    return 1


# Tổng hợp kết quả toàn bộ chunk của một video.
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


def build_tasks(asr_root: Path, vsr_root: Path, output_root: Path) -> List[Dict]:
    video_ids = sorted(
        {p.name for p in asr_root.iterdir() if p.is_dir()} |
        {p.name for p in vsr_root.iterdir() if p.is_dir()}
    )

    tasks: List[Dict] = []
    for video_id in video_ids:
        asr_video_dir = asr_root / video_id
        vsr_video_dir = vsr_root / video_id
        asr_chunks = collect_chunk_paths(asr_video_dir) if asr_video_dir.exists() else {}
        vsr_chunks = collect_chunk_paths(vsr_video_dir) if vsr_video_dir.exists() else {}
        chunk_names = sorted(set(asr_chunks) | set(vsr_chunks), key=lambda x: chunk_sort_key(Path(f"{x}.json")))

        for chunk_name in chunk_names:
            out_json = output_root / video_id / f"{chunk_name}.json"
            tasks.append({
                "video_id": video_id,
                "chunk": chunk_name,
                "asr_json": str(asr_chunks[chunk_name]) if chunk_name in asr_chunks else None,
                "vsr_json": str(vsr_chunks[chunk_name]) if chunk_name in vsr_chunks else None,
                "output_json": str(out_json),
            })

    return tasks


def process_task(payload: Tuple[Dict, bool]) -> Dict:
    task, overwrite = payload
    video_id = task["video_id"]
    chunk_name = task["chunk"]
    out_json = Path(task["output_json"])

    if out_json.exists() and not overwrite:
        return load_json(out_json)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "video_id": video_id,
        "chunk": chunk_name,
        "asr_json": task.get("asr_json"),
        "vsr_json": task.get("vsr_json"),
        "output_json": str(out_json),
        "ok": False,
        "reason": None,
    }

    try:
        if not task.get("asr_json"):
            row["reason"] = "missing_asr_chunk"
        elif not task.get("vsr_json"):
            row["reason"] = "missing_vsr_chunk"
        else:
            asr_data = load_json(Path(task["asr_json"]))
            vsr_data = load_json(Path(task["vsr_json"]))

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
    except Exception as exc:
        row["ok"] = False
        row["reason"] = f"error: {exc}"

    out_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def write_summaries_and_manifest(
    asr_root: Path,
    vsr_root: Path,
    output_root: Path,
    rows: List[Dict],
    num_workers: int,
) -> None:
    rows = sorted(rows, key=lambda r: (r.get("video_id", ""), chunk_sort_key(Path(f"{r.get('chunk', '')}.json"))))

    rows_by_video: Dict[str, List[Dict]] = {}
    for row in rows:
        rows_by_video.setdefault(row["video_id"], []).append(row)

    video_summaries: List[Dict] = []
    for video_id, video_rows in sorted(rows_by_video.items()):
        summary = build_video_summary(video_id, video_rows)
        summary_path = output_root / video_id / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        video_summaries.append(summary)

    manifest = {
        "asr_root": str(asr_root),
        "vsr_root": str(vsr_root),
        "output_root": str(output_root),
        "output_layout": str(output_root / "<video_id>" / "chunk_*.json"),
        "num_workers": num_workers,
        "cuda_visible_devices": parse_cuda_visible_devices(),
        "num_videos": len(rows_by_video),
        "num_chunks": len(rows),
        "num_ok": sum(1 for r in rows if r.get("ok")),
        "num_failed": sum(1 for r in rows if not r.get("ok")),
        "video_summaries": video_summaries,
        "results": rows,
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "mode": "ccfd_parallel_workers" if num_workers > 1 else "ccfd_single_worker",
            "output_root": str(output_root),
            "output_layout": str(output_root / "<video_id>" / "chunk_*.json"),
            "manifest": str(manifest_path),
            "num_workers": num_workers,
            "cuda_visible_devices": parse_cuda_visible_devices(),
            "num_videos": manifest["num_videos"],
            "num_chunks": manifest["num_chunks"],
            "num_ok": manifest["num_ok"],
            "num_failed": manifest["num_failed"],
        },
        ensure_ascii=False,
        indent=2,
    ))


# Đọc input, duyệt video/chunk, tính kết quả, lưu theo layout output_root/<video_id>/chunk_*.json.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete CCFD by using ASR text as reference, VSR text as hypothesis, and computing WER per chunk."
    )
    parser.add_argument("--asr-root", type=str, required=True, help="Example: ./data/processed/asr_output")
    parser.add_argument("--vsr-root", type=str, required=True, help="Example: ./data/processed/vsr_output")
    parser.add_argument("--output-root", type=str, required=True, help="Layout: <output-root>/<video_id>/chunk_*.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0, help="0 = auto. Nếu CUDA_VISIBLE_DEVICES có N GPU thì dùng N worker. CCFD vẫn là xử lý CPU/text.")
    args = parser.parse_args()

    asr_root = Path(args.asr_root)
    vsr_root = Path(args.vsr_root)
    output_root = Path(args.output_root)

    if not asr_root.exists():
        raise FileNotFoundError(f"Khong ton tai asr_root: {asr_root}")
    if not vsr_root.exists():
        raise FileNotFoundError(f"Khong ton tai vsr_root: {vsr_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(asr_root, vsr_root, output_root)

    if not tasks:
        raise RuntimeError("Không tìm thấy chunk ASR/VSR nào để xử lý.")

    num_workers = args.num_workers if args.num_workers and args.num_workers > 0 else auto_num_workers(len(tasks))
    num_workers = max(1, min(num_workers, len(tasks)))

    print(json.dumps(
        {
            "mode": "ccfd_parallel_workers" if num_workers > 1 else "ccfd_single_worker",
            "note": "CCFD chỉ xử lý text/JSON, không dùng CUDA trực tiếp. CUDA_VISIBLE_DEVICES chỉ dùng để suy ra số worker mặc định.",
            "output_layout": str(output_root / "<video_id>" / "chunk_*.json"),
            "num_workers": num_workers,
            "num_tasks": len(tasks),
            "cuda_visible_devices": parse_cuda_visible_devices(),
        },
        ensure_ascii=False,
        indent=2,
    ))

    payloads = [(task, bool(args.overwrite)) for task in tasks]
    if num_workers == 1:
        rows = [process_task(payload) for payload in payloads]
    else:
        with Pool(processes=num_workers) as pool:
            rows = list(pool.imap_unordered(process_task, payloads))

    write_summaries_and_manifest(
        asr_root=asr_root,
        vsr_root=vsr_root,
        output_root=output_root,
        rows=rows,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
