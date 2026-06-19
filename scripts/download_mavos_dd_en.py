"""
Download + organize the MAVOS-DD English subset needed for Module 3 (MVAE-PoE).

What gets downloaded (15,165 clips total, ~38GB):
    genuine_train  -> split=='train'      & label=='real'   (3,234 clips)
    genuine_val    -> split=='validation' & label=='real'   (  513 clips)
    eval_track1    -> split=='test' & open_set_model==False (3,474 clips)
    eval_track2    -> split=='test' & open_set_model==True  (7,944 clips)

Output layout -- matches this project's --mode {genuine,infer} convention,
so the result can be dropped straight into data/raw/genuine and data/raw/infer:
    <output_dir>/
      manifest.csv              <- one row per clip, tracks usage/labels/download status
      genuine/
        train/<file>.mp4        <- genuine_train (3,234 clips, always generative_method=='real')
        val/<file>.mp4          <- genuine_val   (  513 clips, always generative_method=='real')
      infer/
        real/<file>.mp4         <- test-split clips, grouped by generative_method
        inswapper/<file>.mp4
        hififace/<file>.mp4
        roop/<file>.mp4
        knnvc/<file>.mp4
        echomimic/<file>.mp4
        liveportrait/<file>.mp4
        sonic/<file>.mp4
        memo/<file>.mp4
        freevc/<file>.mp4
        (11,418 clips total; manifest.csv 'usage' column distinguishes
         eval_track1 vs eval_track2 for per-track metrics)

Resumable / checkpoint-friendly:
    - On first run, the manifest is built from MAVOS-DD's metadata and saved.
    - Every file's status (pending/done/failed) is written back to manifest.csv
      after each small batch, so a network drop never loses progress.
    - Re-running the script reuses the existing manifest: files already
      marked "done" are skipped entirely. "pending" files are retried
      automatically; "failed" files are retried only with --retry_failed.
    - hf_hub_download() itself also skips re-downloading a file that is
      already correctly present on disk, and resumes partial downloads
      via HTTP range requests when possible.

Setup:
    pip install huggingface_hub pyarrow tqdm

Usage:
    # download everything into data/external/mavos_dd_en/
    python scripts/download_mavos_dd_en.py

    # download to a Google Drive mount (e.g. on Colab)
    python scripts/download_mavos_dd_en.py --output_dir /content/drive/MyDrive/mavos_dd_en

    # only download the genuine train/val pools first
    python scripts/download_mavos_dd_en.py --usage genuine_train genuine_val

    # after a run with failures, retry just the failed files
    python scripts/download_mavos_dd_en.py --retry_failed

    # safe to Ctrl+C at any time; just re-run the same command to resume
"""

import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "unibuc-cs/MAVOS-DD"
REPO_TYPE = "dataset"
METADATA_FILENAME = "data-00000-of-00001.arrow"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "mavos_dd_en"
MANIFEST_NAME = "manifest.csv"
FLUSH_EVERY = 25  # write manifest.csv back to disk every N completed files

MANIFEST_FIELDS = [
    "video_path", "local_path", "usage", "split", "label",
    "video_fake", "audio_fake",
    "generative_method", "audio_generative_method",
    "open_set_model", "open_set_language", "language",
    "status", "error",
]


def load_english_metadata():
    """Download MAVOS-DD's metadata arrow file and return English rows as dicts."""
    print(f"Downloading metadata ({METADATA_FILENAME}) ...")
    path = hf_hub_download(REPO_ID, METADATA_FILENAME, repo_type=REPO_TYPE)
    with pa.memory_map(path, "r") as source:
        table = ipc.open_stream(source).read_all()
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    n = len(cols["video_path"])
    rows = [{k: cols[k][i] for k in cols} for i in range(n)]
    english_rows = [r for r in rows if r["video_path"].startswith("english/")]
    print(f"  -> {len(english_rows)} English rows total")
    return english_rows


def classify_usage(row):
    """Return the usage bucket for this row, or None if it's not needed."""
    split, label = row["split"], row["label"]
    if split == "train" and label == "real":
        return "genuine_train"
    if split == "validation" and label == "real":
        return "genuine_val"
    if split == "test":
        return "eval_track2" if row["open_set_model"] else "eval_track1"
    return None


def dest_relpath(usage, generative_method, video_path):
    """Where this clip lands under output_dir, matching data/raw/<mode>/ layout."""
    filename = Path(video_path).name
    if usage == "genuine_train":
        return Path("genuine") / "train" / filename
    if usage == "genuine_val":
        return Path("genuine") / "val" / filename
    return Path("infer") / generative_method / filename


def build_manifest(english_rows):
    manifest = []
    for row in english_rows:
        usage = classify_usage(row)
        if usage is None:
            continue
        local_path = dest_relpath(usage, row["generative_method"], row["video_path"])
        manifest.append({
            "video_path": row["video_path"],
            "local_path": str(local_path),
            "usage": usage,
            "split": row["split"],
            "label": row["label"],
            "video_fake": row["video_fake"],
            "audio_fake": row["audio_fake"],
            "generative_method": row["generative_method"],
            "audio_generative_method": row["audio_generative_method"],
            "open_set_model": row["open_set_model"],
            "open_set_language": row["open_set_language"],
            "language": row["language"],
            "status": "pending",
            "error": "",
        })
    return manifest


def read_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(manifest_path, manifest):
    tmp_path = manifest_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    tmp_path.replace(manifest_path)


def download_one(video_path, local_path, output_dir):
    try:
        tmp_path = Path(hf_hub_download(REPO_ID, video_path, repo_type=REPO_TYPE, local_dir=str(output_dir)))
        dest = output_dir / local_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), str(dest))
        return "done", ""
    except Exception as e:
        return "failed", str(e)


def main():
    parser = argparse.ArgumentParser(description="Download the MAVOS-DD English subset for Module 3.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                         help="Where to store videos + manifest.csv (e.g. a Drive mount path).")
    parser.add_argument("--workers", type=int, default=4,
                         help="Number of parallel download threads.")
    parser.add_argument("--usage", type=str, nargs="*", default=None,
                         choices=["genuine_train", "genuine_val", "eval_track1", "eval_track2"],
                         help="Restrict download to these usage buckets (default: all).")
    parser.add_argument("--retry_failed", action="store_true",
                         help="Reset previously 'failed' rows back to 'pending' before running.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME

    if manifest_path.exists():
        print(f"Found existing manifest at {manifest_path}, resuming ...")
        manifest = read_manifest(manifest_path)
        if args.retry_failed:
            reset = 0
            for row in manifest:
                if row["status"] == "failed":
                    row["status"] = "pending"
                    row["error"] = ""
                    reset += 1
            print(f"  -> reset {reset} failed rows to pending")
    else:
        english_rows = load_english_metadata()
        manifest = build_manifest(english_rows)
        write_manifest(manifest_path, manifest)
        print(f"Wrote new manifest with {len(manifest)} entries -> {manifest_path}")

    if args.usage:
        todo = [r for r in manifest if r["status"] != "done" and r["usage"] in args.usage]
    else:
        todo = [r for r in manifest if r["status"] != "done"]

    counts = {}
    for r in manifest:
        counts[r["usage"]] = counts.get(r["usage"], 0) + 1
    print(f"Manifest totals by usage: {counts}")
    print(f"{len(todo)} of {len(manifest)} files left to download -> {output_dir}")

    if not todo:
        print("Nothing to do. All requested files are already downloaded.")
        return

    by_path = {r["video_path"]: r for r in manifest}
    completed_since_flush = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_one, r["video_path"], r["local_path"], output_dir): r["video_path"]
                for r in todo
            }
            with tqdm(total=len(futures), unit="file") as pbar:
                for future in as_completed(futures):
                    video_path = futures[future]
                    status, error = future.result()
                    row = by_path[video_path]
                    row["status"] = status
                    row["error"] = error
                    if status == "failed":
                        tqdm.write(f"FAILED: {video_path} -> {error}")
                    pbar.update(1)
                    completed_since_flush += 1
                    if completed_since_flush >= FLUSH_EVERY:
                        write_manifest(manifest_path, manifest)
                        completed_since_flush = 0
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved -- re-run this command to resume.")
    finally:
        write_manifest(manifest_path, manifest)
        # hf_hub_download's own "english/..." staging dirs are now empty; tidy up.
        shutil.rmtree(output_dir / "english", ignore_errors=True)

    n_done = sum(1 for r in manifest if r["status"] == "done")
    n_failed = sum(1 for r in manifest if r["status"] == "failed")
    print(f"\nDone: {n_done}/{len(manifest)} downloaded, {n_failed} failed.")
    if n_failed:
        print("Re-run with --retry_failed to retry the failed files.")


if __name__ == "__main__":
    main()

# # Batch 1: genuine_train + genuine_val (~7GB)
# python scripts/download_mavos_dd_en.py --usage genuine_train genuine_val
# rclone copy data/external/mavos_dd_en gdrive_personal:mavos_dd_en --progress
# rm -rf data/external/mavos_dd_en/genuine

# # Batch 2: eval_track1 (~9GB)
# python scripts/download_mavos_dd_en.py --usage eval_track1
# rclone copy data/external/mavos_dd_en gdrive_personal:mavos_dd_en --progress
# rm -rf data/external/mavos_dd_en/infer

# # Batch 3: eval_track2 (~20GB)
# python scripts/download_mavos_dd_en.py --usage eval_track2
# rclone copy data/external/mavos_dd_en gdrive_personal:mavos_dd_en --progress
# rm -rf data/external/mavos_dd_en/infer

