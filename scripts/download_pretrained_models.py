#!/usr/bin/env python3
"""
Download all pretrained model weights needed by the VERA pipeline.

Prerequisites:
    pip install gdown

Google Drive note:
    Files are stored in your private Drive (MVDD_backup/pretrained_model/).
    gdown can download them if you do ONE of the following on the server:
      Option A (easiest for VastAI): In Drive, right-click each file/folder
                → Share → "Anyone with the link" → Copy link.
                Then replace the file_id values below with the new public IDs.
      Option B: Run `gdown --login` once on the server to authenticate with your Google account.

Module 5 models (Qwen3-VL-8B, InternVL2_5-8B):
    Downloaded automatically by HuggingFace `from_pretrained()` when Module 5 first runs.
    No manual download needed — just make sure HF_HOME has enough disk space (~16 GB each).

Module 3 model (mvae_poe.pt):
    Trained locally — not downloaded here. Run `python src/module_3_autoencoder/train.py`.
"""

import sys
import urllib.request
from pathlib import Path

try:
    import gdown
except ImportError:
    print("ERROR: gdown not installed. Run:  pip install gdown")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_model"
WHISPER_DIR = PRETRAINED_DIR / "whisper-medium-en"
SFACE_DIR = PROJECT_ROOT / "models" / "sface"


# ─── Module 1: SFace face re-identification model ────────────────────────────
# Downloaded directly from OpenCV Zoo (public CDN, no auth needed).

SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_DEST = SFACE_DIR / "face_recognition_sface_2021dec.onnx"


# ─── Module 2.2: pretrained model weights (Google Drive) ─────────────────────
# File IDs from your Drive: My Drive > MVDD_backup > pretrained_model/

DRIVE_FILES = [
    # (destination_path, drive_file_id, display_name)
    (
        PRETRAINED_DIR / "vsr_trlrs2lrs3vox2avsp_base.pth",
        "1z0OQBs_ERsArA9YCSUPwWHNsxFcmEHTV",
        "VSR model (955 MB)",
    ),
    (
        PRETRAINED_DIR / "pure_MTDVocaLiST.pth",
        "19Sbq_DTJ5Q2qQhMGNafL5KnOR52zB-R3",
        "TCFD VocaLiST checkpoint (51 MB)",
    ),
    (
        PRETRAINED_DIR / "base_vox_iter5.pt",
        "1061J_qHrdhT0T1JWcgGKP8IkIAz9wEgm",
        "SCFD AV-HuBERT base (1.16 GB)",
    ),
]

# whisper-medium-en: only the PyTorch-compatible files needed
# (tf_model.h5, flax_model.msgpack, pytorch_model.bin skipped — safetensors used)
WHISPER_FILES = [
    (WHISPER_DIR / "model.safetensors",      "101DIbaHMdwl9JHUuHIykUJW5xPHo1xh4", "Whisper weights (2.8 GB)"),
    (WHISPER_DIR / "config.json",             "1zRHUq4gslAS-llkrCTaLQlUOaI5ZlhYH", "config.json"),
    (WHISPER_DIR / "generation_config.json",  "1MRLigxpmijY1QXi5FM9o9VppDgovGOr8", "generation_config.json"),
    (WHISPER_DIR / "vocab.json",              "1j-g3O4ywmtr4wciTk8UgyZsQi4xAAGI_", "vocab.json"),
    (WHISPER_DIR / "tokenizer.json",          "1qKYW3K6HE0pd591GzBxN752jsNsuqbjf", "tokenizer.json"),
    (WHISPER_DIR / "tokenizer_config.json",   "1l5hDbhiyOLCeocr5l9U_gSCZxMttm7oE", "tokenizer_config.json"),
    (WHISPER_DIR / "special_tokens_map.json", "1rNGFXBXd5-xS5LdbuilUdlpJ_e-9Gvze", "special_tokens_map.json"),
    (WHISPER_DIR / "normalizer.json",         "12Q7J5FiMmWkes2hXtvxi4WzLryl8LDwn", "normalizer.json"),
    (WHISPER_DIR / "preprocessor_config.json","1EtFz-15i9B13O0njecqUQZu_DW04p5H_", "preprocessor_config.json"),
    (WHISPER_DIR / "merges.txt",              "19oou-1RGo6IxaFHftwp8eTmCjiKDzyP3", "merges.txt"),
    (WHISPER_DIR / "added_tokens.json",       "1yAXw4wB2FA4lZXYMYq2U2eGYGB8jQKty", "added_tokens.json"),
]


def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        print(f"\r  {pct}%  ({count * block_size / 1e6:.1f} MB)", end="", flush=True)


def download_sface():
    if SFACE_DEST.exists():
        print(f"[skip] SFace model already exists: {SFACE_DEST}")
        return
    SFACE_DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[download] SFace ONNX → {SFACE_DEST}")
    urllib.request.urlretrieve(SFACE_URL, SFACE_DEST, reporthook=_progress)
    print(f"\n  Done. {SFACE_DEST.stat().st_size / 1e6:.1f} MB")


def download_drive_file(dest: Path, file_id: str, label: str):
    if dest.exists():
        print(f"[skip] {label} already exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[download] {label} → {dest}")
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, str(dest), quiet=False)
    if result is None or not dest.exists():
        print(f"  ERROR: download failed for {label}.")
        print(f"  Make sure the file is shared publicly or run `gdown --login` first.")


def main():
    print("=" * 60)
    print("VERA pretrained model downloader")
    print("=" * 60)

    # 1. SFace (Module 1)
    print("\n── Module 1: SFace ──")
    download_sface()

    # 2. Module 2.2 model weights
    print("\n── Module 2.2: VSR / TCFD / SCFD ──")
    for dest, file_id, label in DRIVE_FILES:
        download_drive_file(dest, file_id, label)

    # 3. Whisper ASR (Module 2.2)
    print("\n── Module 2.2: Whisper medium-en ──")
    for dest, file_id, label in WHISPER_FILES:
        download_drive_file(dest, file_id, label)

    print("\n" + "=" * 60)
    print("Done. Module 3 (mvae_poe.pt) must be trained locally.")
    print("Module 5 (Qwen/InternVL) auto-downloads on first run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
