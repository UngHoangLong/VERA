import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TỔNG QUÁT
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_REPORTS_DIR = PROJECT_ROOT / "final_reports"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN SCRIPT
# ==========================================
MODULE_DIR = "src/module_2_extraction/module_22_audio_visual_consistency"

SCRIPT_BUILD_INPUT = f"{MODULE_DIR}/build_vsr_asr_input_from_slides.py"
SCRIPT_VSR = f"{MODULE_DIR}/run_vsr_inference_per_chunk.py"
SCRIPT_ASR = f"{MODULE_DIR}/run_asr_inference_per_chunk.py"
SCRIPT_CCFD = f"{MODULE_DIR}/CCFD_per_chunk.py"
SCRIPT_SCFD = f"{MODULE_DIR}/SCFD_per_chunk.py"
SCRIPT_TCFD = f"{MODULE_DIR}/TCFD_per_chunk.py"


# ==========================================
# CẤU HÌNH MODEL
# ==========================================
VSR_MODEL_PATH = PROJECT_ROOT / "pretrained_model" / "vsr_trlrs2lrs3vox2avsp_base.pth"
ASR_MODEL_PATH = PROJECT_ROOT / "pretrained_model" / "whisper-medium-en" / "model.safetensors"

SCFD_AVHUBERT_ROOT = Path(os.environ.get(
    "SCFD_AVHUBERT_ROOT",
    str(PROJECT_ROOT.parent / "av_hubert")
))

SCFD_MODEL_PATH = Path(os.environ.get(
    "SCFD_MODEL_PATH",
    str(PROJECT_ROOT / "pretrained_model" / "base_vox_iter5.pt")
))

TCFD_CHECKPOINT_PATH = PROJECT_ROOT / "pretrained_model" / "pure_MTDVocaLiST.pth"

TCFD_MTDVOCALIST_ROOT = Path(os.environ.get(
    "TCFD_MTDVOCALIST_ROOT",
    str(PROJECT_ROOT.parent / "MTDVocaLiST")
))


# ==========================================
# CẤU HÌNH OUTPUT
# ==========================================
VSR_OUTPUT_ROOT = PROCESSED_DIR / "vsr_output"
ASR_OUTPUT_ROOT = PROCESSED_DIR / "asr_output"
CCFD_OUTPUT_ROOT = PROCESSED_DIR / "ccfd_output"
SCFD_OUTPUT_ROOT = PROCESSED_DIR / "scfd_output"
TCFD_OUTPUT_JSON = PROCESSED_DIR / "tcfd_output.json"


# ==========================================
# CẤU HÌNH THAM SỐ CHUNG
# ==========================================
INPUT_VIDEO_NAME = "vsr_input.mp4"
INPUT_AUDIO_NAME = "sync_audio.wav"
LANGUAGE = "english"
VIDEO_LAYOUT = "mouth96"
OVERWRITE = True


def make_env():
    return os.environ.copy()


def run_command(command_list, step_name):
    print("\n" + "=" * 80)
    print(f"Đang chạy: {step_name}")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Không set"))
    print("Lệnh:", " ".join(map(str, command_list)))
    print("=" * 80)

    try:
        subprocess.run(
            command_list,
            check=True,
            cwd=str(PROJECT_ROOT),
            env=make_env(),
        )
        print(f"Hoàn thành: {step_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"LỖI tại bước: {step_name}")
        print(f"Mã lỗi: {e.returncode}")
        return False


def add_overwrite(command_list):
    if OVERWRITE:
        command_list.append("--overwrite")
    return command_list


def ensure_required_paths():
    required_paths = {
        "VSR_MODEL_PATH": VSR_MODEL_PATH,
        "ASR_MODEL_PATH": ASR_MODEL_PATH,
        "SCFD_AVHUBERT_ROOT": SCFD_AVHUBERT_ROOT,
        "SCFD_MODEL_PATH": SCFD_MODEL_PATH,
        "TCFD_CHECKPOINT_PATH": TCFD_CHECKPOINT_PATH,
        "TCFD_MTDVOCALIST_ROOT": TCFD_MTDVOCALIST_ROOT,
    }

    missing = []

    for name, path in required_paths.items():
        if not Path(path).exists():
            missing.append((name, path))

    if missing:
        print("\nKhông tìm thấy một số đường dẫn bắt buộc:")
        for name, path in missing:
            print(f"- {name}: {path}")
        return False

    return True


def run_build_input(python_bin):
    command = [
        python_bin,
        SCRIPT_BUILD_INPUT,
        "--input-root", str(INTERIM_DIR),
    ]
    command = add_overwrite(command)

    return run_command(command, "Build VSR/ASR input")


def run_vsr(python_bin):
    command = [
        python_bin,
        SCRIPT_VSR,
        "--input-root", str(INTERIM_DIR),
        "--model-path", str(VSR_MODEL_PATH),
        "--output-root", str(VSR_OUTPUT_ROOT),
    ]
    command = add_overwrite(command)

    return run_command(command, "VSR inference")


def run_asr(python_bin):
    command = [
        python_bin,
        SCRIPT_ASR,
        "--input-root", str(INTERIM_DIR),
        "--model-path", str(ASR_MODEL_PATH),
        "--output-root", str(ASR_OUTPUT_ROOT),
        "--language", LANGUAGE,
    ]
    command = add_overwrite(command)

    return run_command(command, "ASR inference")


def run_ccfd(python_bin):
    command = [
        python_bin,
        SCRIPT_CCFD,
        "--asr-root", str(ASR_OUTPUT_ROOT),
        "--vsr-root", str(VSR_OUTPUT_ROOT),
        "--output-root", str(CCFD_OUTPUT_ROOT),
    ]
    command = add_overwrite(command)

    return run_command(command, "CCFD")


def run_scfd(python_bin):
    command = [
        python_bin,
        SCRIPT_SCFD,
        "--input-root", str(INTERIM_DIR),
        "--input-video-name", INPUT_VIDEO_NAME,
        "--input-audio-name", INPUT_AUDIO_NAME,
        "--avhubert-root", str(SCFD_AVHUBERT_ROOT),
        "--model-path", str(SCFD_MODEL_PATH),
        "--output-root", str(SCFD_OUTPUT_ROOT),
    ]
    command = add_overwrite(command)

    return run_command(command, "SCFD")


def run_tcfd(python_bin):
    command = [
        python_bin,
        SCRIPT_TCFD,
        "--input-root", str(INTERIM_DIR),
        "--checkpoint-path", str(TCFD_CHECKPOINT_PATH),
        "--output-json", str(TCFD_OUTPUT_JSON),
        "--input-video-name", INPUT_VIDEO_NAME,
        "--audio-name", INPUT_AUDIO_NAME,
        "--video-layout", VIDEO_LAYOUT,
        "--mtdvocalist-root", str(TCFD_MTDVOCALIST_ROOT),
        "--device", "cuda",
    ]

    return run_command(command, "TCFD")


def safe_read_json(filepath):
    path = Path(filepath)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def read_scfd_data(video_id: str, chunk_id: str) -> dict:
    """SCFD_per_chunk.py lưu: scfd_output/<video_id>/<chunk_id>/vsr_input_semantic.json"""
    path = SCFD_OUTPUT_ROOT / video_id / chunk_id / "vsr_input_semantic.json"
    data = safe_read_json(path)
    if not data or not data.get("ok"):
        return {}

    summary = data.get("summary", {}) or {}
    return {
        "mean_cosine_similarity": summary.get("mean_cosine"),
        "min_cosine_similarity": summary.get("min_cosine"),
        "percentile_3rd_cosine": summary.get("third_percentile"),
    }


@lru_cache(maxsize=1)
def build_tcfd_index() -> dict:
    """TCFD_per_chunk.py lưu một file tổng tại TCFD_OUTPUT_JSON."""
    data = safe_read_json(TCFD_OUTPUT_JSON)
    index = {}

    for video in data.get("videos", []) or []:
        video_id = video.get("video_id")
        for chunk in video.get("chunks", []) or []:
            chunk_id = chunk.get("chunk_id")
            if video_id and chunk_id:
                index[(video_id, chunk_id)] = chunk

    return index


def read_tcfd_data(video_id: str, chunk_id: str) -> dict:
    # TCFD hiện lưu theo dạng:
    # data/processed/tcfd_output/<video_id>/<chunk_id>.json
    # Ví dụ:
    # data/processed/tcfd_output/Donald_Trump/chunk_0000.json

    tcfd_path = PROCESSED_DIR / "tcfd_output" / video_id / f"{chunk_id}.json"
    chunk = safe_read_json(tcfd_path)

    if not chunk or not (chunk.get("status") == "ok" or chunk.get("ok") is True):
        return {}

    std = chunk.get("window_score_std")
    variance = None
    if std is not None:
        try:
            variance = float(std) ** 2
        except Exception:
            variance = None

    return {
        "sync_score": chunk.get("tcfd_score"),
        "min_sync_score": chunk.get("window_score_min"),
        "variance": variance,
    }


def update_final_report(video_id: str, chunk_id: str):
    report_path = FINAL_REPORTS_DIR / f"{video_id}_report.json"
    if not report_path.exists():
        print(f"Bỏ qua {video_id}/{chunk_id}: Không tìm thấy khung báo cáo từ Module 2.1.")
        return

    ccfd_data = safe_read_json(CCFD_OUTPUT_ROOT / video_id / f"{chunk_id}.json")
    if not ccfd_data:
        return

    scfd_payload = read_scfd_data(video_id, chunk_id)
    tcfd_payload = read_tcfd_data(video_id, chunk_id)

    audio_visual_payload = {
        "transcripts": {
            "asr_text_audio": ccfd_data.get("reference_text_norm", ""),
            "vsr_text_lips": ccfd_data.get("hypothesis_text_norm", ""),
            "wer_score": ccfd_data.get("wer", None),
        },
        "semantic_consistency": {
            "mean_cosine_similarity": scfd_payload.get("mean_cosine_similarity"),
            "min_cosine_similarity": scfd_payload.get("min_cosine_similarity"),
            "percentile_3rd_cosine": scfd_payload.get("percentile_3rd_cosine"),
        },
        "temporal_sync": {
            "sync_score": tcfd_payload.get("sync_score"),
            "min_sync_score": tcfd_payload.get("min_sync_score"),
            "variance": tcfd_payload.get("variance"),
        },
    }

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if chunk_id in report.get("chunks", {}):
        report["chunks"][chunk_id]["audio_visual_consistency"] = audio_visual_payload
        report["video_metadata"]["status"] = "fully_analyzed"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"Đã cập nhật Report: {video_id} - {chunk_id}")


def synthesize_reports():
    print("\n" + "=" * 80)
    print("BẮT ĐẦU TỔNG HỢP FINAL REPORT")
    print("=" * 80)

    build_tcfd_index.cache_clear()

    if not INTERIM_DIR.exists():
        print("Không tìm thấy thư mục interim!")
        return

    videos = sorted([d.name for d in INTERIM_DIR.iterdir() if d.is_dir()])
    for video_id in videos:
        video_interim_dir = INTERIM_DIR / video_id
        chunk_dirs = sorted([d.name for d in video_interim_dir.glob("chunk_*") if d.is_dir()])

        for chunk_id in chunk_dirs:
            update_final_report(video_id, chunk_id)


def run_pipeline():
    print("\n" + "=" * 80)
    print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE MODULE 2.2")
    print("=" * 80)

    python_bin = sys.executable

    if not ensure_required_paths():
        return False

    steps = [
        run_build_input,
        run_vsr,
        run_asr,
        run_ccfd,
        run_scfd,
        run_tcfd,
    ]

    for step in steps:
        if not step(python_bin):
            return False

    return True


if __name__ == "__main__":
    ok = run_pipeline()

    if ok:
        synthesize_reports()
        print("\nHOÀN THÀNH TOÀN BỘ PIPELINE MODULE 2.2!")
    else:
        print("\nPIPELINE DỪNG DO CÓ LỖI Ở MỘT BƯỚC TRƯỚC ĐÓ.")
        sys.exit(1)
