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
# CẤU HÌNH ĐƯỜNG DẪN SCRIPT & MODEL
# ==========================================
MODULE_DIR = "src/module_2_extraction/module_22_audio_visual_consistency"
SCRIPT_BUILD_INPUT = f"{MODULE_DIR}/build_vsr_asr_input_from_slides.py"
SCRIPT_ASR = f"{MODULE_DIR}/run_asr_inference_per_chunk.py"
SCRIPT_VSR = f"{MODULE_DIR}/run_vsr_inference_per_chunk.py"
SCRIPT_CCFD = f"{MODULE_DIR}/CCFD_per_chunk.py"
SCRIPT_SCFD = f"{MODULE_DIR}/SCFD_per_chunk.py"
SCRIPT_TCFD = f"{MODULE_DIR}/TCFD_per_chunk.py"

# Sửa đúng theo máy của bạn nếu các đường dẫn này khác.
TCFD_CHECKPOINT = Path("./pretrained_model/pure_MTDVocaLiST.pth")
TCFD_MTDVOCALIST_ROOT = Path(os.environ.get("TCFD_MTDVOCALIST_ROOT", str(PROJECT_ROOT.parent / "MTDVocaLiST")))

# SCFD_per_chunk.py yêu cầu bắt buộc 2 tham số này. Có thể set bằng biến môi trường
# để vẫn giữ được lệnh chạy main_22.py ngắn gọn.
SCFD_AVHUBERT_ROOT = Path(os.environ.get(
    "SCFD_AVHUBERT_ROOT",
    "/mmlab_students/storageStudents/nguyenvd/truongdtd/av_hubert"
))

SCFD_MODEL_PATH = Path(os.environ.get(
    "SCFD_MODEL_PATH",
    "/mmlab_students/storageStudents/nguyenvd/truongdtd/av_hubert/base_lrs3_iter4.pt"
))

# ==========================================
# CẤU HÌNH ASR/WHISPER
# ==========================================
ASR_BATCH_SIZE = "1"
ASR_NUM_BEAMS = "1"
ASR_RETRY_NUM_BEAMS = "1"
ASR_CHUNK_LENGTH_S = "30"
ASR_DEVICE_MAP = "balanced_low_0"
ASR_USE_MODEL_PARALLEL = True
ASR_OVERWRITE = False
ASR_EMPTY_CACHE_EACH_CHUNK = True

# ==========================================
# CẤU HÌNH OUTPUT SCFD/TCFD
# ==========================================
SCFD_OUTPUT_ROOT = PROCESSED_DIR / "scfd_output"
TCFD_OUTPUT_JSON = PROCESSED_DIR / "tcfd_output" / "tcfd_interim.json"

# TCFD/MTDVocaLiST dễ crash khi tự bật torch.nn.DataParallel.
# Vì vậy chỉ cho riêng bước TCFD nhìn thấy 1 GPU.
TCFD_SINGLE_GPU = os.environ.get("TCFD_SINGLE_GPU", "").strip()
TCFD_BATCH_SIZE = os.environ.get("TCFD_BATCH_SIZE", "8").strip()


def run_command(command_list, step_name, env=None):
    print(f"\nĐang chạy: {step_name}...")
    try:
        subprocess.run(command_list, check=True, cwd=str(PROJECT_ROOT), env=env)
        print(f"Hoàn thành: {step_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"LỖI tại bước {step_name}. Mã lỗi: {e.returncode}")
        return False


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
    """TCFD_per_chunk.py lưu một file tổng: tcfd_output/tcfd_interim.json."""
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
    chunk = build_tcfd_index().get((video_id, chunk_id), {})
    if not chunk or chunk.get("status") != "ok":
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


def update_final_report(video_id, chunk_id):
    report_path = FINAL_REPORTS_DIR / f"{video_id}_report.json"
    if not report_path.exists():
        print(f"Bỏ qua {chunk_id}: Không tìm thấy khung báo cáo từ Module 2.1.")
        return

    ccfd_data = safe_read_json(PROCESSED_DIR / "ccfd_output" / video_id / f"{chunk_id}.json")
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


def ensure_scfd_paths():
    if not SCFD_AVHUBERT_ROOT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy SCFD_AVHUBERT_ROOT: {SCFD_AVHUBERT_ROOT}\n"
            "Hãy sửa biến SCFD_AVHUBERT_ROOT trong main_22.py hoặc export SCFD_AVHUBERT_ROOT=/path/to/AV-HuBERT"
        )
    if not SCFD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy SCFD_MODEL_PATH: {SCFD_MODEL_PATH}\n"
            "Hãy sửa biến SCFD_MODEL_PATH trong main_22.py hoặc export SCFD_MODEL_PATH=/path/to/checkpoint.pt"
        )


def run_pipeline():
    print("\n" + "=" * 60)
    print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE MODULE 2.2 (BATCH MODE)")
    print("=" * 60)

    python_bin = sys.executable

    if not run_command([python_bin, SCRIPT_BUILD_INPUT, "--input-root", str(INTERIM_DIR)], "Build Input"):
        return False

    asr_cmd = [
        python_bin, SCRIPT_ASR,
        "--input-root", str(INTERIM_DIR),
        "--output-root", str(PROCESSED_DIR / "asr_output"),
        "--batch-size", ASR_BATCH_SIZE,
        "--num-beams", ASR_NUM_BEAMS,
        "--retry-num-beams", ASR_RETRY_NUM_BEAMS,
        "--chunk-length-s", ASR_CHUNK_LENGTH_S,
    ]
    if ASR_USE_MODEL_PARALLEL:
        asr_cmd += ["--model-parallel", "--device-map", ASR_DEVICE_MAP]
    if ASR_OVERWRITE:
        asr_cmd += ["--overwrite"]
    if ASR_EMPTY_CACHE_EACH_CHUNK:
        asr_cmd += ["--empty-cache-each-chunk"]
    if not run_command(asr_cmd, "ASR (Whisper - Model Parallel)"):
        return False

    if not run_command([python_bin, SCRIPT_VSR, "--input-root", str(INTERIM_DIR), "--output-root", str(PROCESSED_DIR / "vsr_output")], "VSR (Auto-AVSR)"):
        return False

    if not run_command([
        python_bin, SCRIPT_CCFD,
        "--asr-root", str(PROCESSED_DIR / "asr_output"),
        "--vsr-root", str(PROCESSED_DIR / "vsr_output"),
        "--output-root", str(PROCESSED_DIR / "ccfd_output"),
    ], "CCFD (Text)"):
        return False

    ensure_scfd_paths()
    if not run_command([
        python_bin, SCRIPT_SCFD,
        "--input-root", str(INTERIM_DIR),
        "--output-root", str(SCFD_OUTPUT_ROOT),
        "--avhubert-root", str(SCFD_AVHUBERT_ROOT),
        "--model-path", str(SCFD_MODEL_PATH),
        "--input-video-name", "vsr_input.mp4",
        "--input-audio-name", "sync_audio.wav",
        "--device", "cuda:0",
    ], "SCFD (Semantic)"):
        return False

    # TCFD được chạy trong subprocess riêng và chỉ expose 1 GPU để tránh
    # torch.nn.DataParallel trong TCFD_per_chunk.py gây segmentation fault (-11).
    tcfd_env = os.environ.copy()
    visible = tcfd_env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if TCFD_SINGLE_GPU:
        tcfd_env["CUDA_VISIBLE_DEVICES"] = TCFD_SINGLE_GPU
    elif visible:
        tcfd_env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip()
    else:
        tcfd_env["CUDA_VISIBLE_DEVICES"] = "0"

    if not run_command([
        python_bin, SCRIPT_TCFD,
        "--input-root", str(INTERIM_DIR),
        "--checkpoint-path", str(TCFD_CHECKPOINT),
        "--output-json", str(TCFD_OUTPUT_JSON),
        "--mtdvocalist-root", str(TCFD_MTDVOCALIST_ROOT),
        "--input-video-name", "vsr_input.mp4",
        "--audio-name", "sync_audio.wav",
        "--device", "cuda",
        "--batch-size", TCFD_BATCH_SIZE,
    ], "TCFD (Temporal - Single GPU)", env=tcfd_env):
        return False

    return True


def synthesize_reports():
    print("\n" + "=" * 60)
    print("BẮT ĐẦU TỔNG HỢP FINAL REPORT")
    print("=" * 60)
    build_tcfd_index.cache_clear()

    if not INTERIM_DIR.exists():
        print("Không tìm thấy thư mục interim!")
        return

    videos = sorted([d.name for d in INTERIM_DIR.iterdir() if d.is_dir()])
    for vid in videos:
        video_interim_dir = INTERIM_DIR / vid
        chunk_dirs = sorted([d.name for d in video_interim_dir.glob("chunk_*") if d.is_dir()])
        for chunk_id in chunk_dirs:
            update_final_report(vid, chunk_id)


if __name__ == "__main__":
    ok = run_pipeline()
    if ok:
        synthesize_reports()
        print("\nHOÀN THÀNH TOÀN BỘ PIPELINE MODULE 2.2!")
    else:
        print("\nPIPELINE DỪNG DO CÓ LỖI Ở MỘT BƯỚC TRƯỚC ĐÓ.")
