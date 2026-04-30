import json
import os
import subprocess
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN TỔNG QUÁT ---
# Lùi 4 cấp để ra tới thư mục gốc của project (từ src/module_2_extraction/module_22_audio_visual_consistency/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_REPORTS_DIR = PROJECT_ROOT / "final_reports"

def run_command(command_list, step_name):
    """Hàm chạy các script độc lập bằng Subprocess để giải phóng VRAM sau mỗi bước."""
    print(f"\n Đang chạy: {step_name}...")
    try:
        subprocess.run(command_list, check=True)
        print(f"Hoàn thành: {step_name}")
    except subprocess.CalledProcessError as e:
        print(f"LỖI tại bước {step_name}. Mã lỗi: {e.returncode}")
        return False
    return True

def safe_read_json(filepath):
    """Hàm đọc file JSON an toàn, trả về dict rỗng nếu file không tồn tại."""
    path = Path(filepath)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def update_final_report(video_id, chunk_id):
    """Tổng hợp kết quả từ data/processed vào báo cáo đã có ở final_reports."""
    
    report_path = FINAL_REPORTS_DIR / f"{video_id}_report.json"
    if not report_path.exists():
        print(f"Bỏ qua {chunk_id}: Không tìm thấy khung báo cáo từ Module 2.1.")
        return

    # 1. Thu thập dữ liệu từ các chuyên gia trong data/processed
    # Dữ liệu VSR/ASR thô (để lấy text thô nếu cần)
    vsr_data = safe_read_json(PROCESSED_DIR / "vsr_output" / video_id / f"{chunk_id}.json")
    # Dữ liệu phân tích chuyên sâu
    ccfd_data = safe_read_json(PROCESSED_DIR / "ccfd_output" / video_id / f"{chunk_id}.json")
    scfd_data = safe_read_json(PROCESSED_DIR / "scfd_output" / video_id / f"{chunk_id}.json")
    tcfd_data = safe_read_json(PROCESSED_DIR / "tcfd_output" / video_id / f"{chunk_id}.json")

    # Nếu không có dữ liệu CCFD (mô hình so sánh text), nghĩa là chunk này bị lỗi ở bước VSR/ASR
    if not ccfd_data:
        print(f"Bỏ qua {chunk_id}: Dữ liệu CCFD không tồn tại.")
        return

    # 2. Rút trích các đặc trưng tinh hoa
    # Lấy text đã qua xử lý chuẩn hóa (normalize) từ CCFD
    asr_text_clean = ccfd_data.get("reference_text_norm", "")
    vsr_text_clean = ccfd_data.get("hypothesis_text_norm", "")

    audio_visual_payload = {
        "transcripts": {
            "asr_text_audio": asr_text_clean,
            "vsr_text_lips": vsr_text_clean,
            "wer_score": ccfd_data.get("wer", None)
        },
        "semantic_consistency": {
            "mean_cosine_similarity": scfd_data.get("mean", None),
            "min_cosine_similarity": scfd_data.get("min", None),
            "percentile_3rd_cosine": scfd_data.get("third_percentile", None)
        },
        "temporal_sync": {
            "sync_score": tcfd_data.get("mean_score", None),
            "min_sync_score": tcfd_data.get("min_score", None),
            "variance": tcfd_data.get("variance", None)
        }
    }

    # 3. Ghi đè cập nhật vào Final Report
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if chunk_id in report.get("chunks", {}):
        # Chỉ cập nhật phần audio_visual_consistency, KHÔNG ghi đè time_metadata của 2.1
        report["chunks"][chunk_id]["audio_visual_consistency"] = audio_visual_payload
        
        # Cập nhật trạng thái video
        report["video_metadata"]["status"] = "fully_analyzed"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"Đã cập nhật kết quả Đồng bộ (Module 2.2) cho {chunk_id}")

def process_single_video(video_id):
    print(f"\n" + "="*50)
    print(f"BẮT ĐẦU CHẠY MODULE 2.2: {video_id}")
    print("="*50)

    video_interim_dir = INTERIM_DIR / video_id
    if not video_interim_dir.exists(): return

    # Quét các chunk thực tế đang có trong data/interim
    chunk_dirs = sorted([d.name for d in video_interim_dir.glob("chunk_*") if d.is_dir()])

    for chunk_id in chunk_dirs:
        print(f"\n--- Đang xử lý: {chunk_id} ---")
        
        # --- BƯỚC 1: CHẠY CÁC MÔ HÌNH (Sử dụng Subprocess để giải phóng VRAM GPU sau mỗi lượt) ---
        
        # 1. Tạo dữ liệu đầu vào cho VSR/ASR (Face crop & Audio extraction)
        if not run_command(["python", "src/module_2_extraction/module_22_audio_visual_consistency/build_vsr_asr_input_from_slides.py", 
                            "--video-id", video_id, "--chunk-id", chunk_id], "Build Input"): continue

        # 2. Chạy ASR (Whisper)
        run_command(["python", "src/module_2_extraction/module_22_audio_visual_consistency/run_asr_inference_per_chunk.py", 
                     "--video-id", video_id, "--chunk-id", chunk_id], "ASR (Whisper)")

        # 3. Chạy VSR (Auto-AVSR)
        run_command(["python", "src/module_2_extraction/module_22_audio_visual_consistency/run_vsr_inference_per_chunk.py", 
                     "--video-id", video_id, "--chunk-id", chunk_id], "VSR (Auto-AVSR)")

        # 4. Chạy CCFD (So sánh văn bản) - Truyền tham số root để script tự tìm video/chunk
        # Lưu ý: Sửa lại tham số nếu CCFD của cậu yêu cầu --asr-root thay vì --video-id
        run_command(["python", "src/module_2_extraction/module_22_audio_visual_consistency/CCFD_per_chunk.py", 
                     "--asr-root", str(PROCESSED_DIR/"asr_output"), "--vsr-root", str(PROCESSED_DIR/"vsr_output"), 
                     "--output-root", str(PROCESSED_DIR/"ccfd_output")], "CCFD (Text)")

        # 5. Chạy SCFD (Đồng bộ ngữ nghĩa)
        run_command(["python", "src/module_2_extraction/module_22_audio_visual_consistency/SCFD_per_chunk.py", 
                     "--video-id", video_id, "--chunk-id", chunk_id], "SCFD (Semantic)")

        # 6. Chạy TCFD (Đồng bộ nhịp điệu)
        run_command(["python", "src/module_2_extraction/module_22_audio_visual_consistency/TCFD_per_chunk.py", 
                     "--video-id", video_id, "--chunk-id", chunk_id], "TCFD (Temporal)")

        # --- BƯỚC 2: TỔNG HỢP VÀO REPORT ---
        update_final_report(video_id, chunk_id)

if __name__ == "__main__":
    # Tự động quét các thư mục video đã được Module 1 cắt ra
    videos = sorted([d.name for d in INTERIM_DIR.iterdir() if d.is_dir()])
    for vid in videos:
        process_single_video(vid)
    print("\n HOÀN THÀNH TOÀN BỘ PIPELINE MODULE 2.2!")