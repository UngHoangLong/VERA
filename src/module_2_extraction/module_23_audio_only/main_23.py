import argparse
import json
import os
import sys
from pathlib import Path
from audio_artifacts import AudioArtifactFeature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.paths import get_pipeline_paths, VALID_MODES

class AudioOrchestrator:
    def __init__(self, base_interim_dir, report_dir):
        self.base_interim_dir = Path(base_interim_dir)
        self.report_dir = Path(report_dir)

    def process_video_report(self, video_id: str):
        """Xử lý âm thanh từ thư mục interim và lưu đè vào file JSON ở thư mục final_reports"""
        # 1. Đường dẫn tới file báo cáo (Nằm ở final_reports)
        report_path = self.report_dir / f"{video_id}_report.json"
        
        # 2. Đường dẫn tới thư mục chứa Audio (Nằm ở interim)
        video_interim_dir = self.base_interim_dir / video_id

        if not report_path.exists():
            print(f"Không tìm thấy báo cáo tại: {report_path}")
            return

        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        chunks_data = report_data.get("chunks", {})
        if not chunks_data:
            return

        print(f"\n🎬 Đang giải phẫu âm thanh: {video_id}...")
        for chunk_id, chunk_info in chunks_data.items():
            # Lấy file âm thanh từ thư mục interim
            audio_file = video_interim_dir / chunk_id / "sync_audio.wav"
            chunk_info["audio_artifacts"] = {}

            if audio_file.exists():
                result = AudioArtifactFeature.process_audio_chunk(str(audio_file))
                if result.get("status") == "success":
                    chunk_info["audio_artifacts"] = {
                        "vocal_jitter_relative": result.get("vocal_jitter_relative", 0.0),
                        "vocal_shimmer_relative": result.get("vocal_shimmer_relative", 0.0)
                    }
                    print(f"  -> {chunk_id}: Thành công")
                else:
                    print(f"  -> {chunk_id}: Lỗi ({result.get('reason')})")
            else:
                print(f"  -> {chunk_id}: Bỏ qua (Không tìm thấy sync_audio.wav)")

        report_data["video_metadata"]["status"] = "audio_artifacts_completed"

        # Lưu đè lại vào đúng thư mục final_reports
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)

    def process_dataset(self):
        """Quét toàn bộ thư mục và chạy tuần tự"""
        if not self.base_interim_dir.exists():
            print(f" Không tìm thấy thư mục: {self.base_interim_dir}")
            return
            
        if not self.report_dir.exists():
            print(f" Không tìm thấy thư mục báo cáo: {self.report_dir}")
            return
            
        videos = sorted([d.name for d in self.base_interim_dir.iterdir() if d.is_dir()])
        for vid in videos:
            self.process_video_report(vid)

# ==========================================
# KHỞI CHẠY
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2.3: audio-only artifact extraction.")
    parser.add_argument(
        "--mode", type=str, required=True, choices=VALID_MODES,
        help="genuine: genuine-only videos for Module 3 training; "
             "infer: videos to be scored/evaluated.",
    )
    args = parser.parse_args()

    paths = get_pipeline_paths(args.mode)
    orchestrator = AudioOrchestrator(
        base_interim_dir=paths["interim_dir"],
        report_dir=paths["final_reports_dir"],
    )
    orchestrator.process_dataset()
    print("\n🏆 HOÀN THÀNH TOÀN BỘ PIPELINE MODULE 2.3!")