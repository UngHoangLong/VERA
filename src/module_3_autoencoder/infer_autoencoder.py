import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEATURE_INTERPRETATIONS, FEATURE_NAMES
from features import iter_report_files, load_json, parse_report, save_json
from model import build_autoencoder


def get_level(score: float) -> str:
    if score >= 1.0:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def build_interpretation(top_features: List[str]) -> str:
    if not top_features:
        return "Chunk này có lỗi tái tạo, nhưng chưa xác định được đặc trưng đóng góp chính."
    readable = [FEATURE_INTERPRETATIONS.get(name, name) for name in top_features]
    return "Chunk này có lỗi tái tạo cao hơn mẫu genuine baseline; các đặc trưng đóng góp lớn nhất gồm: " + ", ".join(readable) + "."


def load_model(model_dir: Path, device: torch.device):
    payload = torch.load(model_dir / "autoencoder.pt", map_location=device)
    config = payload["model_config"]
    model = build_autoencoder(**config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def infer_one_report(report_path: Path, model_dir: Path, output_dir: Path, device: torch.device, top_n: int) -> Path:
    report = load_json(report_path)
    video_id = report.get("video_metadata", {}).get("video_id", report_path.stem.replace("_report", ""))
    rows = parse_report(report_path)
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    imputer = preprocessor["imputer"]
    scaler = preprocessor["scaler"]
    threshold_payload = load_json(model_dir / "threshold.json")
    threshold = float(threshold_payload["threshold"])
    model, _ = load_model(model_dir, device)
    if not rows:
        output = {"video_metadata": {"video_id": video_id, "source_report": str(report_path), "module": "module_3_autoencoder_anomaly_scoring", "status": "no_valid_chunks"}, "chunks": {}}
        out_path = output_dir / f"{video_id}_evidence.json"
        save_json(output, out_path)
        return out_path
    raw_matrix = np.stack([row["vector"] for row in rows]).astype(np.float32)
    x = scaler.transform(imputer.transform(raw_matrix)).astype(np.float32)
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model(tensor).detach().cpu().numpy()
    per_feature_error = (x - recon) ** 2
    reconstruction_error = np.mean(per_feature_error, axis=1)
    chunks: Dict[str, Any] = {}
    for idx, row in enumerate(rows):
        err = float(reconstruction_error[idx])
        score = float(min(1.0, err / (threshold + 1e-12)))
        feature_errors = per_feature_error[idx]
        top_indices = np.argsort(feature_errors)[::-1][:top_n]
        top_features = [FEATURE_NAMES[i] for i in top_indices]
        feature_vector_clean = {}
        for name, value in row["feature_dict"].items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                feature_vector_clean[name] = None
            else:
                feature_vector_clean[name] = float(value)
        chunks[row["chunk_id"]] = {
            "time_metadata": row["time_metadata"],
            "frames_analyzed": row["frames_analyzed"],
            "feature_vector": feature_vector_clean,
            "anomaly": {
                "reconstruction_error": err,
                "normalized_anomaly_score": score,
                "threshold": threshold,
                "level": get_level(score),
            },
            "top_reconstruction_error_features": top_features,
            "per_feature_reconstruction_error": {FEATURE_NAMES[i]: float(feature_errors[i]) for i in range(len(FEATURE_NAMES))},
            "raw_text_evidence": row["raw_text_evidence"],
            "interpretation": build_interpretation(top_features),
        }
    output = {
        "video_metadata": {"video_id": video_id, "source_report": str(report_path), "module": "module_3_autoencoder_anomaly_scoring", "status": "analyzed"},
        "model_metadata": {"model_type": "one_class_mlp_autoencoder", "model_dir": str(model_dir), "feature_names": FEATURE_NAMES, "threshold_percentile": threshold_payload.get("threshold_percentile"), "threshold": threshold},
        "chunks": chunks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_evidence.json"
    save_json(output, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 3 AutoEncoder inference on Module 2 reports.")
    parser.add_argument("--input", type=str, default=None, help="Path to one <video_id>_report.json file.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing *_report.json files.")
    parser.add_argument("--model_dir", type=str, default="./module3_models")
    parser.add_argument("--output_dir", type=str, default="./evidence_reports")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if args.input is None and args.input_dir is None:
        raise ValueError("Provide either --input or --input_dir.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    report_paths: List[Path] = []
    if args.input is not None:
        report_paths.append(Path(args.input))
    if args.input_dir is not None:
        report_paths.extend(iter_report_files(Path(args.input_dir)))
    if not report_paths:
        raise FileNotFoundError("No report files found.")
    written = []
    for report_path in report_paths:
        written.append(str(infer_one_report(report_path, model_dir, output_dir, device, args.top_n)))
    print(json.dumps({"status": "ok", "num_reports": len(written), "outputs": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
