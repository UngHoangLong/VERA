"""
infer.py — Inference pipeline for MVAE-PoE (Module 3).

Reads a Module 2 *_report.json, runs the trained MVAE-PoE, and writes
an *_evidence.json designed to be consumed by the Module 5 MLLM agent.

Key design choices for MLLM compatibility:
  - audio_reconstruction_score is null (not zero) when no audio observed.
  - missing_features lists every null feature explicitly.
  - modalities_analyzed / modalities_missing give clear context.
  - No imputed values ever appear in the output JSON.

Usage:
    cd src/module_3_autoencoder
    python infer.py --input_dir ../../final_reports --model_dir ./module3_models
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch

from config import AUDIO_FEATURE_NAMES, FEATURE_INTERPRETATIONS, VISUAL_FEATURE_NAMES
from dataset import (
    iter_report_files,
    load_json,
    parse_report,
    rows_to_modality_matrices,
    save_json,
)
from loss import compute_chunk_scores
from model import build_mvae_poe


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: Path, device: torch.device):
    payload = torch.load(model_dir / "mvae_poe.pt", map_location=device)
    model = build_mvae_poe(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


# ---------------------------------------------------------------------------
# Evidence JSON helpers
# ---------------------------------------------------------------------------

def _clean(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _level(score: float) -> str:
    if score >= 1.0:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


# def _interpretation(top_features: List[str]) -> str:
#     if not top_features:
#         return "Chunk này có điểm dị thường, nhưng chưa xác định được đặc trưng đóng góp chính."
#     labels = [FEATURE_INTERPRETATIONS.get(n, n) for n in top_features]
#     return (
#         "Chunk này có lỗi tái tạo cao hơn baseline genuine; "
#         "các đặc trưng đóng góp lớn nhất gồm: " + ", ".join(labels) + "."
#     )

# def _interpretation(top_features: List[str]) -> str:
#     if not top_features:
#         return "This chunk has an anomaly score, but the main contributing features have not been identified yet."
#     labels = [FEATURE_INTERPRETATIONS.get(n, n) for n in top_features]
#     return (
#         "This chunk has a higher reconstruction error than the genuine baseline; "
#         "the most contributing features include: " + ", ".join(labels) + "."
#     )

def _interpretation(top_features: List[str], level: str, norm_score: float) -> str:
    labels = [FEATURE_INTERPRETATIONS.get(n, n) for n in top_features]

    if not top_features:
        return (
            "This chunk was scored by the anomaly model, but no observed feature "
            "could be identified as a main contributor."
        )

    feature_text = ", ".join(labels)

    if level == "high":
        return (
            "This chunk exceeds the genuine baseline threshold and is considered highly anomalous. "
            "The main contributing features are: " + feature_text + "."
        )

    if level == "medium":
        return (
            "This chunk shows a moderate deviation from the genuine baseline. "
            "The largest reconstruction errors are associated with: " + feature_text + "."
        )

    return (
        "This chunk has a low anomaly score. "
        "The listed features are the largest reconstruction-error contributors within this chunk, "
        "but they do not necessarily indicate a strong anomaly: " + feature_text + "."
    )

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def infer_one_report(
    report_path: Path,
    model_dir: Path,
    output_dir: Path,
    device: torch.device,
    top_n: int,
) -> Path:
    report = load_json(report_path)
    video_id = report.get("video_metadata", {}).get("video_id", report_path.stem.replace("_report", ""))
    rows = parse_report(report_path)

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    visual_scaler = preprocessor["visual_scaler"]
    audio_scaler = preprocessor["audio_scaler"]

    threshold_info = load_json(model_dir / "threshold.json")
    threshold = float(threshold_info["threshold"])
    beta = float(threshold_info.get("beta", 1.0))

    model, _ = load_model(model_dir, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_evidence.json"

    if not rows:
        save_json(
            {
                "video_metadata": {
                    "video_id": video_id, "source_report": str(report_path),
                    "module": "module_3_mvae_poe", "status": "no_valid_chunks",
                },
                "chunks": {},
            },
            out_path,
        )
        return out_path

    # Build raw modality matrices (zero-filled, NaN tracked in masks)
    x_v_raw, x_a_raw, mask_v, mask_a, avail_v, avail_a = rows_to_modality_matrices(rows)

    # Scale: NaN preserved through scaler, then zero-fill for encoder
    x_v_nan = np.where(mask_v, x_v_raw, np.nan)
    x_a_nan = np.where(mask_a, x_a_raw, np.nan)
    x_v = np.where(mask_v, visual_scaler.transform(x_v_nan), 0.0).astype(np.float32)
    x_a = np.where(mask_a, audio_scaler.transform(x_a_nan), 0.0).astype(np.float32)

    # Run model
    with torch.no_grad():
        outputs = model(
            torch.tensor(x_v, device=device),
            torch.tensor(x_a, device=device),
            torch.tensor(avail_v, dtype=torch.bool, device=device),
            torch.tensor(avail_a, dtype=torch.bool, device=device),
        )

    recon_v = outputs["recon_visual"].cpu().numpy()
    recon_a = outputs["recon_audio"].cpu().numpy()
    mu_z = outputs["mu_z"].cpu().numpy()
    logvar_z = outputs["logvar_z"].cpu().numpy()

    vis_scores, aud_scores, kl_scores, joint_scores = compute_chunk_scores(
        recon_v, recon_a, x_v, x_a, mask_v.astype(float), mask_a.astype(float),
        avail_a, mu_z, logvar_z, beta=beta,
    )

    # Build per-feature reconstruction errors
    err_v_all = (x_v - recon_v) ** 2  # [N, visual_dim]
    err_a_all = (x_a - recon_a) ** 2  # [N, audio_dim]

    chunks: Dict[str, Any] = {}
    for idx, row in enumerate(rows):
        mv = mask_v[idx]   # [visual_dim] bool
        ma = mask_a[idx]   # [audio_dim] bool

        vis_score = float(vis_scores[idx])
        aud_score = float(aud_scores[idx]) if avail_a[idx] else None
        kl_score = float(kl_scores[idx])
        joint_score = float(joint_scores[idx])
        norm_score = float(min(1.0, joint_score / (threshold + 1e-12)))
        level = _level(norm_score)

        # Per-feature errors (None where feature was missing)
        per_feat_err: Dict[str, Optional[float]] = {}
        for i, name in enumerate(VISUAL_FEATURE_NAMES):
            per_feat_err[name] = float(err_v_all[idx, i]) if mv[i] else None
        for i, name in enumerate(AUDIO_FEATURE_NAMES):
            per_feat_err[name] = float(err_a_all[idx, i]) if ma[i] else None

        # Top N anomalous features (observed only)
        observed_errs = {k: v for k, v in per_feat_err.items() if v is not None}
        top_features = sorted(observed_errs, key=lambda k: observed_errs[k], reverse=True)[:top_n]

        # Clean feature values from original (un-scaled, un-imputed) feature_dict
        fd = row["feature_dict"]
        visual_features = {n: _clean(fd.get(n)) for n in VISUAL_FEATURE_NAMES}
        audio_features = {n: _clean(fd.get(n)) for n in AUDIO_FEATURE_NAMES}

        missing = [n for n, v in {**visual_features, **audio_features}.items() if v is None]
        modalities_analyzed = (["visual"] if avail_v[idx] else []) + (["audio_visual"] if avail_a[idx] else [])
        modalities_missing = ([] if avail_v[idx] else ["visual"]) + ([] if avail_a[idx] else ["audio_visual"])

        chunks[row["chunk_id"]] = {
            "time_metadata": row["time_metadata"],
            "frames_analyzed": row["frames_analyzed"],
            "features": {
                "visual": visual_features,
                "audio_visual": audio_features,
            },
            "missing_features": missing,
            "modalities_analyzed": modalities_analyzed,
            "modalities_missing": modalities_missing,
            "anomaly": {
                "visual_reconstruction_score": vis_score,
                "audio_reconstruction_score": aud_score,
                "kl_divergence": kl_score,
                "joint_anomaly_score": joint_score,
                "normalized_anomaly_score": norm_score,
                "threshold": threshold,
                "level": level,
            },
            "top_anomalous_features": top_features,
            "top_reconstruction_error_features": top_features,
            "per_feature_reconstruction_error": per_feat_err,
            "raw_text_evidence": row["raw_text_evidence"],
            "interpretation": _interpretation(top_features, level, norm_score),
        }

    save_json(
        {
            "video_metadata": {
                "video_id": video_id, "source_report": str(report_path),
                "module": "module_3_mvae_poe", "status": "analyzed",
            },
            "model_metadata": {
                "model_type": "mvae_poe",
                "visual_feature_names": VISUAL_FEATURE_NAMES,
                "audio_feature_names": AUDIO_FEATURE_NAMES,
                "threshold_percentile": threshold_info.get("threshold_percentile"),
                "threshold": threshold, "beta": beta,
            },
            "chunks": chunks,
        },
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MVAE-PoE inference on Module 2 reports.")
    parser.add_argument("--input", type=str, default=None, help="Single *_report.json file.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of *_report.json files.")
    parser.add_argument("--model_dir", type=str, default="./module3_models")
    parser.add_argument("--output_dir", type=str, default="./evidence_reports")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        raise ValueError("Provide --input or --input_dir.")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    report_paths: List[Path] = []
    if args.input:
        report_paths.append(Path(args.input))
    if args.input_dir:
        report_paths.extend(iter_report_files(Path(args.input_dir)))

    if not report_paths:
        raise FileNotFoundError("No report files found.")

    written = [
        str(infer_one_report(p, Path(args.model_dir), Path(args.output_dir), device, args.top_n))
        for p in report_paths
    ]
    print(json.dumps({"status": "ok", "num_reports": len(written), "outputs": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
