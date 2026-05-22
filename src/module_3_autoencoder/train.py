"""
train.py — Training pipeline for MVAE-PoE (Module 3).

Usage:
    cd src/module_3_autoencoder
    python train.py --input_dir ../../final_reports_genuine --model_dir ./module3_models

Entry point: main()
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AUDIO_FEATURE_NAMES, VISUAL_FEATURE_NAMES
from dataset import (
    ModalityDataset,
    ModalityScaler,
    iter_report_files,
    parse_reports,
    rows_to_modality_matrices,
    save_json,
    split_report_files_by_video,
)
from loss import compute_joint_scores_dataset, elbo_loss
from model import build_mvae_poe


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data preprocessing helpers
# ---------------------------------------------------------------------------

def _scale_split(
    rows_raw_v: np.ndarray,
    rows_raw_a: np.ndarray,
    mask_v: np.ndarray,
    mask_a: np.ndarray,
    visual_scaler: ModalityScaler,
    audio_scaler: ModalityScaler,
    fit: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale visual and audio feature matrices.
    NaN is preserved through transform; then zero-filled for encoder input.
    """
    x_v_nan = np.where(mask_v, rows_raw_v, np.nan)
    x_a_nan = np.where(mask_a, rows_raw_a, np.nan)

    if fit:
        x_v_scaled = visual_scaler.fit_transform(x_v_nan)
        x_a_scaled = audio_scaler.fit_transform(x_a_nan)
    else:
        x_v_scaled = visual_scaler.transform(x_v_nan)
        x_a_scaled = audio_scaler.transform(x_a_nan)

    x_v = np.where(mask_v, x_v_scaled, 0.0).astype(np.float32)
    x_a = np.where(mask_a, x_a_scaled, 0.0).astype(np.float32)
    return x_v, x_a


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    train_ds: ModalityDataset,
    val_ds: ModalityDataset,
    visual_dim: int,
    audio_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    beta: float,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict]:
    model = build_mvae_poe(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history: Dict = {
        "train_total": [], "train_recon_visual": [],
        "train_recon_audio": [], "train_kl": [], "val_joint_score": [],
    }

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        te = time.time()
        model.train()
        totals, rvs, ras, kls = [], [], [], []

        for x_v, x_a, mask_v, mask_a, avail_v, avail_a in train_loader:
            x_v, x_a = x_v.to(device), x_a.to(device)
            mask_v, mask_a = mask_v.to(device), mask_a.to(device)
            avail_v, avail_a = avail_v.to(device), avail_a.to(device)

            optimizer.zero_grad()
            outputs = model(x_v, x_a, avail_v, avail_a)
            loss, bd = elbo_loss(outputs, x_v, x_a, mask_v, mask_a, beta=beta)
            loss.backward()
            optimizer.step()

            totals.append(bd["total"])
            rvs.append(bd["recon_visual"])
            ras.append(bd["recon_audio"])
            kls.append(bd["kl"])

        train_total = float(np.mean(totals))
        train_rv = float(np.mean(rvs))
        train_ra = float(np.mean(ras))
        train_kl = float(np.mean(kls))
        val_score = float(np.mean(compute_joint_scores_dataset(model, val_ds, device, beta)))

        history["train_total"].append(train_total)
        history["train_recon_visual"].append(train_rv)
        history["train_recon_audio"].append(train_ra)
        history["train_kl"].append(train_kl)
        history["val_joint_score"].append(val_score)

        print(
            f"Epoch [{epoch}/{epochs}] | total={train_total:.4f} | "
            f"recon_v={train_rv:.4f} | recon_a={train_ra:.4f} | kl={train_kl:.4f} | "
            f"val_score={val_score:.4f} | "
            f"epoch={time.time()-te:.1f}s | elapsed={(time.time()-t0)/60:.1f}min",
            flush=True,
        )

        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best val_score={best_val:.6f}", flush=True)
            break

    print(f"Training finished in {(time.time()-t0)/60:.2f}min.", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVAE-PoE on genuine Module 2 reports.")
    parser.add_argument("--input_dir", type=str, default="./final_reports_genuine")
    parser.add_argument("--model_dir", type=str, default="./module3_models")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    report_files = iter_report_files(input_dir)
    if not report_files:
        raise FileNotFoundError(f"No *_report.json files found in {input_dir}")

    train_files, val_files = split_report_files_by_video(report_files, args.val_ratio, args.seed)
    print("Train:", [str(p) for p in train_files])
    print("Val:", [str(p) for p in val_files])

    train_rows = parse_reports(train_files)
    val_rows = parse_reports(val_files) if val_files else []

    if not train_rows:
        raise RuntimeError("No chunks parsed from training reports.")

    print(
        f"Chunks | train={len(train_rows)} val={len(val_rows)} | "
        f"files: train={len(train_files)} val={len(val_files)}",
        flush=True,
    )

    visual_scaler = ModalityScaler()
    audio_scaler = ModalityScaler()

    x_v_tr, x_a_tr, mv_tr, ma_tr, av_tr, aa_tr = rows_to_modality_matrices(train_rows)
    x_v_train, x_a_train = _scale_split(x_v_tr, x_a_tr, mv_tr, ma_tr, visual_scaler, audio_scaler, fit=True)
    train_ds = ModalityDataset(x_v_train, x_a_train, mv_tr, ma_tr, av_tr, aa_tr)

    if val_rows:
        x_v_vr, x_a_vr, mv_vr, ma_vr, av_vr, aa_vr = rows_to_modality_matrices(val_rows)
        x_v_val, x_a_val = _scale_split(x_v_vr, x_a_vr, mv_vr, ma_vr, visual_scaler, audio_scaler, fit=False)
        val_ds = ModalityDataset(x_v_val, x_a_val, mv_vr, ma_vr, av_vr, aa_vr)
    else:
        val_ds = train_ds

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    model, history = train_model(
        train_ds=train_ds, val_ds=val_ds,
        visual_dim=len(VISUAL_FEATURE_NAMES), audio_dim=len(AUDIO_FEATURE_NAMES),
        hidden_dims=args.hidden_dims, latent_dim=args.latent_dim,
        dropout=args.dropout, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
        beta=args.beta, device=device,
    )

    threshold_ds = val_ds if val_rows else train_ds
    scores = compute_joint_scores_dataset(model, threshold_ds, device, args.beta)
    threshold = float(np.percentile(scores, args.threshold_percentile))
    if threshold <= 0:
        threshold = float(np.max(scores) + 1e-8)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "visual_feature_names": VISUAL_FEATURE_NAMES,
            "audio_feature_names": AUDIO_FEATURE_NAMES,
            "beta": args.beta,
        },
        model_dir / "mvae_poe.pt",
    )
    joblib.dump(
        {"visual_scaler": visual_scaler, "audio_scaler": audio_scaler},
        model_dir / "preprocessor.joblib",
    )
    threshold_payload = {
        "threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": "validation" if val_rows else "training",
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": float(np.max(scores)),
        "beta": args.beta,
    }
    save_json(threshold_payload, model_dir / "threshold.json")
    save_json(
        {
            "num_train_chunks": len(train_rows), "num_val_chunks": len(val_rows),
            "visual_feature_names": VISUAL_FEATURE_NAMES,
            "audio_feature_names": AUDIO_FEATURE_NAMES,
            "model_config": model.config(),
            "beta": args.beta, "threshold": threshold_payload, "history": history,
        },
        model_dir / "train_summary.json",
    )

    print(json.dumps({
        "status": "ok", "model_dir": str(model_dir),
        "num_train_chunks": len(train_rows), "num_val_chunks": len(val_rows),
        "threshold": threshold,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
