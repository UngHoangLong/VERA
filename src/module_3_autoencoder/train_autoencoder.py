import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import FEATURE_NAMES
from features import iter_report_files, parse_reports, rows_to_matrix, save_json, split_report_files_by_video
from model import build_autoencoder


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruction_errors(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    errors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), 2048):
            batch = tensor[start:start + 2048]
            recon = model(batch)
            err = torch.mean((batch - recon) ** 2, dim=1)
            errors.append(err.detach().cpu().numpy())
    if not errors:
        return np.array([], dtype=np.float32)
    return np.concatenate(errors)


def train_model(
    x_train: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, List[float]]]:

    model = build_autoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    train_start_time = time.time()

    for _epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        model.train()
        batch_losses = []

        for batch_idx, (batch,) in enumerate(train_loader, start=1):
            batch = batch.to(device)

            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        if len(x_val) > 0:
            val_errors = reconstruction_errors(model, x_val, device)
            val_loss = float(np.mean(val_errors))
        else:
            val_loss = train_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - train_start_time

        print(
            f"Epoch [{_epoch}/{epochs}] | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"epoch_time={epoch_time:.2f}s | "
            f"elapsed_time={elapsed_time / 60:.2f}min",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {_epoch}. "
                f"Best val_loss={best_val_loss:.6f}",
                flush=True
            )
            break

    total_train_time = time.time() - train_start_time

    print(
        f"Training finished in {total_train_time:.2f}s "
        f"({total_train_time / 60:.2f}min).",
        flush=True
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Module 3 AutoEncoder on genuine Module 2 reports.")
    parser.add_argument("--input_dir", type=str, default="./final_reports_genuine")
    parser.add_argument("--model_dir", type=str, default="./module3_models")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
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
    train_files, val_files = split_report_files_by_video(
        report_files,
        args.val_ratio,
        args.seed
    )

    print("Train files:")
    for path in train_files:
        print(f"  {path}")

    print("Validation files:")
    for path in val_files:
        print(f"  {path}")

    train_rows = parse_reports(train_files)
    val_rows = parse_reports(val_files) if val_files else []

    print(
        f"Split summary | "
        f"train_files={len(train_files)} | "
        f"val_files={len(val_files)} | "
        f"train_chunks={len(train_rows)} | "
        f"val_chunks={len(val_rows)}",
        flush=True
    )
    if not train_rows:
        raise RuntimeError("No chunks were parsed from training reports.")
    x_train_raw = rows_to_matrix(train_rows)
    all_nan_cols = np.isnan(x_train_raw).all(axis=0)

    if np.any(all_nan_cols):
        missing_features = [
            FEATURE_NAMES[i]
            for i, is_all_nan in enumerate(all_nan_cols)
            if is_all_nan
        ]
        raise RuntimeError(
            "These features are all-NaN in training data: "
            + ", ".join(missing_features)
            + ". Please check Module 2 output or remove/fallback these features."
        )
    x_val_raw = rows_to_matrix(val_rows) if val_rows else np.empty((0, x_train_raw.shape[1]), dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw)).astype(np.float32)
    if len(x_val_raw) > 0:
        x_val = scaler.transform(imputer.transform(x_val_raw)).astype(np.float32)
    else:
        x_val = np.empty((0, x_train.shape[1]), dtype=np.float32)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, history = train_model(x_train, x_val, x_train.shape[1], args.hidden_dim, args.latent_dim, args.dropout, args.epochs, args.batch_size, args.lr, args.weight_decay, args.patience, device)
    threshold_source = x_val if len(x_val) > 0 else x_train
    threshold_errors = reconstruction_errors(model, threshold_source, device)
    threshold = float(np.percentile(threshold_errors, args.threshold_percentile))
    if threshold <= 0:
        threshold = float(np.max(threshold_errors) + 1e-8)
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config(), "feature_names": FEATURE_NAMES}, model_dir / "autoencoder.pt")
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_names": FEATURE_NAMES}, model_dir / "preprocessor.joblib")
    threshold_payload = {
        "threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": "validation" if len(x_val) > 0 else "training",
        "mean_reconstruction_error": float(np.mean(threshold_errors)),
        "median_reconstruction_error": float(np.median(threshold_errors)),
        "max_reconstruction_error": float(np.max(threshold_errors)),
    }
    save_json(threshold_payload, model_dir / "threshold.json")
    summary = {
        "input_dir": str(input_dir),
        "num_report_files": len(report_files),
        "num_train_files": len(train_files),
        "num_val_files": len(val_files),
        "num_train_chunks": len(train_rows),
        "num_val_chunks": len(val_rows),
        "feature_names": FEATURE_NAMES,
        "model_config": model.config(),
        "threshold": threshold_payload,
        "history": history,
    }
    save_json(summary, model_dir / "train_summary.json")
    print(json.dumps({"status": "ok", "model_dir": str(model_dir), "num_train_chunks": len(train_rows), "num_val_chunks": len(val_rows), "threshold": threshold}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
