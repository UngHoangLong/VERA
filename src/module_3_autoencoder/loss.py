"""
loss.py — ELBO loss and anomaly scoring for MVAE-PoE.

ELBO = masked_recon_visual + masked_recon_audio + beta * KL

Key design decisions:
  - Masked reconstruction: only observed features (mask=1) contribute.
    This means a silent chunk (null audio) does not penalize the model
    for failing to reconstruct fake audio values.
  - KL divergence pushes the posterior q(z|x) toward N(0,I), ensuring
    the latent space stays compact and anomalies produce high KL scores.
  - Anomaly score = visual_score + audio_score + beta * kl_score.
    Each term can be reported independently in the evidence JSON.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# ELBO loss (used during training)
# ---------------------------------------------------------------------------

def elbo_loss(
    outputs: Dict[str, torch.Tensor],
    x_visual: torch.Tensor,
    x_audio: torch.Tensor,
    mask_visual: torch.Tensor,
    mask_audio: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute ELBO = masked_recon_visual + masked_recon_audio + beta * KL.

    Args:
        outputs    : dict from MVAEPoE.forward()
        x_visual   : [B, visual_dim] zero-filled scaled input
        x_audio    : [B, audio_dim]  zero-filled scaled input
        mask_visual: [B, visual_dim] float — 1.0 where feature was observed
        mask_audio : [B, audio_dim]  float
        beta       : KL weight (β=1 → standard ELBO)

    Returns:
        total_loss : scalar tensor for backprop
        breakdown  : dict of individual loss components (for logging)
    """
    recon_v = outputs["recon_visual"]
    recon_a = outputs["recon_audio"]
    mu_z = outputs["mu_z"]
    logvar_z = outputs["logvar_z"]

    def _masked_mse(x: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        err = (x - recon).pow(2)
        n_obs = mask.sum().clamp(min=1.0)
        return (err * mask).sum() / n_obs

    l_v = _masked_mse(x_visual, recon_v, mask_visual)
    l_a = _masked_mse(x_audio, recon_a, mask_audio)
    kl = -0.5 * (1.0 + logvar_z - mu_z.pow(2) - logvar_z.exp()).mean()
    total = l_v + l_a + beta * kl

    return total, {
        "recon_visual": float(l_v.item()),
        "recon_audio": float(l_a.item()),
        "kl": float(kl.item()),
        "total": float(total.item()),
    }


# ---------------------------------------------------------------------------
# Anomaly scoring (used at inference and for threshold calibration)
# ---------------------------------------------------------------------------

def compute_chunk_scores(
    recon_v: np.ndarray,
    recon_a: np.ndarray,
    x_v: np.ndarray,
    x_a: np.ndarray,
    mask_v: np.ndarray,
    mask_a: np.ndarray,
    avail_a: np.ndarray,
    mu_z: np.ndarray,
    logvar_z: np.ndarray,
    beta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-sample anomaly scores (numpy, used after inference).

    Returns:
        visual_scores : [N] — masked MSE over visual features
        audio_scores  : [N] — masked MSE over audio features (0 if silent)
        kl_scores     : [N] — KL divergence from prior
        joint_scores  : [N] — total anomaly score for threshold comparison
    """
    # Visual reconstruction score
    n_v = mask_v.sum(axis=1).clip(min=1)
    visual_scores = ((x_v - recon_v) ** 2 * mask_v).sum(axis=1) / n_v

    # Audio reconstruction score (0 if no audio features observed)
    n_a = mask_a.sum(axis=1).clip(min=1)
    raw_audio = ((x_a - recon_a) ** 2 * mask_a).sum(axis=1) / n_a
    audio_scores = raw_audio * avail_a.astype(float)

    # KL divergence per sample
    kl_scores = -0.5 * (1.0 + logvar_z - mu_z ** 2 - np.exp(logvar_z)).sum(axis=1)

    joint_scores = visual_scores + audio_scores + beta * kl_scores
    return visual_scores, audio_scores, kl_scores, joint_scores


@torch.no_grad()
def compute_joint_scores_dataset(
    model: nn.Module,
    dataset,
    device: torch.device,
    beta: float = 1.0,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Run the model over a ModalityDataset and return joint anomaly scores.
    Used for threshold calibration after training.
    """
    model.eval()
    all_scores: List[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for x_v, x_a, mask_v, mask_a, avail_v, avail_a in loader:
        x_v, x_a = x_v.to(device), x_a.to(device)
        mask_v, mask_a = mask_v.to(device), mask_a.to(device)
        avail_v, avail_a = avail_v.to(device), avail_a.to(device)

        outputs = model(x_v, x_a, avail_v, avail_a)
        mu_z, logvar_z = outputs["mu_z"], outputs["logvar_z"]
        recon_v, recon_a = outputs["recon_visual"], outputs["recon_audio"]

        n_v = mask_v.sum(dim=1).clamp(min=1.0)
        score_v = ((x_v - recon_v).pow(2) * mask_v).sum(dim=1) / n_v

        n_a = mask_a.sum(dim=1).clamp(min=1.0)
        score_a = ((x_a - recon_a).pow(2) * mask_a).sum(dim=1) / n_a
        score_a = score_a * avail_a.float()

        kl = -0.5 * (1.0 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1)
        joint = score_v + score_a + beta * kl

        all_scores.append(joint.cpu().numpy())

    return np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
