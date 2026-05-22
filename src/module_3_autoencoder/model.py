from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.logvar_head = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 4.0)
        return mu, logvar


class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def product_of_experts(
    mus: List[torch.Tensor],
    logvars: List[torch.Tensor],
    masks: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines N expert distributions + standard Gaussian prior via PoE.
    masks: list of [B] bool tensors - True if that expert's modality is available.
    """
    # Start with N(0,I) prior: precision=1, mean=0
    precision = torch.ones_like(mus[0])
    mean = torch.zeros_like(mus[0])

    for i, (mu, lv) in enumerate(zip(mus, logvars)):
        prec_i = torch.exp(-lv)
        if masks is not None:
            m = masks[i].float().unsqueeze(-1)  # [B, 1] → broadcasts to [B, latent_dim]
            precision = precision + prec_i * m
            mean = mean + mu * prec_i * m
        else:
            precision = precision + prec_i
            mean = mean + mu * prec_i

    joint_logvar = -torch.log(precision + 1e-8)
    joint_mu = mean / (precision + 1e-8)
    return joint_mu, joint_logvar


class MVAEPoE(nn.Module):
    """
    Multimodal VAE with Product of Experts for deepfake anomaly detection.

    Two modalities:
      - Visual  : visual_spatial features (always available)
      - Audio/AV: audio + audio-visual consistency (null when silent)

    Missing modalities are excluded from the PoE fusion via mask,
    so silent chunks do not produce fake audio anomaly scores.
    """

    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        hidden_dims: List[int],
        latent_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.visual_encoder = ModalityEncoder(visual_dim, hidden_dims, latent_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dims, latent_dim, dropout)

        dec_hidden = list(reversed(hidden_dims))
        self.visual_decoder = ModalityDecoder(latent_dim, dec_hidden, visual_dim, dropout)
        self.audio_decoder = ModalityDecoder(latent_dim, dec_hidden, audio_dim, dropout)

    def encode(
        self,
        x_visual: torch.Tensor,
        x_audio: torch.Tensor,
        visual_avail: torch.Tensor,
        audio_avail: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_v, lv_v = self.visual_encoder(x_visual)
        mu_a, lv_a = self.audio_encoder(x_audio)
        return product_of_experts([mu_v, mu_a], [lv_v, lv_a], masks=[visual_avail, audio_avail])

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.visual_decoder(z), self.audio_decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(
        self,
        x_visual: torch.Tensor,
        x_audio: torch.Tensor,
        visual_avail: torch.Tensor,
        audio_avail: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu_z, logvar_z = self.encode(x_visual, x_audio, visual_avail, audio_avail)
        z = self.reparameterize(mu_z, logvar_z)
        recon_v, recon_a = self.decode(z)
        return {
            "recon_visual": recon_v,
            "recon_audio": recon_a,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
        }

    def config(self) -> Dict:
        return {
            "visual_dim": self.visual_dim,
            "audio_dim": self.audio_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
        }


def build_mvae_poe(
    visual_dim: int,
    audio_dim: int,
    hidden_dims: List[int],
    latent_dim: int = 6,
    dropout: float = 0.1,
) -> MVAEPoE:
    return MVAEPoE(visual_dim, audio_dim, hidden_dims, latent_dim, dropout)
