from typing import Dict

import torch
from torch import nn


class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 10, latent_dim: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        if latent_dim >= input_dim:
            raise ValueError("latent_dim should be smaller than input_dim.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def config(self) -> Dict[str, float]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
        }


def build_autoencoder(input_dim: int, hidden_dim: int = 10, latent_dim: int = 5, dropout: float = 0.05) -> MLPAutoEncoder:
    return MLPAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)
