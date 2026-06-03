from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from picasso_protocol.compact_models import PAD, VOCAB_SIZE, decode_tokens, encode_text


@dataclass
class BrainConfig:
    max_bytes: int = 160
    embedding_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 192
    private_dim: int = 96
    dropout: float = 0.15
    latent_noise: float = 1.0
    latent_dropout: float = 0.75
    architecture: str = "brain-v5"


def apply_brain_shield(
    latent: torch.Tensor,
    dropout: float,
    noise: float,
) -> torch.Tensor:
    if dropout:
        if not 0.0 <= dropout < 1.0:
            raise ValueError("latent dropout must be in the range [0.0, 1.0)")
        keep_probability = 1.0 - dropout
        latent = latent * torch.empty_like(latent).bernoulli_(keep_probability) / keep_probability
    if noise:
        latent = latent + torch.randn_like(latent) * noise
    return latent


class PublicBrainFragmentY(nn.Module):
    def __init__(self, config: BrainConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
        self.cortex = nn.Sequential(
            nn.Conv1d(config.embedding_dim, config.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, config.latent_dim, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        return self.cortex(embedded).transpose(1, 2)


class PrivateBrainFragmentX(nn.Module):
    def __init__(self, config: BrainConfig) -> None:
        super().__init__()
        self.private_context = nn.Parameter(
            torch.randn(1, config.max_bytes, config.private_dim) * 0.02
        )
        self.resolver = nn.Sequential(
            nn.Conv1d(
                config.latent_dim + config.private_dim,
                config.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, VOCAB_SIZE, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        context = self.private_context.expand(latent.shape[0], -1, -1)
        joined = torch.cat([latent, context], dim=-1)
        return self.resolver(joined.transpose(1, 2)).transpose(1, 2)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward(latent).argmax(dim=-1)


class PicassoBrain(nn.Module):
    def __init__(self, config: BrainConfig) -> None:
        super().__init__()
        self.config = config
        self.public_fragment = PublicBrainFragmentY(config)
        self.private_fragment = PrivateBrainFragmentX(config)

    def latent(self, tokens: torch.Tensor, shielded: bool = False) -> torch.Tensor:
        latent = self.public_fragment(tokens)
        if shielded:
            latent = apply_brain_shield(
                latent,
                dropout=self.config.latent_dropout,
                noise=self.config.latent_noise,
            )
        return latent

    def forward(self, tokens: torch.Tensor, shielded: bool = False) -> torch.Tensor:
        return self.private_fragment(self.latent(tokens, shielded=shielded))


def load_public_brain_fragment(
    path: Path, device: torch.device
) -> tuple[PublicBrainFragmentY, BrainConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if set(checkpoint) != {"config", "public_fragment"}:
        raise ValueError("public Artist Y brain fragment must contain only config and public_fragment")
    config = BrainConfig(**checkpoint["config"])
    model = PublicBrainFragmentY(config).to(device)
    model.load_state_dict(checkpoint["public_fragment"])
    return model.eval(), config


def load_private_brain_fragment(
    path: Path, device: torch.device
) -> tuple[PrivateBrainFragmentX, BrainConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if set(checkpoint) != {"config", "private_fragment"}:
        raise ValueError("private Artist X brain fragment must contain only config and private_fragment")
    config = BrainConfig(**checkpoint["config"])
    model = PrivateBrainFragmentX(config).to(device)
    model.load_state_dict(checkpoint["private_fragment"])
    return model.eval(), config


def export_brain_config(config: BrainConfig) -> dict[str, object]:
    return asdict(config)
