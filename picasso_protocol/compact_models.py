from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

PAD = 256
EOS = 258
VOCAB_SIZE = 259


@dataclass
class ModelConfig:
    max_bytes: int = 160
    embedding_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 128
    layers: int = 2
    dropout: float = 0.15
    latent_noise: float = 0.02
    latent_dropout: float = 0.0
    architecture: str = "sequence"


def encode_text(text: str, max_bytes: int) -> list[int]:
    payload = list(text.encode("utf-8"))
    if len(payload) > max_bytes - 1:
        raise ValueError(f"text requires {len(payload)} UTF-8 bytes; maximum is {max_bytes - 1}")
    return payload + [EOS] + [PAD] * (max_bytes - len(payload) - 1)


def decode_tokens(tokens: list[int]) -> str:
    payload: list[int] = []
    for token in tokens:
        if token == EOS:
            break
        if 0 <= token <= 255:
            payload.append(token)
    return bytes(payload).decode("utf-8", errors="replace")


class PublicArtistYEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
        self.network = nn.Sequential(
            nn.Conv1d(config.embedding_dim, config.latent_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(tokens).transpose(1, 2)).transpose(1, 2)


class PrivateArtistXDecoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(config.latent_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, VOCAB_SIZE, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent.transpose(1, 2)).transpose(1, 2)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward(latent).argmax(dim=-1)


def load_public_encoder(path: Path, device: torch.device) -> tuple[PublicArtistYEncoder, ModelConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if set(checkpoint) != {"config", "encoder"}:
        raise ValueError("public Artist Y model must contain only config and encoder")
    config = ModelConfig(**checkpoint["config"])
    model = PublicArtistYEncoder(config).to(device)
    model.load_state_dict(checkpoint["encoder"])
    return model.eval(), config


def load_private_decoder(path: Path, device: torch.device) -> tuple[PrivateArtistXDecoder, ModelConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if set(checkpoint) != {"config", "decoder"}:
        raise ValueError("private Artist X model must contain only config and decoder")
    config = ModelConfig(**checkpoint["config"])
    model = PrivateArtistXDecoder(config).to(device)
    model.load_state_dict(checkpoint["decoder"])
    return model.eval(), config


def export_config(config: ModelConfig) -> dict[str, object]:
    return asdict(config)
