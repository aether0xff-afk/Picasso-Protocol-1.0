from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

FORMAT = "picasso-latent-v1"


def latent_to_payload(latent: torch.Tensor) -> dict[str, object]:
    array = latent.detach().cpu().numpy().astype(np.float32)
    return {
        "format": FORMAT,
        "dtype": "float32",
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def payload_to_latent(payload: dict[str, object]) -> torch.Tensor:
    if payload.get("format") != FORMAT or payload.get("dtype") != "float32":
        raise ValueError("unsupported latent payload")
    shape = tuple(int(value) for value in payload["shape"])
    if len(shape) not in (3, 4) or shape[0] != 1:
        raise ValueError(f"invalid latent shape: {shape}")
    raw = base64.b64decode(str(payload["data"]), validate=True)
    expected = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if len(raw) != expected:
        raise ValueError(f"latent payload has {len(raw)} bytes; expected {expected}")
    array = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(array)


def save_latent(path: Path, latent: torch.Tensor) -> None:
    path.write_text(json.dumps(latent_to_payload(latent)), encoding="utf-8")


def load_latent(path: Path) -> torch.Tensor:
    return payload_to_latent(json.loads(path.read_text(encoding="utf-8")))
