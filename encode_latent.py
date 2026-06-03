from __future__ import annotations

import argparse
from pathlib import Path

import torch

from picasso_protocol.compact_models import encode_text, load_public_encoder
from picasso_protocol.latent_io import save_latent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/public/artist_y_public_encoder.pt"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, config = load_public_encoder(args.model, device)
    tokens = torch.tensor([encode_text(args.text, config.max_bytes)], device=device)
    with torch.no_grad():
        latent = encoder(tokens)
    save_latent(args.output, latent)
    print(f"saved={args.output}")
    print(f"latent_shape={list(latent.shape)}")


if __name__ == "__main__":
    main()
