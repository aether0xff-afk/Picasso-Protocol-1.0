from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.compact_models import decode_tokens, load_private_decoder
from picasso_protocol.latent_io import load_latent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/private/artist_x_private_decoder.pt"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, config = load_private_decoder(args.model, device)
    latent = load_latent(args.input).to(device)
    if tuple(latent.shape[1:]) != (config.max_bytes, config.latent_dim):
        raise ValueError(f"latent shape {tuple(latent.shape)} does not match private decoder")
    with torch.no_grad():
        tokens = decoder.generate(latent)[0].tolist()
    print(decode_tokens(tokens))


if __name__ == "__main__":
    main()
