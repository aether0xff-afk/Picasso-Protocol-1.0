from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.compact_models import encode_text, load_public_encoder
from picasso_protocol.latent_io import save_latent

PUBLIC_MODEL = Path("models/public/artist_y_public_encoder.pt")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_latent_dropout(latent: torch.Tensor, probability: float) -> torch.Tensor:
    if not probability:
        return latent
    if not 0.0 <= probability < 1.0:
        raise ValueError("--shield-dropout must be in the range [0.0, 1.0)")
    keep_probability = 1.0 - probability
    mask = torch.empty_like(latent).bernoulli_(keep_probability)
    return latent * mask / keep_probability


def encode(args: argparse.Namespace) -> None:
    current_device = device()
    encoder, config = load_public_encoder(args.model, current_device)
    tokens = torch.tensor([encode_text(args.text, config.max_bytes)], device=current_device)
    with torch.no_grad():
        latent = encoder(tokens)
        if args.shield_samples > 1:
            clean_latent = latent
            samples = [
                apply_latent_dropout(clean_latent, args.shield_dropout)
                + torch.randn_like(clean_latent) * args.shield_noise
                for _ in range(args.shield_samples)
            ]
            latent = torch.stack(samples, dim=1)
        else:
            latent = apply_latent_dropout(latent, args.shield_dropout)
            if args.shield_noise:
                latent = latent + torch.randn_like(latent) * args.shield_noise
    save_latent(args.output, latent)
    print("[Artist Y Public Encoder]")
    print(f"saved: {args.output}")
    print(f"latent shape: {list(latent.shape)}")
    print(f"shield noise: {args.shield_noise}")
    print(f"shield dropout: {args.shield_dropout}")
    print(f"shield samples: {args.shield_samples}")


def status(args: argparse.Namespace) -> None:
    _, config = load_public_encoder(args.model, device())
    print("[Artist Y Public Encoder]")
    print(f"device: {device()}")
    print(f"model: {args.model}")
    print("contents: encoder only")
    print(f"latent shape: [1, {config.max_bytes}, {config.latent_dim}]")
    print(f"maximum UTF-8 bytes: {config.max_bytes - 1}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Artist Y public encoder CLI")
    subparsers = result.add_subparsers(dest="command", required=True)
    encode_parser = subparsers.add_parser("encode", help="encode text into latent JSON")
    encode_parser.add_argument("text")
    encode_parser.add_argument("-o", "--output", type=Path, default=Path("picasso_latent.json"))
    encode_parser.add_argument("--model", type=Path, default=PUBLIC_MODEL)
    encode_parser.add_argument(
        "--shield-noise",
        type=float,
        default=0.0,
        help="add Gaussian noise to the public latent output",
    )
    encode_parser.add_argument(
        "--shield-dropout",
        type=float,
        default=0.0,
        help="randomly mask latent values; use only with a decoder trained for this shield",
    )
    encode_parser.add_argument(
        "--shield-samples",
        type=int,
        default=1,
        help="emit multiple shielded samples for private majority-vote decoding",
    )
    encode_parser.set_defaults(handler=encode)
    status_parser = subparsers.add_parser("status", help="show public model status")
    status_parser.add_argument("--model", type=Path, default=PUBLIC_MODEL)
    status_parser.set_defaults(handler=status)
    return result


def main() -> None:
    try:
        args = parser().parse_args()
        args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
