from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.brain_models import (
    apply_brain_shield,
    load_public_brain_fragment,
)
from picasso_protocol.compact_models import encode_text
from picasso_protocol.latent_io import save_latent

PUBLIC_MODEL = Path("models/public_v5b/artist_y_public_brain_fragment.pt")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(args: argparse.Namespace) -> None:
    current_device = device()
    fragment, config = load_public_brain_fragment(args.model, current_device)
    tokens = torch.tensor([encode_text(args.text, config.max_bytes)], device=current_device)
    with torch.no_grad():
        latent = fragment(tokens)
        if args.shield:
            latent = apply_brain_shield(
                latent,
                dropout=args.shield_dropout,
                noise=args.shield_noise,
            )
    save_latent(args.output, latent)
    print("[Artist Y Public Brain Fragment]")
    print(f"saved: {args.output}")
    print(f"latent shape: {list(latent.shape)}")
    print(f"shield: {args.shield}")
    print(f"shield dropout: {args.shield_dropout}")
    print(f"shield noise: {args.shield_noise}")


def status(args: argparse.Namespace) -> None:
    _, config = load_public_brain_fragment(args.model, device())
    print("[Artist Y Public Brain Fragment]")
    print(f"device: {device()}")
    print(f"model: {args.model}")
    print("contents: public brain fragment only")
    print(f"latent shape: [1, {config.max_bytes}, {config.latent_dim}]")
    print(f"default shield dropout: {config.latent_dropout}")
    print(f"default shield noise: {config.latent_noise}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Artist Y public brain fragment CLI")
    subparsers = result.add_subparsers(dest="command", required=True)
    encode_parser = subparsers.add_parser("encode", help="encode text into V5 brain latent")
    encode_parser.add_argument("text")
    encode_parser.add_argument("-o", "--output", type=Path, default=Path("brain_v5_latent.json"))
    encode_parser.add_argument("--model", type=Path, default=PUBLIC_MODEL)
    encode_parser.add_argument("--shield", action=argparse.BooleanOptionalAction, default=True)
    encode_parser.add_argument("--shield-dropout", type=float)
    encode_parser.add_argument("--shield-noise", type=float)
    encode_parser.set_defaults(handler=encode)
    status_parser = subparsers.add_parser("status", help="show public V5 fragment status")
    status_parser.add_argument("--model", type=Path, default=PUBLIC_MODEL)
    status_parser.set_defaults(handler=status)
    return result


def main() -> None:
    try:
        args = parser().parse_args()
        if hasattr(args, "shield_dropout") and args.shield_dropout is None:
            _, config = load_public_brain_fragment(args.model, device())
            args.shield_dropout = config.latent_dropout
            args.shield_noise = config.latent_noise if args.shield_noise is None else args.shield_noise
        args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
