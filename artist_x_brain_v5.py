from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from picasso_protocol.brain_models import load_private_brain_fragment
from picasso_protocol.compact_models import decode_tokens
from picasso_protocol.latent_io import load_latent

PRIVATE_MODEL = Path("models/private_v5b/artist_x_private_brain_fragment.pt")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(args: argparse.Namespace) -> None:
    current_device = device()
    fragment, config = load_private_brain_fragment(args.model, current_device)
    latent = load_latent(args.input).to(current_device)
    expected = (1, config.max_bytes, config.latent_dim)
    if tuple(latent.shape) != expected:
        raise ValueError(f"latent shape {tuple(latent.shape)} does not match {expected}")
    with torch.no_grad():
        tokens = fragment.generate(latent)[0].tolist()
    print("[Artist X Private Brain Fragment]")
    print(decode_tokens(tokens))


def status(args: argparse.Namespace) -> None:
    _, config = load_private_brain_fragment(args.model, device())
    print("[Artist X Private Brain Fragment]")
    print(f"device: {device()}")
    print(f"model: {args.model}")
    print("contents: private brain fragment only")
    print(f"expected latent shape: [1, {config.max_bytes}, {config.latent_dim}]")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Artist X private brain fragment CLI")
    subparsers = result.add_subparsers(dest="command", required=True)
    decode_parser = subparsers.add_parser("decode", help="decode V5 brain latent into text")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--model", type=Path, default=PRIVATE_MODEL)
    decode_parser.set_defaults(handler=decode)
    status_parser = subparsers.add_parser("status", help="show private V5 fragment status")
    status_parser.add_argument("--model", type=Path, default=PRIVATE_MODEL)
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
