from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from picasso_protocol.compact_models import decode_tokens, load_private_decoder
from picasso_protocol.latent_io import load_latent

PRIVATE_MODEL = Path("models/private/artist_x_private_decoder.pt")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(args: argparse.Namespace) -> None:
    current_device = device()
    decoder, config = load_private_decoder(args.model, current_device)
    latent = load_latent(args.input).to(current_device)
    expected_single = (1, config.max_bytes, config.latent_dim)
    if len(latent.shape) == 3 and tuple(latent.shape) != expected_single:
        raise ValueError(f"latent shape {tuple(latent.shape)} does not match {expected_single}")
    if len(latent.shape) == 4 and tuple(latent.shape[2:]) != (config.max_bytes, config.latent_dim):
        raise ValueError(f"latent sample shape {tuple(latent.shape)} does not match private decoder")
    with torch.no_grad():
        if len(latent.shape) == 4:
            samples = latent.squeeze(0)
            logits = decoder(samples)
            tokens = logits.argmax(dim=-1)
            voted = torch.mode(tokens, dim=0).values.tolist()
            print("[Artist X Private Decoder]")
            print(f"samples: {samples.shape[0]}")
            print(decode_tokens(voted))
            return
        tokens = decoder.generate(latent)[0].tolist()
    print("[Artist X Private Decoder]")
    print(decode_tokens(tokens))


def status(args: argparse.Namespace) -> None:
    _, config = load_private_decoder(args.model, device())
    print("[Artist X Private Decoder]")
    print(f"device: {device()}")
    print(f"model: {args.model}")
    print("contents: decoder only")
    print(f"expected latent shape: [1, {config.max_bytes}, {config.latent_dim}]")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Artist X private decoder CLI")
    subparsers = result.add_subparsers(dest="command", required=True)
    decode_parser = subparsers.add_parser("decode", help="latent JSON을 원문으로 복원")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--model", type=Path, default=PRIVATE_MODEL)
    decode_parser.set_defaults(handler=decode)
    status_parser = subparsers.add_parser("status", help="비공개 모델 상태 확인")
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
