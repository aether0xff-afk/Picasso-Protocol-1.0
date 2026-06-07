from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.compact_models import (
    decode_tokens,
    encode_text,
    load_private_decoder,
    load_public_encoder,
)
from picasso_protocol.latent_io import load_latent, save_latent

PUBLIC_MODEL = Path("models/public/artist_y_public_encoder.pt")
PRIVATE_MODEL = Path("models/private/artist_x_private_decoder.pt")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(args: argparse.Namespace) -> None:
    current_device = device()
    encoder, config = load_public_encoder(args.public_model, current_device)
    tokens = torch.tensor([encode_text(args.text, config.max_bytes)], device=current_device)
    with torch.no_grad():
        latent = encoder(tokens)
    save_latent(args.output, latent)
    print(f"완료: {args.output}")
    print(f"latent shape: {list(latent.shape)}")


def decode(args: argparse.Namespace) -> None:
    current_device = device()
    decoder, config = load_private_decoder(args.private_model, current_device)
    latent = load_latent(args.input).to(current_device)
    expected = (1, config.max_bytes, config.latent_dim)
    if tuple(latent.shape) != expected:
        raise ValueError(f"latent shape {tuple(latent.shape)} does not match {expected}")
    with torch.no_grad():
        tokens = decoder.generate(latent)[0].tolist()
    print(decode_tokens(tokens))


def status(args: argparse.Namespace) -> None:
    current_device = device()
    _, public_config = load_public_encoder(args.public_model, current_device)
    print(f"device: {current_device}")
    print(f"public encoder: {args.public_model}")
    print(f"private decoder: {'found' if args.private_model.exists() else 'not found'}")
    print(f"latent shape: [1, {public_config.max_bytes}, {public_config.latent_dim}]")
    print(f"maximum UTF-8 bytes: {public_config.max_bytes - 1}")


def verify_public(args: argparse.Namespace) -> None:
    load_public_encoder(args.public_model, torch.device("cpu"))
    print("안전 검사 통과: 공개 Artist Y 파일에는 encoder만 있습니다.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Picasso Protocol latent CLI")
    subparsers = result.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="공개 Artist Y로 텍스트를 latent JSON으로 변환")
    encode_parser.add_argument("text")
    encode_parser.add_argument("-o", "--output", type=Path, default=Path("picasso_latent.json"))
    encode_parser.add_argument("--public-model", type=Path, default=PUBLIC_MODEL)
    encode_parser.set_defaults(handler=encode)

    decode_parser = subparsers.add_parser("decode", help="비공개 Artist X로 latent JSON을 원문으로 복원")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--private-model", type=Path, default=PRIVATE_MODEL)
    decode_parser.set_defaults(handler=decode)

    status_parser = subparsers.add_parser("status", help="모델과 GPU 상태 확인")
    status_parser.add_argument("--public-model", type=Path, default=PUBLIC_MODEL)
    status_parser.add_argument("--private-model", type=Path, default=PRIVATE_MODEL)
    status_parser.set_defaults(handler=status)

    verify_parser = subparsers.add_parser("verify-public", help="공개 모델에 decoder가 없는지 검사")
    verify_parser.add_argument("--public-model", type=Path, default=PUBLIC_MODEL)
    verify_parser.set_defaults(handler=verify_public)
    return result


def main() -> None:
    try:
        args = parser().parse_args()
        args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
