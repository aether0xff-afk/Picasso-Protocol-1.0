from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import torch

from picasso_protocol.compact_models import (
    PrivateArtistXDecoder,
    decode_tokens,
    load_private_decoder,
)
from picasso_protocol.latent_io import load_latent

PUBLIC_MODEL = Path("models/public/artist_y_public_encoder.pt")


def heading(text: str) -> None:
    print()
    print("=" * 68)
    print(text)
    print("=" * 68)


def inspect_latent(path: Path, known_text: str | None) -> None:
    heading("ATTACK 1: Inspect the captured latent JSON")
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    print(f"captured file: {path}")
    print(f"format: {payload.get('format')}")
    print(f"shape: {payload.get('shape')}")
    print(f"file characters: {len(raw)}")
    print(f"base64 preview: {str(payload.get('data'))[:96]}...")
    if known_text:
        visible = known_text.lower() in raw.lower()
        print(f"plaintext search for {known_text!r}: {'FOUND' if visible else 'not found'}")


def try_public_as_private(path: Path) -> None:
    heading("ATTACK 2: Try to use the public Artist Y file as a private decoder")
    try:
        load_private_decoder(PUBLIC_MODEL, torch.device("cpu"))
    except ValueError as error:
        print("result: BLOCKED")
        print(f"reason: {error}")
        return
    raise RuntimeError("public model unexpectedly loaded as a private decoder")


def try_public_server_decode(path: Path, server: str) -> None:
    heading("ATTACK 3: Ask the public server to decode the captured latent")
    latent = json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(
        f"{server.rstrip('/')}/decode-latent",
        data=json.dumps({"latent": latent}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            print(f"result: server unexpectedly returned HTTP {response.status}")
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"result: BLOCKED with HTTP {error.code}")
        print(error.read().decode("utf-8"))
    except urllib.error.URLError as error:
        print(f"result: server unavailable ({error.reason})")


def try_random_decoder(path: Path, public_model: Path, seed: int) -> None:
    heading("ATTACK 4: Guess a decoder structure but use random weights")
    checkpoint = torch.load(public_model, map_location="cpu", weights_only=False)
    from picasso_protocol.compact_models import ModelConfig

    config = ModelConfig(**checkpoint["config"])
    torch.manual_seed(seed)
    guessed_decoder = PrivateArtistXDecoder(config).eval()
    latent = load_latent(path)
    with torch.no_grad():
        guessed = decode_tokens(guessed_decoder.generate(latent)[0].tolist())
    print("attacker output preview:")
    print(repr(guessed[:160]))
    print("result: random decoder does not restore a readable plaintext")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Picasso Protocol attacker demonstration")
    parser.add_argument("latent", type=Path)
    parser.add_argument("--known-text", help="영상에서 검색 실패를 보여줄 선택적 원문")
    parser.add_argument("--server", default="http://127.0.0.1:5000")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("Picasso Protocol attacker demo")
    print("This is a reproducible prototype demonstration, not a proof of cryptographic security.")
    inspect_latent(args.latent, args.known_text)
    try_public_as_private(args.latent)
    try_public_server_decode(args.latent, args.server)
    try_random_decoder(args.latent, PUBLIC_MODEL, args.seed)


if __name__ == "__main__":
    main()
