from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_brain_v5_inversion_demo import encode_latents, load_sentences, tensorize
from picasso_protocol.brain_models import load_public_brain_fragment


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: collect chosen plaintext and public brain latents"
    )
    parser.add_argument(
        "--public-model",
        type=Path,
        default=Path("models/public_v6_ultradrop/artist_y_public_brain_fragment.pt"),
    )
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--public-dropout", type=float)
    parser.add_argument("--public-noise", type=float)
    parser.add_argument("--output", type=Path, default=Path("attack_results/brain_attack_pairs.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    current_device = device()
    fragment, config = load_public_brain_fragment(args.public_model, current_device)
    public_dropout = config.latent_dropout if args.public_dropout is None else args.public_dropout
    public_noise = config.latent_noise if args.public_noise is None else args.public_noise

    train_sentences = load_sentences("train", config.max_bytes, args.seed)
    test_sentences = load_sentences("test", config.max_bytes, args.seed + 1)
    if len(train_sentences) < args.train_size:
        raise ValueError(f"only {len(train_sentences)} train sentences are available")
    if len(test_sentences) < args.test_size:
        raise ValueError(f"only {len(test_sentences)} test sentences are available")

    train_sentences = train_sentences[: args.train_size]
    test_sentences = test_sentences[: args.test_size]
    train_tokens = tensorize(train_sentences, config.max_bytes)
    test_tokens = tensorize(test_sentences, config.max_bytes)
    train_latents = encode_latents(
        fragment,
        train_tokens,
        current_device,
        args.batch_size,
        dropout=public_dropout,
        noise=public_noise,
    )
    test_latents = encode_latents(
        fragment,
        test_tokens,
        current_device,
        args.batch_size,
        dropout=public_dropout,
        noise=public_noise,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "public_model": str(args.public_model),
            "public_dropout": public_dropout,
            "public_noise": public_noise,
            "config": asdict(config),
            "train_sentences": train_sentences,
            "test_sentences": test_sentences,
            "train_tokens": train_tokens,
            "test_tokens": test_tokens,
            "train_latents": train_latents,
            "test_latents": test_latents,
        },
        args.output,
    )
    print(json.dumps({
        "saved": str(args.output),
        "device": str(current_device),
        "train_size": len(train_sentences),
        "test_size": len(test_sentences),
        "latent_shape": list(train_latents.shape),
        "public_dropout": public_dropout,
        "public_noise": public_noise,
        "private_fragment_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
