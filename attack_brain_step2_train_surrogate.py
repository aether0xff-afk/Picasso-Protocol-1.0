from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attacker_brain_v5_inversion_demo import LatentDataset, train_surrogate
from picasso_protocol.brain_models import BrainConfig


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: train a surrogate private decoder from collected public latents"
    )
    parser.add_argument("--input", type=Path, default=Path("attack_results/brain_attack_pairs.pt"))
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("attack_results/surrogate_private_brain_fragment.pt"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("attack_results/surrogate_private_brain_metrics.json"),
    )
    args = parser.parse_args()

    bundle = torch.load(args.input, map_location="cpu", weights_only=False)
    config = BrainConfig(**bundle["config"])
    train_latents = bundle["train_latents"][: args.pairs]
    train_tokens = bundle["train_tokens"][: args.pairs]
    test_loader = DataLoader(
        LatentDataset(bundle["test_latents"], bundle["test_tokens"]),
        batch_size=args.batch_size,
    )
    current_device = device()
    decoder, metrics = train_surrogate(
        config,
        train_latents,
        train_tokens,
        test_loader,
        current_device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
    metrics = {
        **metrics,
        "pairs": len(train_tokens),
        "public_model": bundle["public_model"],
        "public_dropout": bundle["public_dropout"],
        "public_noise": bundle["public_noise"],
        "private_fragment_loaded": False,
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "surrogate_private_fragment": decoder.state_dict(),
            "metrics": metrics,
        },
        args.output_model,
    )
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "saved_model": str(args.output_model),
        "saved_metrics": str(args.output_metrics),
        **metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
