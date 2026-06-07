from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_brain_v5_inversion_demo import LatentDataset, evaluate
from picasso_protocol.brain_models import BrainConfig, PrivateBrainFragmentX
from picasso_protocol.compact_models import decode_tokens


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: evaluate a trained surrogate decoder on held-out public latents"
    )
    parser.add_argument("--pairs-file", type=Path, default=Path("attack_results/brain_attack_pairs.pt"))
    parser.add_argument(
        "--surrogate-model",
        type=Path,
        default=Path("attack_results/surrogate_private_brain_fragment.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    pairs = torch.load(args.pairs_file, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.surrogate_model, map_location="cpu", weights_only=False)
    config = BrainConfig(**checkpoint["config"])
    current_device = device()
    decoder = PrivateBrainFragmentX(config).to(current_device)
    decoder.load_state_dict(checkpoint["surrogate_private_fragment"])
    loader = DataLoader(
        LatentDataset(pairs["test_latents"], pairs["test_tokens"]),
        batch_size=args.batch_size,
    )
    metrics = evaluate(decoder, loader, current_device)
    print(json.dumps(metrics, indent=2))

    decoder.eval()
    with torch.no_grad():
        latents = pairs["test_latents"][: args.samples].to(current_device)
        predicted = decoder.generate(latents).cpu().tolist()
    for index, tokens in enumerate(predicted):
        expected = pairs["test_sentences"][index]
        restored = decode_tokens(tokens)
        print()
        print(f"sample {index + 1}")
        print(f"expected: {expected}")
        print(f"surrogate: {restored}")


if __name__ == "__main__":
    main()
