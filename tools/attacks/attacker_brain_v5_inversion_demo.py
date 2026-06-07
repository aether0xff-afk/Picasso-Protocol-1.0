from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.brain_models import (
    PrivateBrainFragmentX,
    apply_brain_shield,
    load_public_brain_fragment,
)
from picasso_protocol.compact_models import PAD, decode_tokens, encode_text

PUBLIC_MODEL = Path("models/public_v5/artist_y_public_brain_fragment.pt")


def load_sentences(split: str, max_bytes: int, seed: int) -> list[str]:
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    sentences: list[str] = []
    for row in raw["text"]:
        text = " ".join(row.strip().split())
        if not text or text.startswith("="):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if 12 <= len(sentence.encode("utf-8")) <= max_bytes - 1:
                sentences.append(sentence)
    sentences = sorted(set(sentences))
    random.Random(seed).shuffle(sentences)
    return sentences


def tensorize(sentences: list[str], max_bytes: int) -> torch.Tensor:
    return torch.tensor([encode_text(sentence, max_bytes) for sentence in sentences])


class LatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, tokens: torch.Tensor) -> None:
        self.latents = latents
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[index], self.tokens[index]


def encode_latents(
    fragment: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int,
    dropout: float,
    noise: float,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    fragment.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start : start + batch_size].to(device)
            latent = apply_brain_shield(fragment(batch), dropout=dropout, noise=noise)
            latents.append(latent.cpu())
    return torch.cat(latents)


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predicted = logits.argmax(dim=-1)
    mask = targets != PAD
    return int((predicted[mask] == targets[mask]).sum().item()), int(mask.sum().item())


def evaluate(
    decoder: PrivateBrainFragmentX,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    decoder.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0.0
    correct = total = exact = examples = 0
    with torch.no_grad():
        for latent, target in loader:
            latent, target = latent.to(device), target.to(device)
            logits = decoder(latent)
            total_loss += float(loss_fn(logits.flatten(0, 1), target.flatten()).item())
            batch_correct, batch_total = token_accuracy(logits, target)
            correct += batch_correct
            total += batch_total
            predicted = decoder.generate(latent).cpu().tolist()
            expected = target.cpu().tolist()
            exact += sum(
                decode_tokens(left) == decode_tokens(right)
                for left, right in zip(predicted, expected)
            )
            examples += len(expected)
    return {
        "test_loss": total_loss / max(1, len(loader)),
        "token_accuracy": correct / max(1, total),
        "exact_match_ratio": exact / max(1, examples),
    }


def train_surrogate(
    config,
    train_latents: torch.Tensor,
    train_tokens: torch.Tensor,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[PrivateBrainFragmentX, dict[str, float]]:
    decoder = PrivateBrainFragmentX(config).to(device)
    loader = DataLoader(LatentDataset(train_latents, train_tokens), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    started = time.time()
    final_train_loss = 0.0
    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0.0
        for latent, target in loader:
            latent, target = latent.to(device), target.to(device)
            logits = decoder(latent)
            loss = loss_fn(logits.flatten(0, 1), target.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        final_train_loss = total_loss / max(1, len(loader))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch={epoch:03d} train_loss={final_train_loss:.6f}")
    metrics = evaluate(decoder, test_loader, device)
    metrics["train_loss"] = final_train_loss
    metrics["seconds"] = round(time.time() - started, 2)
    return decoder, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 brain fragment decoder inversion demo")
    parser.add_argument("--public-model", type=Path, default=PUBLIC_MODEL)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--public-dropout", type=float)
    parser.add_argument("--public-noise", type=float)
    parser.add_argument("--output", type=Path, default=Path("attack_results/brain_v5_inversion_results.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fragment, config = load_public_brain_fragment(args.public_model, device)
    public_dropout = config.latent_dropout if args.public_dropout is None else args.public_dropout
    public_noise = config.latent_noise if args.public_noise is None else args.public_noise
    print("V5 brain fragment inversion demo")
    print("Attacker inputs: public brain fragment + chosen plaintext queries")
    print("Attacker does not load the private brain fragment.")
    print(
        f"device={device} public_model={args.public_model} "
        f"public_dropout={public_dropout} public_noise={public_noise}"
    )

    train_sentences = load_sentences("train", config.max_bytes, args.seed)
    test_sentences = load_sentences("test", config.max_bytes, args.seed + 1)[: args.test_size]
    maximum = max(args.sizes)
    if len(train_sentences) < maximum:
        raise ValueError(f"only {len(train_sentences)} training sentences are available")
    train_tokens = tensorize(train_sentences[:maximum], config.max_bytes)
    test_tokens = tensorize(test_sentences, config.max_bytes)
    train_latents = encode_latents(
        fragment,
        train_tokens,
        device,
        args.batch_size,
        dropout=public_dropout,
        noise=public_noise,
    )
    test_latents = encode_latents(
        fragment,
        test_tokens,
        device,
        args.batch_size,
        dropout=public_dropout,
        noise=public_noise,
    )
    test_loader = DataLoader(LatentDataset(test_latents, test_tokens), batch_size=args.batch_size)

    results = []
    for size in args.sizes:
        print()
        print(f"Training surrogate private brain fragment with {size} chosen pairs")
        _, metrics = train_surrogate(
            config,
            train_latents[:size],
            train_tokens[:size],
            test_loader,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        results.append({"pairs": size, **metrics})
        print(
            "  result "
            f"token_accuracy={metrics['token_accuracy']:.4f} "
            f"exact_match_ratio={metrics['exact_match_ratio']:.4f} "
            f"test_loss={metrics['test_loss']:.6f}"
        )

    report = {
        "attack": "brain_v5_chosen_plaintext_inversion",
        "public_model": str(args.public_model),
        "public_dropout": public_dropout,
        "public_noise": public_noise,
        "private_fragment_loaded": False,
        "config": asdict(config),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
