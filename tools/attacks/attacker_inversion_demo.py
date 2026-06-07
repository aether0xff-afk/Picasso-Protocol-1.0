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

from picasso_protocol.compact_models import (
    PAD,
    PrivateArtistXDecoder,
    decode_tokens,
    encode_text,
    load_public_encoder,
)

PUBLIC_MODEL = Path("models/public/artist_y_public_encoder.pt")


def load_sentences(split: str, max_bytes: int, seed: int) -> list[str]:
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    sentences: list[str] = []
    for row in raw["text"]:
        text = " ".join(row.strip().split())
        if not text or text.startswith("="):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            length = len(sentence.encode("utf-8"))
            if 12 <= length <= max_bytes - 1:
                sentences.append(sentence)
    sentences = sorted(set(sentences))
    random.Random(seed).shuffle(sentences)
    return sentences


class LatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, tokens: torch.Tensor) -> None:
        if len(latents.shape) == 4:
            sample_count = latents.shape[1]
            latents = latents.flatten(0, 1)
            tokens = tokens.repeat_interleave(sample_count, dim=0)
        self.latents = latents
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[index], self.tokens[index]


class GroupedLatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, tokens: torch.Tensor) -> None:
        self.latents = latents
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[index], self.tokens[index]


def tensorize(sentences: list[str], max_bytes: int) -> torch.Tensor:
    return torch.tensor([encode_text(sentence, max_bytes) for sentence in sentences])


def encode_latents(
    encoder: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int,
    noise_std: float = 0.0,
    dropout: float = 0.0,
    samples: int = 1,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start : start + batch_size].to(device)
            latent = encoder(batch)
            if samples > 1:
                latent = torch.stack(
                    [
                        apply_latent_dropout(latent, dropout)
                        + torch.randn_like(latent) * noise_std
                        for _ in range(samples)
                    ],
                    dim=1,
                )
            else:
                latent = apply_latent_dropout(latent, dropout)
            if samples <= 1 and noise_std:
                latent = latent + torch.randn_like(latent) * noise_std
            latents.append(latent.cpu())
    return torch.cat(latents)


def apply_latent_dropout(latent: torch.Tensor, probability: float) -> torch.Tensor:
    if not probability:
        return latent
    if not 0.0 <= probability < 1.0:
        raise ValueError("--public-dropout must be in the range [0.0, 1.0)")
    keep_probability = 1.0 - probability
    mask = torch.empty_like(latent).bernoulli_(keep_probability)
    return latent * mask / keep_probability


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predicted = logits.argmax(dim=-1)
    mask = targets != PAD
    correct = int((predicted[mask] == targets[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def evaluate(
    decoder: PrivateArtistXDecoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    decoder.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0.0
    correct = tokens = exact = examples = 0
    with torch.no_grad():
        for latent, target in loader:
            if len(latent.shape) == 4:
                batch_size, sample_count = latent.shape[:2]
                flat_latent = latent.flatten(0, 1).to(device)
                flat_target = target.repeat_interleave(sample_count, dim=0).to(device)
                logits = decoder(flat_latent)
                total_loss += float(loss_fn(logits.flatten(0, 1), flat_target.flatten()).item())
                batch_correct, batch_tokens = token_accuracy(logits, flat_target)
                correct += batch_correct
                tokens += batch_tokens
                generated = decoder.generate(flat_latent).view(batch_size, sample_count, -1)
                voted = torch.mode(generated, dim=1).values.cpu().tolist()
                expected = target.cpu().tolist()
                exact += sum(
                    decode_tokens(left) == decode_tokens(right)
                    for left, right in zip(voted, expected)
                )
                examples += batch_size
                continue
            latent, target = latent.to(device), target.to(device)
            logits = decoder(latent)
            total_loss += float(loss_fn(logits.flatten(0, 1), target.flatten()).item())
            batch_correct, batch_tokens = token_accuracy(logits, target)
            correct += batch_correct
            tokens += batch_tokens
            predicted = decoder.generate(latent).cpu().tolist()
            expected = target.cpu().tolist()
            exact += sum(
                decode_tokens(left) == decode_tokens(right)
                for left, right in zip(predicted, expected)
            )
            examples += len(target)
    return {
        "test_loss": total_loss / max(1, len(loader)),
        "token_accuracy": correct / max(1, tokens),
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
) -> tuple[PrivateArtistXDecoder, dict[str, float]]:
    decoder = PrivateArtistXDecoder(config).to(device)
    train_loader = DataLoader(
        LatentDataset(train_latents, train_tokens),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    started = time.time()
    final_train_loss = 0.0
    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0.0
        for latent, target in train_loader:
            latent, target = latent.to(device), target.to(device)
            logits = decoder(latent)
            loss = loss_fn(logits.flatten(0, 1), target.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        final_train_loss = total_loss / max(1, len(train_loader))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch={epoch:03d} train_loss={final_train_loss:.6f}")
    metrics = evaluate(decoder, test_loader, device)
    metrics["train_loss"] = final_train_loss
    metrics["seconds"] = round(time.time() - started, 2)
    return decoder, metrics


def demonstrate_recovery(
    decoder: PrivateArtistXDecoder,
    encoder: nn.Module,
    config,
    messages: list[str],
    device: torch.device,
    noise_std: float,
    dropout: float,
    samples: int,
) -> list[dict[str, str]]:
    tokens = tensorize(messages, config.max_bytes)
    latents = encode_latents(
        encoder,
        tokens,
        device,
        batch_size=64,
        noise_std=noise_std,
        dropout=dropout,
        samples=samples,
    ).to(device)
    decoder.eval()
    with torch.no_grad():
        if len(latents.shape) == 4:
            batch_size, sample_count = latents.shape[:2]
            generated = decoder.generate(latents.flatten(0, 1)).view(batch_size, sample_count, -1)
            recovered = torch.mode(generated, dim=1).values.cpu().tolist()
        else:
            recovered = decoder.generate(latents).cpu().tolist()
    return [
        {"plaintext": source, "surrogate_output": decode_tokens(output)}
        for source, output in zip(messages, recovered)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chosen-plaintext decoder inversion demo using only the public Artist Y encoder"
    )
    parser.add_argument("--public-model", type=Path, default=PUBLIC_MODEL)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("attack_results/decoder_inversion_results.json"))
    parser.add_argument("--public-noise", type=float, default=0.0)
    parser.add_argument("--public-dropout", type=float, default=0.0)
    parser.add_argument("--public-samples", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, config = load_public_encoder(args.public_model, device)
    print("Decoder inversion attack demo")
    print("Attacker inputs: public Artist Y encoder + chosen plaintext queries")
    print("Attacker does not load the private Artist X decoder.")
    print(
        f"device={device} public_model={args.public_model} "
        f"public_noise={args.public_noise} public_dropout={args.public_dropout} "
        f"public_samples={args.public_samples}"
    )

    train_sentences = load_sentences("train", config.max_bytes, args.seed)
    test_sentences = load_sentences("test", config.max_bytes, args.seed + 1)[: args.test_size]
    maximum = max(args.sizes)
    if len(train_sentences) < maximum:
        raise ValueError(f"only {len(train_sentences)} training sentences are available")
    chosen_sentences = train_sentences[:maximum]
    train_tokens = tensorize(chosen_sentences, config.max_bytes)
    test_tokens = tensorize(test_sentences, config.max_bytes)
    train_latents = encode_latents(
        encoder,
        train_tokens,
        device,
        args.batch_size,
        noise_std=args.public_noise,
        dropout=args.public_dropout,
        samples=args.public_samples,
    )
    test_latents = encode_latents(
        encoder,
        test_tokens,
        device,
        args.batch_size,
        noise_std=args.public_noise,
        dropout=args.public_dropout,
        samples=args.public_samples,
    )
    test_dataset = (
        GroupedLatentDataset(test_latents, test_tokens)
        if args.public_samples > 1
        else LatentDataset(test_latents, test_tokens)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    results = []
    best_decoder = None
    for size in args.sizes:
        print()
        print(f"Training surrogate decoder with {size} chosen plaintext-latent pairs")
        decoder, metrics = train_surrogate(
            config,
            train_latents[:size],
            train_tokens[:size],
            test_loader,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        row = {"pairs": size, **metrics}
        results.append(row)
        best_decoder = decoder
        print(
            "  result "
            f"token_accuracy={metrics['token_accuracy']:.4f} "
            f"exact_match_ratio={metrics['exact_match_ratio']:.4f} "
            f"test_loss={metrics['test_loss']:.6f}"
        )

    demo_messages = [
        "The secret code is apple.",
        "Latent vectors should be encrypted.",
        "The painter remembers a hidden message.",
    ]
    recoveries = demonstrate_recovery(
        best_decoder,
        encoder,
        config,
        demo_messages,
        device,
        noise_std=args.public_noise,
        dropout=args.public_dropout,
        samples=args.public_samples,
    )
    print()
    print("Recovered messages with the attacker-trained surrogate decoder")
    for row in recoveries:
        print(f"  plaintext: {row['plaintext']}")
        print(f"  recovered: {row['surrogate_output']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "attack": "chosen_plaintext_decoder_inversion",
        "public_model": str(args.public_model),
        "public_noise": args.public_noise,
        "public_dropout": args.public_dropout,
        "public_samples": args.public_samples,
        "private_decoder_loaded": False,
        "config": asdict(config),
        "results": results,
        "recoveries": recoveries,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
