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
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.brain_models import (  # noqa: E402
    BrainConfig,
    PicassoBrain,
    PrivateBrainFragmentX,
)
from picasso_protocol.compact_models import PAD, decode_tokens, encode_text  # noqa: E402


def make_wikitext_sentences(
    split: str,
    max_bytes: int,
    limit: int | None,
    seed: int,
) -> list[str]:
    from datasets import load_dataset

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
    if limit is not None:
        sentences = sentences[:limit]
    return sentences


class TextDataset(Dataset):
    def __init__(self, sentences: list[str], config: BrainConfig) -> None:
        self.sentences = sentences
        self.config = config

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(
            encode_text(self.sentences[index], self.config.max_bytes),
            dtype=torch.long,
        )


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = logits.argmax(dim=-1)
    mask = targets != PAD
    return float((predicted[mask] == targets[mask]).float().mean().item())


def exact_match(model: PicassoBrain, loader: DataLoader, device: torch.device) -> float:
    exact = examples = 0
    model.eval()
    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            generated = model.private_fragment.generate(
                model.latent(tokens, shielded=True)
            ).cpu().tolist()
            expected = tokens.cpu().tolist()
            exact += sum(
                decode_tokens(left) == decode_tokens(right)
                for left, right in zip(generated, expected)
            )
            examples += len(expected)
    return exact / max(1, examples)


def evaluate(model: PicassoBrain, loader: DataLoader, device: torch.device) -> dict[str, float]:
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    model.eval()
    total_loss = total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            logits = model(tokens, shielded=True)
            total_loss += float(loss_fn(logits.flatten(0, 1), tokens.flatten()).item())
            total_acc += token_accuracy(logits, tokens)
            batches += 1
    return {
        "loss": total_loss / max(1, batches),
        "token_accuracy": total_acc / max(1, batches),
        "exact_match_ratio": exact_match(model, loader, device),
    }


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def save_checkpoint(
    model: PicassoBrain,
    attacker: PrivateBrainFragmentX,
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(model.config)
    torch.save(
        {
            "config": config,
            "metrics": metrics,
            "brain": model.state_dict(),
            "training_only_attacker": attacker.state_dict(),
        },
        output_dir / "picasso_brain_v6_bundle.pt",
    )
    torch.save(
        {
            "config": config,
            "public_fragment": model.public_fragment.state_dict(),
        },
        output_dir / "artist_y_public_brain_fragment.pt",
    )
    torch.save(
        {
            "config": config,
            "private_fragment": model.private_fragment.state_dict(),
        },
        output_dir / "artist_x_private_brain_fragment.pt",
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PicassoBrain V6 with a training-only attacker")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--attacker-learning-rate", type=float, default=0.003)
    parser.add_argument("--adversarial-weight", type=float, default=0.08)
    parser.add_argument("--attacker-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("ver.3/train/brain_v6"))
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--max-bytes", type=int, default=160)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--private-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--latent-noise", type=float, default=1.0)
    parser.add_argument("--latent-dropout", type=float, default=0.85)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BrainConfig(
        max_bytes=args.max_bytes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        private_dim=args.private_dim,
        dropout=args.dropout,
        latent_noise=args.latent_noise,
        latent_dropout=args.latent_dropout,
        architecture="brain-v6-adversarial",
    )
    train_set = TextDataset(
        make_wikitext_sentences("train", config.max_bytes, args.train_limit, args.seed),
        config,
    )
    validation_set = TextDataset(
        make_wikitext_sentences("validation", config.max_bytes, args.validation_limit, args.seed),
        config,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    model = PicassoBrain(config).to(device)
    attacker = PrivateBrainFragmentX(config).to(device)
    brain_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    attacker_optimizer = torch.optim.AdamW(attacker.parameters(), lr=args.attacker_learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    started = time.time()
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    best_brain_state: dict[str, torch.Tensor] = {}
    best_attacker_state: dict[str, torch.Tensor] = {}

    print(
        f"device={device} train={len(train_set)} validation={len(validation_set)} "
        f"latent_dim={config.latent_dim} private_dim={config.private_dim} "
        f"adv={args.adversarial_weight}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        attacker.train()
        total_private_loss = total_attacker_loss = 0.0
        batches = 0
        for tokens in train_loader:
            tokens = tokens.to(device)

            set_requires_grad(model, False)
            set_requires_grad(attacker, True)
            for _ in range(args.attacker_steps):
                with torch.no_grad():
                    latent = model.latent(tokens, shielded=True)
                logits = attacker(latent)
                attacker_loss = loss_fn(logits.flatten(0, 1), tokens.flatten())
                attacker_optimizer.zero_grad()
                attacker_loss.backward()
                nn.utils.clip_grad_norm_(attacker.parameters(), 1.0)
                attacker_optimizer.step()

            set_requires_grad(model, True)
            set_requires_grad(attacker, False)
            latent = model.latent(tokens, shielded=True)
            private_logits = model.private_fragment(latent)
            private_loss = loss_fn(private_logits.flatten(0, 1), tokens.flatten())
            attacker_logits = attacker(latent)
            adversarial_loss = loss_fn(attacker_logits.flatten(0, 1), tokens.flatten())
            loss = private_loss - args.adversarial_weight * adversarial_loss.clamp(max=8.0)
            brain_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            brain_optimizer.step()
            set_requires_grad(attacker, True)

            total_private_loss += float(private_loss.item())
            total_attacker_loss += float(adversarial_loss.item())
            batches += 1

        metrics = evaluate(model, validation_loader, device)
        metrics["training_private_loss"] = total_private_loss / max(1, batches)
        metrics["training_attacker_loss"] = total_attacker_loss / max(1, batches)
        score = metrics["token_accuracy"] + metrics["exact_match_ratio"]
        if score > best_score:
            best_score = score
            best_metrics = {
                **metrics,
                "epoch": epoch,
                "seconds": round(time.time() - started, 2),
                "adversarial_weight": args.adversarial_weight,
            }
            best_brain_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_attacker_state = {
                name: value.detach().cpu().clone()
                for name, value in attacker.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} acc={metrics['token_accuracy']:.4f} "
                f"exact={metrics['exact_match_ratio']:.4f} "
                f"private_loss={metrics['training_private_loss']:.4f} "
                f"attacker_loss={metrics['training_attacker_loss']:.4f} "
                f"elapsed={time.time() - started:.1f}s"
            )

    model.load_state_dict(best_brain_state)
    attacker.load_state_dict(best_attacker_state)
    save_checkpoint(model, attacker, best_metrics, args.output_dir)
    sample = validation_set[0].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        restored = decode_tokens(
            model.private_fragment.generate(model.latent(sample, shielded=True))[0].tolist()
        )
    print(f"sample={decode_tokens(sample[0].tolist())}")
    print(f"restored={restored}")
    print(f"best={json.dumps(best_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
