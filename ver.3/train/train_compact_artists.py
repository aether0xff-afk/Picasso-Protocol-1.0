from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

PAD = 256
BOS = 257
EOS = 258
VOCAB_SIZE = 259


@dataclass
class ModelConfig:
    max_bytes: int = 72
    embedding_dim: int = 32
    hidden_dim: int = 160
    latent_dim: int = 96
    layers: int = 2
    dropout: float = 0.25
    latent_noise: float = 0.04
    latent_dropout: float = 0.0
    architecture: str = "mlp"


def encode_text(text: str, max_bytes: int) -> list[int]:
    payload = list(text.encode("utf-8"))[: max_bytes - 1]
    return payload + [EOS]


def decode_tokens(tokens: list[int]) -> str:
    payload: list[int] = []
    for token in tokens:
        if token == EOS:
            break
        if 0 <= token <= 255:
            payload.append(token)
    return bytes(payload).decode("utf-8", errors="replace")


def make_pairs(seed: int = 7) -> list[tuple[str, str]]:
    random.seed(seed)
    subjects = [
        ("the painter", "the artist"),
        ("the student", "the learner"),
        ("the scientist", "the researcher"),
        ("the traveler", "the visitor"),
        ("the musician", "the performer"),
        ("the writer", "the author"),
        ("the teacher", "the instructor"),
        ("the engineer", "the designer"),
    ]
    verbs = [
        ("observes", "looks at"),
        ("remembers", "recalls"),
        ("creates", "makes"),
        ("protects", "keeps safe"),
        ("studies", "examines"),
        ("explains", "describes"),
        ("finds", "discovers"),
        ("shares", "passes along"),
    ]
    objects = [
        ("a hidden message", "a concealed note"),
        ("the quiet garden", "the peaceful garden"),
        ("an abstract image", "an abstract picture"),
        ("a bright idea", "a vivid idea"),
        ("the old painting", "the aged painting"),
        ("a simple pattern", "a basic pattern"),
        ("the secret phrase", "the private phrase"),
        ("a new method", "a fresh method"),
    ]
    endings = [
        ("carefully", "with care"),
        ("today", "on this day"),
        ("at night", "during the night"),
        ("in silence", "quietly"),
    ]
    pairs: list[tuple[str, str]] = []
    for subject in subjects:
        for verb in verbs:
            for obj in objects:
                for ending in endings:
                    source = f"{subject[0]} {verb[0]} {obj[0]} {ending[0]}"
                    target = f"{subject[1]} {verb[1]} {obj[1]} {ending[1]}"
                    pairs.append((source, target))
    random.shuffle(pairs)
    return pairs


def make_wikitext_pairs(
    split: str, max_bytes: int, limit: int | None, seed: int
) -> list[tuple[str, str]]:
    from datasets import load_dataset

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    sentences: list[str] = []
    for row in raw["text"]:
        text = " ".join(row.strip().split())
        if not text or text.startswith("="):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            encoded = sentence.encode("utf-8")
            if 12 <= len(encoded) <= max_bytes - 1:
                sentences.append(sentence)
    sentences = sorted(set(sentences))
    random.Random(seed).shuffle(sentences)
    if limit is not None:
        sentences = sentences[:limit]
    return [(sentence, sentence) for sentence in sentences]


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], config: ModelConfig) -> None:
        self.pairs = pairs
        self.config = config

    def __len__(self) -> int:
        return len(self.pairs)

    def _tensorize(self, text: str) -> torch.Tensor:
        tokens = encode_text(text, self.config.max_bytes)
        tokens += [PAD] * (self.config.max_bytes - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, public_target = self.pairs[index]
        return self._tensorize(source), self._tensorize(public_target)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
        self.to_latent = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.max_bytes * config.embedding_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.latent_dim),
            nn.Tanh(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).flatten(start_dim=1)
        return self.to_latent(embedded)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.output = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.max_bytes * VOCAB_SIZE),
        )

    def forward(self, latent: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        return self.output(latent).view(-1, self.config.max_bytes, VOCAB_SIZE)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return self.output(latent).view(-1, self.config.max_bytes, VOCAB_SIZE).argmax(dim=-1)


class CompactArtists(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.artist_x = Decoder(config)
        self.artist_y = Decoder(config)

    def latent(self, source: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        latent = self.encoder(source)
        if noisy and self.config.latent_dropout:
            keep_probability = 1.0 - self.config.latent_dropout
            mask = torch.empty_like(latent).bernoulli_(keep_probability)
            latent = latent * mask / keep_probability
        if noisy and self.config.latent_noise:
            latent = latent + torch.randn_like(latent) * self.config.latent_noise
        return latent


class SequenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
        self.network = nn.Sequential(
            nn.Conv1d(config.embedding_dim, config.latent_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(tokens).transpose(1, 2)).transpose(1, 2)


class SequenceDecoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(config.latent_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, VOCAB_SIZE, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        return self.network(latent.transpose(1, 2)).transpose(1, 2)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward(latent, latent).argmax(dim=-1)


class SequenceArtists(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config)
        self.artist_x = SequenceDecoder(config)
        self.artist_y = SequenceDecoder(config)

    def latent(self, source: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        latent = self.encoder(source)
        if noisy and self.config.latent_dropout:
            keep_probability = 1.0 - self.config.latent_dropout
            mask = torch.empty_like(latent).bernoulli_(keep_probability)
            latent = latent * mask / keep_probability
        if noisy and self.config.latent_noise:
            latent = latent + torch.randn_like(latent) * self.config.latent_noise
        return latent


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = logits.argmax(dim=-1)
    mask = targets != PAD
    return float((predicted[mask] == targets[mask]).float().mean().item())


def evaluate(model: CompactArtists, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_x = total_y = total_x_acc = total_y_acc = 0.0
    batches = 0
    with torch.no_grad():
        for source, public_target in loader:
            source, public_target = source.to(device), public_target.to(device)
            latent = model.latent(source)
            logits_x = model.artist_x(latent, source)
            logits_y = model.artist_y(latent, public_target)
            total_x += float(loss_fn(logits_x.flatten(0, 1), source.flatten()).item())
            total_y += float(loss_fn(logits_y.flatten(0, 1), public_target.flatten()).item())
            total_x_acc += accuracy(logits_x, source)
            total_y_acc += accuracy(logits_y, public_target)
            batches += 1
    return {
        "x_loss": total_x / batches,
        "y_loss": total_y / batches,
        "x_token_accuracy": total_x_acc / batches,
        "y_token_accuracy": total_y_acc / batches,
    }


def save_checkpoint(model: CompactArtists, metrics: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"config": asdict(model.config), "metrics": metrics}
    torch.save({**common, "encoder": model.encoder.state_dict(), "decoder": model.artist_x.state_dict()}, output_dir / "private_training_checkpoint.pt")
    torch.save({**common, "model": model.state_dict()}, output_dir / "compact_artists_bundle.pt")
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("compact_checkpoints"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--latent-noise", type=float, default=0.04)
    parser.add_argument("--latent-dropout", type=float, default=0.0)
    parser.add_argument("--y-loss-weight", type=float, default=1.0)
    parser.add_argument("--dataset", choices=["synthetic", "wikitext"], default="synthetic")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--max-bytes", type=int, default=72)
    parser.add_argument("--architecture", choices=["mlp", "sequence"], default="mlp")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(
        max_bytes=args.max_bytes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        layers=args.layers,
        dropout=args.dropout,
        latent_noise=args.latent_noise,
        latent_dropout=args.latent_dropout,
        architecture=args.architecture,
    )
    if args.dataset == "wikitext":
        train_set = PairDataset(
            make_wikitext_pairs("train", config.max_bytes, args.train_limit, args.seed),
            config,
        )
        validation_set = PairDataset(
            make_wikitext_pairs(
                "validation", config.max_bytes, args.validation_limit, args.seed
            ),
            config,
        )
    else:
        dataset = PairDataset(make_pairs(args.seed), config)
        train_size = int(len(dataset) * 0.9)
        train_set, validation_set = random_split(
            dataset,
            [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    model_class = SequenceArtists if config.architecture == "sequence" else CompactArtists
    model = model_class(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    print(f"device={device} train={len(train_set)} validation={len(validation_set)} latent_dim={config.latent_dim}")
    started = time.time()
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for source, public_target in train_loader:
            source, public_target = source.to(device), public_target.to(device)
            latent = model.latent(source, noisy=True)
            logits_x = model.artist_x(latent, source)
            logits_y = model.artist_y(latent, public_target)
            loss = loss_fn(logits_x.flatten(0, 1), source.flatten())
            loss += args.y_loss_weight * loss_fn(
                logits_y.flatten(0, 1), public_target.flatten()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate(model, validation_loader, device)
        score = metrics["x_token_accuracy"] + metrics["y_token_accuracy"]
        if score > best_score:
            best_score = score
            best_metrics = {**metrics, "epoch": epoch, "seconds": round(time.time() - started, 2)}
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} x_acc={metrics['x_token_accuracy']:.3f} y_acc={metrics['y_token_accuracy']:.3f} elapsed={time.time() - started:.1f}s")

    model.load_state_dict(best_state)
    save_checkpoint(model, best_metrics, args.output_dir)
    model.eval()
    sample_dataset = validation_set.dataset if hasattr(validation_set, "dataset") else validation_set
    source_text, expected_y = sample_dataset.pairs[0]
    source = sample_dataset._tensorize(source_text).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = model.latent(source)
        generated_x = decode_tokens(model.artist_x.generate(latent)[0].tolist())
        generated_y = decode_tokens(model.artist_y.generate(latent)[0].tolist())
    print(f"source={source_text}")
    print(f"artist_x={generated_x}")
    print(f"expected_y={expected_y}")
    print(f"artist_y={generated_y}")
    print(f"best={json.dumps(best_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
