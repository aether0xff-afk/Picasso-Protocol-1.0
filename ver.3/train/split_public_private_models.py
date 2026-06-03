from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).parent / "wikitext_checkpoints_final" / "compact_artists_bundle.pt",
    )
    parser.add_argument("--public-dir", type=Path, default=ROOT / "models" / "public")
    parser.add_argument("--private-dir", type=Path, default=ROOT / "models" / "private")
    args = parser.parse_args()

    checkpoint = torch.load(args.bundle, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state = checkpoint["model"]
    encoder = {
        name.removeprefix("encoder."): value
        for name, value in state.items()
        if name.startswith("encoder.")
    }
    decoder = {
        name.removeprefix("artist_x."): value
        for name, value in state.items()
        if name.startswith("artist_x.")
    }
    args.public_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    public_path = args.public_dir / "artist_y_public_encoder.pt"
    private_path = args.private_dir / "artist_x_private_decoder.pt"
    torch.save({"config": config, "encoder": encoder}, public_path)
    torch.save({"config": config, "decoder": decoder}, private_path)
    public_checkpoint = torch.load(
        public_path, map_location="cpu", weights_only=False
    )
    if set(public_checkpoint) != {"config", "encoder"}:
        raise RuntimeError("public export contains private model data")
    print(f"public={public_path}")
    print(f"private={private_path}")


if __name__ == "__main__":
    main()
