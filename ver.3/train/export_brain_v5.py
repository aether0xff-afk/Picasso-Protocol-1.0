from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a PicassoBrain V5 bundle")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).parent / "brain_v5" / "picasso_brain_bundle.pt",
    )
    parser.add_argument("--public-dir", type=Path, default=ROOT / "models" / "public_v5")
    parser.add_argument("--private-dir", type=Path, default=ROOT / "models" / "private_v5")
    args = parser.parse_args()

    checkpoint = torch.load(args.bundle, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state = checkpoint["brain"]
    public_fragment = {
        name.removeprefix("public_fragment."): value
        for name, value in state.items()
        if name.startswith("public_fragment.")
    }
    private_fragment = {
        name.removeprefix("private_fragment."): value
        for name, value in state.items()
        if name.startswith("private_fragment.")
    }
    args.public_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    public_path = args.public_dir / "artist_y_public_brain_fragment.pt"
    private_path = args.private_dir / "artist_x_private_brain_fragment.pt"
    torch.save({"config": config, "public_fragment": public_fragment}, public_path)
    torch.save({"config": config, "private_fragment": private_fragment}, private_path)
    public_checkpoint = torch.load(public_path, map_location="cpu", weights_only=False)
    if set(public_checkpoint) != {"config", "public_fragment"}:
        raise RuntimeError("public V5 export contains private brain data")
    print(f"public={public_path}")
    print(f"private={private_path}")


if __name__ == "__main__":
    main()
