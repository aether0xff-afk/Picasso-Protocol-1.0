from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from picasso_protocol.image_codec import (  # noqa: E402,F401
    decode_data_from_image,
    encode_data_to_image,
)
