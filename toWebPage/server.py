from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picasso_protocol.compact_models import (  # noqa: E402
    decode_tokens,
    encode_text,
    load_private_decoder,
    load_public_encoder,
)
from picasso_protocol.latent_io import latent_to_payload, payload_to_latent  # noqa: E402

PUBLIC_ENCODER_PATH = Path(
    os.environ.get(
        "PICASSO_PUBLIC_MODEL", ROOT / "models" / "public" / "artist_y_public_encoder.pt"
    )
)
PRIVATE_DECODER_PATH = Path(
    os.environ.get(
        "PICASSO_PRIVATE_MODEL", ROOT / "models" / "private" / "artist_x_private_decoder.pt"
    )
)
ENABLE_PRIVATE_DECODE = os.environ.get("PICASSO_ENABLE_PRIVATE_DECODE") == "1"
PUBLIC_NOISE = float(os.environ.get("PICASSO_PUBLIC_NOISE", "0"))
PUBLIC_DROPOUT = float(os.environ.get("PICASSO_PUBLIC_DROPOUT", "0"))

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
public_encoder, public_config = load_public_encoder(PUBLIC_ENCODER_PATH, device)
private_decoder = None
private_config = None
if ENABLE_PRIVATE_DECODE:
    private_decoder, private_config = load_private_decoder(PRIVATE_DECODER_PATH, device)
    if (
        public_config.max_bytes,
        public_config.latent_dim,
    ) != (
        private_config.max_bytes,
        private_config.latent_dim,
    ):
        raise RuntimeError("public encoder and private decoder configurations do not match")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(device),
            "public_model": "Artist Y Encoder",
            "private_decode_enabled": ENABLE_PRIVATE_DECODE,
            "public_noise": PUBLIC_NOISE,
            "public_dropout": PUBLIC_DROPOUT,
            "latent_shape": [1, public_config.max_bytes, public_config.latent_dim],
        }
    )


@app.post("/encode-latent")
def encode_latent():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text:
        return jsonify({"error": "text is required"}), 400
    try:
        tokens = torch.tensor(
            [encode_text(text, public_config.max_bytes)], device=device
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    with torch.no_grad():
        latent = public_encoder(tokens)
        if PUBLIC_DROPOUT:
            if not 0.0 <= PUBLIC_DROPOUT < 1.0:
                return jsonify({"error": "PICASSO_PUBLIC_DROPOUT must be in [0.0, 1.0)"}), 500
            keep_probability = 1.0 - PUBLIC_DROPOUT
            latent = latent * torch.empty_like(latent).bernoulli_(keep_probability) / keep_probability
        if PUBLIC_NOISE:
            latent = latent + torch.randn_like(latent) * PUBLIC_NOISE
    return jsonify({"latent": latent_to_payload(latent)})


@app.post("/decode-latent")
def decode_latent():
    if not ENABLE_PRIVATE_DECODE or private_decoder is None or private_config is None:
        return jsonify({"error": "private Artist X decoder is disabled on this server"}), 403
    payload = request.get_json(silent=True) or {}
    latent_payload = payload.get("latent")
    if not isinstance(latent_payload, dict):
        return jsonify({"error": "latent is required"}), 400
    try:
        latent = payload_to_latent(latent_payload).to(device)
        expected_shape = (1, private_config.max_bytes, private_config.latent_dim)
        if tuple(latent.shape) != expected_shape:
            raise ValueError(f"latent shape {tuple(latent.shape)} does not match {expected_shape}")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    with torch.no_grad():
        tokens = private_decoder.generate(latent)[0].tolist()
    return jsonify({"text": decode_tokens(tokens)})


@app.post("/encode")
@app.post("/decode")
def image_codec_pending():
    return jsonify({"error": "PNG codec is not implemented yet; use latent endpoints"}), 501


if __name__ == "__main__":
    print(f"Picasso Protocol latent API ready on {device}")
    app.run(debug=True, use_reloader=False)
