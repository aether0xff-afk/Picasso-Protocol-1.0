from __future__ import annotations

import unicodedata
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

DEFAULT_IMAGE_SIZE = (1280, 1280)
DEFAULT_LATENT_SHAPE = (128, 768)
VISUAL_SIZE = (256, 256)
GRID_BLOCKS = 8
PICASSO_PALETTE = np.array(
    [
        [0x2E, 0x4A, 0x34],
        [0x52, 0x3A, 0x62],
        [0xF2, 0xD3, 0x38],
        [0xC8, 0x2D, 0x31],
        [0xD9, 0xD9, 0xD9],
        [0x34, 0x34, 0x34],
    ],
    dtype=np.uint8,
)


def combine_jamo(text: str) -> str:
    """Normalize decomposed Hangul jamo into composed characters."""
    return unicodedata.normalize("NFC", text)


def encode_data_to_image(
    tensor: torch.Tensor,
    original_length: int,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    """Encode a latent tensor and source token length into an RGBA PNG image."""
    tensor_np = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    payload = original_length.to_bytes(4, byteorder="big") + tensor_np.tobytes()

    capacity = target_size[0] * target_size[1] * 4
    if len(payload) > capacity:
        raise ValueError(
            f"payload requires {len(payload)} bytes, but image holds {capacity} bytes"
        )

    image_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    image_array.flat[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)

    visual = _build_picasso_visual(tensor_np)
    visual_w, visual_h = visual.size
    start_x = (target_size[0] - visual_w) // 2
    start_y = (target_size[1] - visual_h) // 2
    image_array[start_y : start_y + visual_h, start_x : start_x + visual_w, :3] = (
        np.array(visual)
    )
    image_array[start_y : start_y + visual_h, start_x : start_x + visual_w, 3] = 255

    return Image.fromarray(image_array, "RGBA")


def decode_data_from_image(
    image: Image.Image,
    original_shape: Tuple[int, int] = DEFAULT_LATENT_SHAPE,
) -> tuple[torch.Tensor, int]:
    """Decode a latent tensor and source token length from an RGBA image."""
    image_array = np.array(image.convert("RGBA"))
    image_bytes = image_array.tobytes()

    original_length = int.from_bytes(image_bytes[:4], byteorder="big")
    required_bytes = original_shape[0] * original_shape[1] * 4
    tensor_bytes = image_bytes[4 : 4 + required_bytes]
    if len(tensor_bytes) != required_bytes:
        raise ValueError(
            f"image has {len(tensor_bytes)} tensor bytes, expected {required_bytes}"
        )

    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(original_shape)
    return torch.from_numpy(tensor_np.copy()).unsqueeze(0), original_length


def _build_picasso_visual(tensor_np: np.ndarray) -> Image.Image:
    normalized = _normalize_to_uint8(tensor_np)
    vis_img = Image.fromarray(normalized).resize(VISUAL_SIZE, Image.Resampling.BICUBIC)
    vis_array = np.array(vis_img)

    block_size = VISUAL_SIZE[0] // GRID_BLOCKS
    canvas = np.zeros_like(vis_array)
    for row in range(GRID_BLOCKS):
        for col in range(GRID_BLOCKS):
            y0 = row * block_size
            y1 = (row + 1) * block_size
            x0 = col * block_size
            x1 = (col + 1) * block_size
            block = vis_array[y0:y1, x0:x1]
            choice = np.random.randint(4)
            if choice == 1:
                block = np.flipud(block)
            elif choice == 2:
                block = np.fliplr(block)
            elif choice == 3:
                block = np.rot90(block)
            canvas[y0:y1, x0:x1] = block

    palette_indices = np.floor(canvas / 256 * len(PICASSO_PALETTE)).astype(np.int16)
    palette_indices = np.clip(palette_indices, 0, len(PICASSO_PALETTE) - 1)
    final_visual = Image.fromarray(PICASSO_PALETTE[palette_indices])

    draw = ImageDraw.Draw(final_visual)
    for index in range(1, GRID_BLOCKS):
        offset = index * block_size
        draw.line([(offset, 0), (offset, VISUAL_SIZE[1] - 1)], fill="black", width=2)
        draw.line([(0, offset), (VISUAL_SIZE[0] - 1, offset)], fill="black", width=2)

    return final_visual


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    shifted = values - values.min()
    max_value = shifted.max()
    if max_value == 0:
        return np.zeros_like(shifted, dtype=np.uint8)
    return (shifted / max_value * 255).astype(np.uint8)

