import unittest
import unicodedata

import numpy as np
import torch

from picasso_protocol.image_codec import (
    DEFAULT_LATENT_SHAPE,
    combine_jamo,
    decode_data_from_image,
    encode_data_to_image,
)


class ImageCodecTest(unittest.TestCase):
    def test_round_trips_tensor_and_original_length(self) -> None:
        tensor_np = np.linspace(
            -1.0,
            1.0,
            num=DEFAULT_LATENT_SHAPE[0] * DEFAULT_LATENT_SHAPE[1],
            dtype=np.float32,
        ).reshape(DEFAULT_LATENT_SHAPE)
        tensor = torch.from_numpy(tensor_np).unsqueeze(0)

        image = encode_data_to_image(tensor, original_length=17)
        decoded_tensor, original_length = decode_data_from_image(image)

        self.assertEqual(original_length, 17)
        self.assertEqual(tuple(decoded_tensor.shape), (1, *DEFAULT_LATENT_SHAPE))
        torch.testing.assert_close(decoded_tensor, tensor)

    def test_combines_hangul_jamo_with_nfc_normalization(self) -> None:
        decomposed = unicodedata.normalize("NFD", "테스트")

        self.assertEqual(combine_jamo(decomposed), "테스트")


if __name__ == "__main__":
    unittest.main()
