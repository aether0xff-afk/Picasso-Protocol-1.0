from pathlib import Path
import sys

import torch
from torch import nn
import warnings

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from picasso_protocol.models import ArtistX, ArtistY  # noqa: E402

warnings.filterwarnings("ignore")

# --- 2. Conceptual Training Block for Y ---
def train_artist_y_conceptual():
    """
    This is a conceptual guide for training Y's decoder.
    It requires a new, creative dataset of (latent_vector, new_text) pairs.
    """
    print("\n--- Conceptual Training Process for ArtistY ---")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. First, load the fully trained ArtistX model
    print("1. Loading the trained ArtistX model...")
    model_X = ArtistX(local_files_only=True)
    # Ensure you have the 'artist_x_best_model.pth' file from training X
    model_X.load_state_dict(torch.load("artist_x_best_model.pth"))
    model_X.to(DEVICE)
    model_X.eval()
    print("✅ ArtistX loaded.")

    # 2. Create an instance of ArtistY and load the encoder from X
    print("2. Preparing ArtistY...")
    model_Y = ArtistY(local_files_only=True).to(DEVICE)
    model_Y.load_encoder_from_x(model_X)
    
    # 3. CRITICAL: Freeze the encoder's weights. We only want to train the decoder.
    for param in model_Y.encoder.parameters():
        param.requires_grad = False
    print("✅ Y's encoder is frozen. Only the decoder will be trained.")

    # 4. Prepare the optimizer for only the decoder's parameters
    optimizer = torch.optim.AdamW(model_Y.decoder.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # 5. Prepare a new "creative" dataset. This is a crucial, non-trivial step.
    #    You need a dataset where the input is a sentence and the target is a *different*,
    #    artistically related sentence.
    #    creative_dataloader = ... 

    # 6. Conceptual Training Loop
    # model_Y.train()
    # for epoch in range(num_epochs):
    #     for batch in creative_dataloader:
    #         original_text = batch['original_text']
    #         creative_target_text = batch['creative_text']
    #         
    #         # Get latent vector from the original text via the frozen encoder
    #         # Train the decoder to produce the creative_target_text from that vector
    #         ...
    
    print("✅ Conceptual training setup for Y is complete.")
    return model_Y


# --- 3. Main Execution Block ---
if __name__ == '__main__':
    print("🎭 'Reinterpreting Artist' ArtistY setup and demonstration.")
    
    # This function prepares a Y model with a trained encoder for demonstration
    Y = train_artist_y_conceptual()
    
    # Since Y's decoder is not actually trained, the output will be random.
    # This demonstrates how the 'reinterpret' function works.
    
    original_text = "a painter who remembers what he drew"
    
    new_text = Y.reinterpret(original_text)

    print("\n--- Y's Reinterpretation Test ---")
    print(f"Original Text: {original_text}")
    print(f"Y's Reinterpreted Text: {new_text}")
    print("\n(Note: Since Y's decoder is untrained, the output is currently random.)")
