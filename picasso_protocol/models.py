from __future__ import annotations

import torch
from torch import nn
from transformers import BertLMHeadModel, BertModel, BertTokenizer


class ArtistX(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.encoder = BertModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.decoder = BertLMHeadModel.from_pretrained(
            model_name,
            is_decoder=True,
            local_files_only=local_files_only,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        latent_vector = encoder_outputs.last_hidden_state
        return self.decoder(
            inputs_embeds=latent_vector,
            attention_mask=attention_mask,
        ).logits

    def reconstruct(self, text: str) -> str:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True,
            ).to(self.encoder.device)
            logits = self.forward(inputs["input_ids"], inputs["attention_mask"])
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)


class ArtistY(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.encoder = BertModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.decoder = BertLMHeadModel.from_pretrained(
            model_name,
            is_decoder=True,
            local_files_only=local_files_only,
        )

    def load_encoder_from_x(self, x_model: ArtistX) -> None:
        self.encoder.load_state_dict(x_model.encoder.state_dict())
        print("Y has successfully inherited X's encoding ability.")

    def reinterpret(self, text: str, max_new_tokens: int = 30) -> str:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.encoder.device)
            latent_vector = self.encoder(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state
            outputs = self.decoder.generate(
                inputs_embeds=latent_vector,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
