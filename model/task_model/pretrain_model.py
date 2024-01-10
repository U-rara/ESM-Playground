import accelerate.utils
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    protein_loss = contrastive_loss(similarity)
    text_loss = contrastive_loss(similarity.t())
    return (protein_loss + text_loss) / 2.0


class ProteinTextCLIPConfig(PretrainedConfig):
    model_type = "protein_text_clip"
    is_composition = True

    def __init__(self, protein_model_config, text_model_config, **kwargs):
        super().__init__(**kwargs)
        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config
        self.hidden_sizes = [self.protein_model_config.hidden_size, self.text_model_config.hidden_size]


class ProteinTextCLIPForPretrain(PreTrainedModel):
    config_class = ProteinTextCLIPConfig

    def __init__(self, run_config, protein_encoder, text_encoder):
        super().__init__(ProteinTextCLIPConfig(protein_encoder.model_config, text_encoder.model_config))
        self.run_config = run_config
        self.protein_encoder = protein_encoder
        self.text_encoder = text_encoder
        self.protein_projection = nn.Linear(self.protein_encoder.model_config.hidden_size, run_config.projection_dim)
        self.text_projection = nn.Linear(self.text_encoder.model_config.hidden_size, run_config.projection_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, protein_input_ids, protein_attention_mask, text_input_ids, text_attention_mask):
        protein_embeds = self.protein_encoder(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).mean(dim=1)
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).mean(dim=1)
        text_embeds = self.text_projection(text_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        protein_embeds = accelerate.utils.gather(protein_embeds)
        text_embeds = accelerate.utils.gather(text_embeds)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_protein = logit_scale * protein_embeds @ text_embeds.t()

        # compute the loss
        loss = clip_loss(logits_per_protein)
        return {
            "loss": loss,
            "protein_embeds": protein_embeds,
            "text_embeds": text_embeds,
        }
