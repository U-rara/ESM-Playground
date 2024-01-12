import accelerate.utils
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from model.layers import CrossAttention, MLP


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    protein_loss = contrastive_loss(similarity)
    text_loss = contrastive_loss(similarity.t())
    return (protein_loss + text_loss) / 2.0


class ProteinTextCLIPConfig(PretrainedConfig):
    model_type = "protein_text_clip"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 projection_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)


class ProteinTextCLIPForPretrain(PreTrainedModel):
    config_class = ProteinTextCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config

        self.protein_model = AutoModel.from_config(protein_model_config)
        self.text_model = AutoModel.from_config(text_model_config)

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

    def forward(self, protein_input_ids, protein_attention_mask, text_input_ids, text_attention_mask):
        protein_embeds = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state.mean(dim=1)
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state.mean(dim=1)
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
        }


class ProtSTConfig(PretrainedConfig):
    model_type = "ProtST"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 mlp_num_layers,
                 fusion_num_heads,
                 projection_dim,
                 fusion_num_layers,
                 fusion_batch_norm,
                 **kwargs):
        super().__init__(**kwargs)

        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.mlp_num_layers = mlp_num_layers
        self.fusion_num_heads = fusion_num_heads
        self.projection_dim = projection_dim
        self.fusion_num_layers = fusion_num_layers
        self.fusion_batch_norm = fusion_batch_norm
        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)
        self.protein_mask_probability = kwargs.pop("protein_mask_probability", 0.15)
        self.text_mask_probability = kwargs.pop("text_mask_probability", 0.15)


class ProtSTForPretrain(PreTrainedModel):
    config_class = ProtSTConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config

        self.protein_model = AutoModel.from_config(protein_model_config)
        self.text_model = AutoModel.from_config(text_model_config)
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

        self.mlm_head = MLP(self.config.projection_dim, self.config.projection_dim,
                            protein_model_config.vocab_size, self.config.mlp_num_layers)
        self.fusion_model = CrossAttention(
            hidden_dim=self.config.projection_dim,
            num_layers=self.config.fusion_num_layers,
            num_heads=self.config.fusion_num_heads,
            batch_norm=self.config.fusion_batch_norm)
        self.mmp_protein_head = MLP(self.config.projection_dim, self.config.projection_dim,
                                    protein_model_config.vocab_size, self.config.mlp_num_layers)
        self.mmp_text_head = MLP(self.config.projection_dim, self.config.projection_dim,
                                 text_model_config.vocab_size, self.config.mlp_num_layers)

    def forward(self,
                protein_input_ids,
                protein_attention_mask,
                text_input_ids,
                text_attention_mask,
                protein_masked_input_ids,
                protein_masked_labels,
                text_masked_input_ids,
                text_masked_labels
                ):
        protein_embeds = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state.mean(dim=1)
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state.mean(dim=1)
        text_embeds = self.text_projection(text_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        protein_embeds = accelerate.utils.gather(protein_embeds)
        text_embeds = accelerate.utils.gather(text_embeds)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_protein = logit_scale * protein_embeds @ text_embeds.t()

        # compute the clip loss
        cl_loss = clip_loss(logits_per_protein)

        # compute outputs
        protein_outputs = self.protein_model(input_ids=protein_masked_input_ids,
                                             attention_mask=protein_attention_mask).last_hidden_state
        protein_outputs = self.protein_projection(protein_outputs)
        text_outputs = self.text_model(input_ids=text_masked_input_ids,
                                       attention_mask=text_attention_mask).last_hidden_state
        text_outputs = self.text_projection(text_outputs)
        fusion_outputs = self.fusion_model(protein_outputs, protein_attention_mask, text_outputs, text_attention_mask)

        # compute the mlm loss
        protein_mlm_logits = self.mlm_head(protein_outputs)
        protein_mlm_loss = cross_entropy(protein_mlm_logits.view(-1, protein_mlm_logits.shape[-1]),
                                         protein_masked_labels.view(-1))

        # compute the mmp loss
        protein_mmp_logits = self.mmp_protein_head(fusion_outputs['protein_output'])
        protein_mmp_loss = cross_entropy(protein_mmp_logits.view(-1, protein_mmp_logits.shape[-1]),
                                         protein_masked_labels.view(-1))
        text_mmp_logits = self.mmp_text_head(fusion_outputs['text_output'])
        text_mmp_loss = cross_entropy(text_mmp_logits.view(-1, text_mmp_logits.shape[-1]),
                                      text_masked_labels.view(-1))

        return {
            "loss": cl_loss + protein_mlm_loss + protein_mmp_loss + text_mmp_loss,
            "cl_loss": cl_loss,
            "protein_mlm_loss": protein_mlm_loss,
            "protein_mmp_loss": protein_mmp_loss,
            "text_mmp_loss": text_mmp_loss
        }
