import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class ESMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.protein_model_name = config.protein_model_name
        self.protein_model_fixed = config.protein_model_fixed

        self.model_config = AutoConfig.from_pretrained(self.protein_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.protein_model_name)
        self.model = AutoModel.from_pretrained(self.protein_model_name)
        self.model.requires_grad_(not self.protein_model_fixed)

    def forward(self, proteins):
        return self.model(proteins).last_hidden_state

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
