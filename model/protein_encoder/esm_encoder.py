import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class ESMEncoder(nn.Module):
    def __init__(self, run_config):
        super().__init__()

        self.run_config = run_config
        self.protein_model_name = run_config.protein_model_name
        self.protein_model_fixed = run_config.protein_model_fixed

        self.model_config = AutoConfig.from_pretrained(self.protein_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.protein_model_name)
        self.model = AutoModel.from_pretrained(self.protein_model_name)
        self.model.requires_grad_(not self.protein_model_fixed)

    def forward(self, **kwargs):
        return self.model(**kwargs).last_hidden_state

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
