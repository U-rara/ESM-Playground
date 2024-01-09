import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.text_model_name = config.text_model_name
        self.text_model_fixed = config.text_model_fixed

        self.model_config = AutoConfig.from_pretrained(self.text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.model = AutoModel.from_pretrained(self.text_model_name)
        self.model.requires_grad_(not self.text_model_fixed)

    def forward(self, texts):
        return self.model(texts).last_hidden_state

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
