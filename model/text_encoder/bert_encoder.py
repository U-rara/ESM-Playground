import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BertEncoder(nn.Module):
    def __init__(self, run_config):
        super().__init__()

        self.run_config = run_config
        self.text_model_name = run_config.text_model_name
        self.text_model_fixed = run_config.text_model_fixed

        self.model_config = AutoConfig.from_pretrained(self.text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.model = AutoModel.from_pretrained(self.text_model_name)
        self.model.requires_grad_(not self.text_model_fixed)

    def forward(self, **kwargs):
        return self.model(**kwargs).last_hidden_state

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
