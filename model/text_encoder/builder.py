from .bert_encoder import BertEncoder


def build_text_encoder(config):
    text_model_name = config.text_model_name
    if 'BERT' in text_model_name:
        return BertEncoder(config)

    raise ValueError(f'Unknown text encoder: {text_model_name}')
