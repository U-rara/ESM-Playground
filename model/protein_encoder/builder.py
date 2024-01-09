from .esm_encoder import ESMEncoder


def build_protein_encoder(config):
    protein_model_name = config.protein_model_name
    if 'esm' in protein_model_name:
        return ESMEncoder(config)

    raise ValueError(f'Unknown protein encoder: {protein_model_name}')
