from .esm_encoder import ESMEncoder


def build_protein_encoder(run_config):
    protein_model_name = run_config.protein_model_name
    if 'esm' in protein_model_name:
        return ESMEncoder(run_config)

    raise ValueError(f'Unknown protein encoder: {protein_model_name}')
