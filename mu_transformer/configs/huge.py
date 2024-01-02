from mu_transformer.configs.base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "huge"

    # mesh
    config.n_shard_data = 16
    config.n_shard_model = 8

    # width
    config.d_model = 4096

    return config
