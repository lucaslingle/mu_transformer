from mu_transformer.configs.base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "small"

    # mesh
    config.n_shard_data = 8
    config.n_shard_model = 1

    # width
    config.d_model = 512

    return config
