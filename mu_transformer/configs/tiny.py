from mu_transformer.configs.base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "tiny"

    # mesh
    config.n_shard_data = 8
    config.n_shard_model = 1

    # batch size, width
    config.d_model = 128

    return config
