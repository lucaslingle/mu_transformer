from mu_transformer.configs.base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "large"

    # mesh
    config.n_shard_data = 16
    config.n_shard_model = 8

    # batch size, width
    config.tokens_per_global_batch = 262144  # when acc_steps > 1, this is microbatch sz
    config.d_model = 2048

    return config
