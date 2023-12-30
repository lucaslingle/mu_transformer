from mu_transformer.configs.config_base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "small"

    # mesh
    config.n_shard_data = 8
    config.n_shard_model = 1

    # batch size, width
    config.tokens_per_global_batch = 65536  # when acc_steps > 1, this is microbatch sz
    config.d_model = 768

    return config