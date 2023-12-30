from mu_transformer.configs.config_base import get_base_config


def get_config():
    config = get_base_config()
    config.model_size = "medium"

    # mesh
    config.n_shard_data = 8
    config.n_shard_model = 4

    # batch size, width
    config.tokens_per_global_batch = 262144  # when acc_steps > 1, this is microbatch sz
    config.d_model = 1024

    return config