import jax.numpy as jnp
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.model_size = "tiny"

    config.param_dtype = jnp.float32
    config.dtype = jnp.bfloat16
    config.sow_intermediates = False

    config.hftr_tokenizer_name = "GPT2TokenizerFast"
    config.hftr_tokenizer_shortname = "gpt2"
    config.hfds_identifier = "skylion007/openwebtext"
    config.hfds_config = None
    config.hfds_datacol = "text"

    config.tokens_per_global_batch = 4096
    config.sequence_len = 512
    config.d_model = 128
    config.n_layer = 24
    config.rotary_base = 10_000
    config.rotary_interp_q = False
    config.rotary_interp_k = False
    config.act_name = "relu"
    config.act_square = False

    config.grad_clip = 1.0
    config.lr_max = 0.0004
    config.wd_lam = 0.01
    config.n_print_step = 10  # print every
    config.n_save_step = 5_000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
