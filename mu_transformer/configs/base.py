import jax.numpy as jnp
from ml_collections import config_dict


def get_base_config():
    config = config_dict.ConfigDict()

    # basics
    config.param_dtype = jnp.float32
    config.dtype = jnp.bfloat16
    config.sow_intermediates = False

    # huggingface tokenizer and dataset settings
    config.hftr_tokenizer_name = "GPT2TokenizerFast"
    config.hftr_tokenizer_shortname = "gpt2"
    config.hfds_identifier = "skylion007/openwebtext"
    config.hfds_config = None
    config.hfds_datacol = "text"
    config.sequence_len = 512

    # architecture
    config.n_layer = 24
    config.rotary_base = 10_000
    config.rotary_interp_q = False
    config.rotary_interp_k = False
    config.act_name = "relu"  # any activation defined jax.nn
    config.act_square = False  # activation squaring

    # optimization
    config.tokens_per_global_batch = 262144  # when acc_steps > 1, this is microbatch sz
    config.grad_clip = 1.0  # gradient clip, applied globally using all parameter grads
    config.lr_max = 0.3  # maximum main lr; scaled by mu-parameterization adam, schedule
    config.adam_b1 = 0.9
    config.adam_b2 = 0.98
    config.adam_eps = 1e-9
    config.wd_lam = 0.0  # weight decay coefficient, is multiplied by each param's lr

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 5_000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
