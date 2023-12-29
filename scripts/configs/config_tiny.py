import jax.numpy as jnp
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.model_size = "tiny"

    config.param_dtype = jnp.float32
    config.dtype = jnp.bfloat16
    config.sow_intermediates = False

    config.hftr_tokenizer_name = "GPT2TokenizerFast"  # huggingface transformers tokeniz
    config.hftr_tokenizer_shortname = "gpt2"  # huggingface transformers tokeniz
    config.hfds_identifier = "skylion007/openwebtext"  # huggingface dataset identifier
    config.hfds_config = None  # huggingface dataset config
    config.hfds_datacol = "text"  # huggingface dataset data column

    config.tokens_per_global_batch = 65536
    config.sequence_len = 512
    config.d_model = 128
    config.n_layer = 24
    config.rotary_base = 10_000
    config.rotary_interp_q = False
    config.rotary_interp_k = False
    config.act_name = "relu"  # any activation defined jax.nn
    config.act_square = False  # activation squaring

    config.grad_clip = 1.0  # gradient clip, applied globally using all parameter grads
    config.lr_max = 0.04  # maximum main lr; scaled by mu-parameterization and schedule
    config.adam_b1 = 0.9
    config.adam_b2 = 0.98
    config.adam_eps = 1e-9
    config.wd_lam = 0.01  # weight decay coefficient, is multiplied by each param's lr
    config.n_print_step = 10  # print every
    config.n_save_step = 5_000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
