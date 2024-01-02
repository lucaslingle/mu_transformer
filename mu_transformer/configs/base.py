# Copyright 2024 Lucas Dax Lingle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax.numpy as jnp
from ml_collections import config_dict


def get_base_config():
    config = config_dict.ConfigDict()

    # basics
    config.param_dtype = jnp.float32
    config.dtype = jnp.bfloat16
    config.sow_intermediates = True

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
    config.act_name = "gelu"  # any activation defined jax.nn
    config.act_square = False  # activation squaring

    # optimization
    config.tokens_per_global_batch = 262144
    config.grad_clip = 1.0  # gradient clip, applied globally using all parameter grads
    config.lr_max = 10.0  # master lr; scaled by mu-parameterization adam, schedule
    config.optim_b1 = 0.9
    config.optim_b2 = 0.98
    config.optim_eps = 1e-9  # used by adam only
    config.wd_lam = 0.0  # weight decay coeff, multiplied by master lr * schedule

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 5_000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
