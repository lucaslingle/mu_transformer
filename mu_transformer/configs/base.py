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

    # logging/plotting
    config.sow_intermediates = True

    # huggingface tokenizer and dataset settings
    config.hftr_tokenizer_name = "GPT2TokenizerFast"
    config.hftr_tokenizer_shortname = "gpt2"
    config.hfds_identifier = "skylion007/openwebtext"
    config.hfds_config = None
    config.hfds_datacol = "text"
    config.hfds_stream_data = True  # require large local download if False
    config.sequence_len = 512

    # architecture
    config.param_dtype = jnp.float32  # master copy of weights in fp32
    config.dtype = jnp.bfloat16  # weights and activations are in bfloat16 on fwd/bwd
    config.output_logits_dtype = jnp.bfloat16  # for bfloat16 grad; is fp32 during eval
    config.n_layer = 12
    config.d_head = 256
    config.ff_multiple = 4
    config.rotary_base = 10_000
    config.act_name = "relu"  # any activation defined in jax.nn
    config.act_square = False  # activation squaring

    # optimization
    config.tokens_per_global_batch = 2**18
    config.grad_clip = 1.0  # gradient clip, applied globally using all parameter grads
    config.lr_max = 10.0  # master lr; scaled by mu-parameterization adam, schedule
    config.adam_b1 = 0.9
    config.adam_b2 = 0.98
    config.adam_eps = 1e-9
    config.adam_mu_dtype = jnp.bfloat16

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 1_000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 1_000  # warmup steps during pretraining
    config.n_pretrain_step = 10_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
