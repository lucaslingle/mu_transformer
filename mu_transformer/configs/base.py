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
from ml_collections import config_dict


def get_base_config():
    config = config_dict.ConfigDict()

    # logging/plotting
    config.sow_intermediates = True

    # huggingface tokenizer and dataset settings
    config.hftr_tokenizer_name = "GPT2TokenizerFast"
    config.hftr_tokenizer_shortname = "gpt2"
    config.hfds_identifier = "Skylion007/openwebtext"
    config.hfds_config = None
    config.hfds_datacol = "text"
    config.hfds_buffer_size = 1024  # example buffer length for batched tokenization
    config.sequence_len = 512

    # architecture
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.output_logits_dtype = "bfloat16"  # for bfloat16 grad; is fp32 during eval
    config.use_mup = True  # use mu-parameterization or standard parametrization?
    config.n_layer = 6
    config.d_head = 64
    config.ff_multiple = 4
    config.rotary_base = 10_000  # can be zero to use NoPE instead of RoPE
    config.act_name = "relu"  # any activation defined in jax.nn
    config.act_square = True  # activation squaring
    config.norm_eps = 0.0

    # optimization
    config.tokens_per_global_batch = 64
    config.grad_clip = 1.0  # gradient clip, applied globally using all parameter grads
    config.lr_max = 0.0  # master lr; scaled by mu-parameterization adam, schedule
    config.adam_b1 = 0.9
    config.adam_b2 = 0.95
    config.adam_eps = 1e-8
    config.adam_mu_dtype = "bfloat16"

    # periodic action settings
    config.n_print_step = 10  # print every
    config.n_save_step = 100  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 400  # warmup steps during pretraining
    config.n_pretrain_step = 1_600  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
