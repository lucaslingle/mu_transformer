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
    config.sow_param_info = True

    # huggingface tokenizer and dataset settings
    config.hftr_tokenizer_name = "GPT2TokenizerFast"
    config.hftr_tokenizer_shortname = "gpt2"
    config.hfds_identifier = "EleutherAI/the_pile_deduplicated"
    config.hfds_config = None
    config.hfds_datacol = "text"
    config.hfds_buffer_size = 1024  # example buffer length for batched tokenization
    config.sequence_len = 512
    config.force_download = True  # should be true unless you know what you're doing

    # architecture
    # options for parameterization: sp, mup, spectral
    #     sp = standard parameterization w/ zero-init on query, readout projections.
    #     mup = max update parameterization w/ rel scaling: assumes only d_model changes
    #     spectral = spectral init w/ rel scaling: assumes only d_model changes
    config.parameterization = "mup"
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.output_logits_dtype = "bfloat16"  # for bfloat16 grad; is fp32 during eval
    config.n_layer = 6  # depth, should stay const for mu-transfer
    config.d_base = 256  # base model width for relative scaling rules
    config.d_head = 64  # attn head width
    config.ff_multiple = 4  # mlp hidden width multiple
    config.rotary_base = 10_000  # can be zero to use NoPE/NPE instead of RoPE
    config.act_name = "relu"  # any activation defined in jax.nn
    config.act_square = False  # activation squaring
    config.norm_eps = 1e-8  # rmsnorm epsilon
    config.norm_gains = False  # rmsnorm gains

    # optimization
    config.tokens_per_global_batch = 2**18  # batch size * sequence len
    config.grad_acc_steps = 1  # steps per parameter update (for micro-batching)
    config.grad_clip = 1.0  # grad clip max l2 norm
    config.lr_base = 1.0  # base learning rate
    config.adam_b1 = 0.9  # adam first moment ema rate
    config.adam_b2 = 0.95  # adam second moment ema rate
    config.adam_eps = 1e-8  # adam epsilon
    config.adam_mu_dtype = "bfloat16"  # adam first moment dtype
    config.wd = 0.0  # weight decay

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 1000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 1_000  # warmup steps during pretraining
    config.n_pretrain_step = 10_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining

    return config
