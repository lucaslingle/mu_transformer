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
    config.is_sweep = False

    # huggingface tokenizer and dataset settings
    config.hftr_tokenizer_name = "T5TokenizerFast"
    config.hftr_tokenizer_instance = "t5-base"
    config.hfds_identifier = "c4"
    config.hfds_config = "en"
    config.hfds_datacol = "text"
    config.hfds_buffer_size = 512  # example buffer length for batched tokenization
    config.sequence_len = 256
    config.force_download = True  # should be true unless you know what you're doing
    config.n_ds_shard = 0  # 0 = shard by host; less < n_host = subshard existing shards

    # architecture
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.n_layer = 24  # depth, should stay const for mu-transfer
    config.d_base = 128
    config.d_head = 128
    config.qkv_sepconv = False  # depthwise separable convolutions on qkv, same as primer?
    config.qk_norm = False  # normalize queries and keys using rmsnorm?
    config.qk_kernel = "softmax"  # softmax or sqrelu
    config.v_gating = False  # gate values
    config.v_act_name = None  # value activation: any activation defined in jax.nn, or None
    config.v_act_square = False # value activation squaring
    config.rotary_base = 10_000
    config.act_name = "relu"  # any activation defined in jax.nn
    config.act_square = True  # activation squaring
    config.norm_eps = 1e-6  # layernorm epsilon
    config.norm_trainable = True  # layernorm trainable

    # optimization
    config.tokens_per_global_batch = 2**18  # batch size * sequence len
    config.grad_acc_steps = 1  # steps per parameter update (for micro-batching)
    config.grad_clip = 1.0  # grad clip max l2 norm
    config.lr_base = 0.002  # base learning rate
    config.lr_schedule_name = "linear"
    config.optim_name = "adamw"
    config.optim_rule = "sp"
    config.optim_beta1 = 0.9
    config.optim_beta2 = 0.98
    config.optim_eps = 10**-9
    config.wd = 0.1  # weight decay

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 5000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining
    config.no_checkpoint = False  # skip saving the model

    return config
