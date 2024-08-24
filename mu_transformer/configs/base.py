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
    config.sow_intermediates = False
    config.sow_param_info = False
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
    config.model_size = "normal"
    config.n_mesh_rows = 128
    config.n_mesh_cols = 1
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.n_layer = 8  # depth, should stay const for mu-transfer
    config.d_model = 1024
    config.d_head = 128
    config.kv_group_size = 8
    config.v_type = "linear"
    config.g_type = "none"
    config.ff_act_name = "relu"  # any activation defined in jax.nn, or "swiglu"
    config.ff_act_square = True  # activation squaring
    config.norm_eps = 1e-6  # rmsnorm epsilon
    config.rotary_base = 10_000  # can be zero to use NoPE/NPE instead of RoPE

    # optimization
    config.tokens_per_global_batch = 2**18  # batch size * sequence len
    config.grad_acc_steps = 1  # steps per parameter update (for micro-batching)
    config.grad_clip = 1.0  # grad clip max l2 norm
    config.lr_base = 0.25  # base learning rate
    config.lr_schedule_name = "linear"
    config.optim_name = "lion"
    config.optim_rule = "abs_mup"  # mup or sp
    config.optim_beta1 = 0.95
    config.optim_beta2 = 0.98
    config.optim_eps = 0.0
    config.wd = 0.00001  # weight decay lambda
    config.use_iwd = True  # use independent weight decay?
    config.use_eps_scaling = False  # multiply adam eps by some factor Theta(1/fan_in)?

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 5000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining
    config.no_checkpoint = False  # skip saving the model

    return config


def compute_ff(ff_act_glu, kv_group_sz, v_type, g_type):
    # todo: verify that this gives same param count approximately for all model variants
    q_contrib = 1
    k_contrib = 1 / kv_group_sz
    v_contrib = (1 / kv_group_sz) * dict(linear=1, depsepconv=1, conv=3)[v_type]
    g_contrib = (1 / kv_group_sz) * dict(none=0, linear=1, depsepconv=1, conv=3)[g_type]
    o_contrib = 1
    a_contrib = q_contrib + k_contrib + v_contrib + g_contrib + o_contrib
    ff_contrib = 12 - a_contrib
    ff_multiple = (0.333 if ff_act_glu else 0.5) * ff_contrib
    return ff_multiple
