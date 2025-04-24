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

    config.n_mesh_rows = 128
    config.n_mesh_cols = 1

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
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.n_layer = 24  # depth, should stay const for mu-transfer
    config.d_model = 1024  # current model width
    config.d_base = 128  # base model width for relative scaling rules
    config.d_head = 128
    config.ff_multiple = 4.0  # mlp width multiple
    config.e_norm = False  # normalize the embeddings using rmsnorm?
    config.q_init = "vs"  # query projection init: vs, zero
    config.r_init = "vs"  # residual projection init: vs, zero
    config.u_init = "mup"  # unembedding projection init: mup, sp, zero
    config.qk_scale = 1 / 128
    config.qk_norm = False  # normalize queries and keys using rmsnorm?
    config.kv_mqa = False
    config.rotary_base = 10_000  # can be zero to use NoPE/NPE instead of RoPE
    config.attn_act_name = "softmax"
    config.ff_act_name = "relu"  # any activation in jax.nn, or "swiglu", or "sqrelu".
    config.norm_eps = 1e-5  # rmsnorm epsilon
    config.norm_gains = False  # rmsnorm gains
    config.norm_gains_type = "vector"  # vector or scalar
    config.proj_biases = False  # projections with bias

    # optimization
    config.tokens_per_global_batch = 2**18  # batch size * sequence len
    config.grad_acc_steps = 1  # steps per parameter update (for micro-batching)
    config.grad_clip = 1.0  # grad clip max l2 norm
    config.lr_base = 1.0  # base learning rate
    config.lr_schedule_name = "linear"
    config.lr_schedule_end_frac = 0.0001
    config.optim_name = "adamw"
    config.optim_rule = "mup"  # abs_mup, mup, or sp
    config.optim_beta1 = 0.9
    config.optim_beta2 = 0.95
    config.optim_eps = 10**-8
    config.wd = 0.0  # weight decay lambda
    config.use_iwd = False  # use independent weight decay?
    config.use_eps_scaling = False  # multiply adam eps by some factor Theta(1/fan_in)?

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 5000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining
    config.no_checkpoint = False  # skip saving the model

    # sampling settings
    config.sampling_method = "nucleus"
    config.sampling_nucleus = 0.8
    config.sampling_topk = 20  # unused w sample_method=nucleus, and value is untuned
    config.sampling_temp = 0.1  # unused w sample_method=nucleus, and value is untuned
    config.sampling_prompt_len = 128
    config.sampling_max_len = 1024

    return config


def get_config():
    return get_base_config()
