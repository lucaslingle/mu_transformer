#!/bin/bash

GROUP_NAME="llama";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [l|h]"
  echo "options:"
  echo "l     learning rate float."
  echo "h     Print this help."
  echo
}

while getopts "l:h" option; do
  case $option in
    l)
      LR=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac;
done;

~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
  --experiment_group="$GROUP_NAME" \
  --config="mu_transformer/configs/base.py" \
  --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
  --mode="train" \
  --rng_seed=0 \
  --rng_fold=True \
  --wb_enabled=True \
  --config.is_sweep=True \
  --config.n_mesh_rows=64 \
  --config.n_mesh_cols=4 \
  --config.hftr_tokenizer_name="LlamaTokenizerFast" \
  --config.hftr_tokenizer_instance="hf-internal-testing/llama-tokenizer" \
  --config.hfds_identifier="togethercomputer/RedPajama-Data-1T" \
  --config.hfds_config="" \
  --config.hfds_datacol="text" \
  --config.force_download=False \
  --config.n_ds_shard=32 \
  --config.dtype="bfloat16" \
  --config.n_layer=32 \
  --config.d_model=4096 \
  --config.d_head=128 \
  --config.ff_multiple=2.67 \
  --config.q_init="vs" \
  --config.r_init="vs" \
  --config.u_init="sp" \
  --config.qk_scale=0.08838834764831845 \
  --config.rotary_base=10000 \
  --config.act_name="swiglu" \
  --config.norm_eps=0.00001 \
  --config.norm_gains=True \
  --config.norm_gains_type="vector" \
  --config.proj_biases=False \
  --config.tokens_per_global_batch=4194304 \
  --config.lr_base="$LR" \
  --config.lr_schedule_name="cosine" \
  --config.optim_name="adam" \
  --config.optim_rule="sp" \
  --config.optim_beta1=0.9 \
  --config.optim_beta2=0.95 \
  --config.optim_eps=0.00000001 \
  --config.wd=0.1 \
  --config.use_iwd=False \
  --config.use_eps_scaling=False \
  --config.n_pretrain_step=250000 \
  --config.n_warmup_step=2500;
