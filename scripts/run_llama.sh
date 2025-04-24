#!/bin/bash

GROUP_NAME="llama";
Help() {
  echo "Syntax: run_llama.sh [l|h]"
  echo ""
  echo "Trains a llama-style model with 4 billion parameters for 100 billion tokens."
  echo "Intended for larger-scale ablation studies."
  echo "Diffs vs llama: mha; d_ff = 4d_model; wd on embs & gains; lr decay to zero."
  echo ""
  echo "Options:"
  echo "l     use learning rate lr = 2^-l for the given l."
  echo "p     use parametric rmsnorm."
  echo "h     Print this Help."
  echo
}

while getopts "l:p:h" option; do
  case $option in
    l)
      LR_IDX=$OPTARG;;
    p)
      NORM_PARAMS=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

LR=$(bc -l <<< "2 ^(-$LR_IDX)");
~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
  --experiment_group="$GROUP_NAME" \
  --config="mu_transformer/configs/dm4096.py" \
  --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
  --mode="train" \
  --rng_seed=0 \
  --rng_fold=False \
  --wb_enabled=True \
  --config.is_sweep=False \
  --config.force_download=False \
  --config.n_ds_shard=16 \
  --config.lr_base="$LR" \
  --config.d_model=3456 \
  --config.n_layer=27 \
  --config.u_init="sp" \
  --config.qk_scale=0.08838834764831845 \
  --config.ff_act_name="swiglu" \
  --config.ff_multiple=4.0 \
  --config.norm_eps=1e-6 \
  --config.norm_gains="$NORM_PARAMS" \
  --config.tokens_per_global_batch=1048576 \
  --config.lr_schedule_name="cosine" \
  --config.optim_rule="sp" \
  --config.wd=0.1 \
  --config.use_iwd=False \
  --config.n_warmup_step=2000 \
  --config.n_pretrain_step=100000;
