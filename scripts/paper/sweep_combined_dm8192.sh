#!/bin/bash

Help() {
  echo "Syntax: sweep_combined_huge.sh [l|h]"
  echo "options:"
  echo "l     -log2(LR): a positive integer."
  echo "h     Print this Help."
  echo
}

while getopts "l:h" option; do
  case $option in
    l)
      LR_IDX=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

GROUP_NAME="combined";
SIZE="dm8192";
LR=$(bc -l <<< "2 ^(-$LR_IDX)");

~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
    --experiment_group="$GROUP_NAME" \
    --config="mu_transformer/configs/$SIZE.py" \
    --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
    --mode="train" \
    --rng_seed=0 \
    --rng_fold=True \
    --wb_enabled=True \
    --config.is_sweep=True \
    --config.force_download=False \
    --config.n_ds_shard=16 \
    --config.lr_base="$LR" \
    --config.n_layer=12 \
    --config.q_init="zero" \
    --config.ff_act_name="sqrelu" \
    --config.n_save_step=500 \
    --config.n_pretrain_step=90000 \
    --config.n_warmup_step=7000 \
    --config.tokens_per_global_batch=2097152 \
    --config.optim_rule="abs_mup" \
    --config.optim_name="lion" \
    --config.optim_beta1=0.95 \
    --config.optim_beta2=0.98 \
    --config.optim_eps=0.0 \
    --config.wd=0.0001 \
    --config.use_iwd=True;
