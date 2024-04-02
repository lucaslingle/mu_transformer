#!/bin/bash

Help() {
  echo "Syntax: sweep_bsz_large.sh [l|h]"
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

GROUP_NAME="bsz";
LR=$(bc -l <<< "2 ^(-$LR_IDX)");
for size in "small" "medium" "large";
do
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/$size.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="train" \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.force_download=False \
        --config.n_ds_shard=16 \
        --config.lr_base="$LR" \
        --config.dtype=bfloat16 \
        --config.tokens_per_global_batch=1048576 \
        --config.optim_beta1=0.9 \
        --config.optim_beta2=0.95 \
        --config.optim_eps=0.00000001 \
        --config.n_warmup_step=2400 \
        --config.n_pretrain_step=30000;
done
