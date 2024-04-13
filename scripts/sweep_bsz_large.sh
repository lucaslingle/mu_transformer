#!/bin/bash

GROUP_NAME="bsz_large";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [l|h]"
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

LR=$(bc -l <<< "2 ^(-$LR_IDX+1)");  # add +1 to exponent to mult LRs by 2x.
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
        --config.optim_beta2=0.98 \
        --config.optim_eps=0.000000001 \
        --config.n_warmup_step=2500 \
        --config.n_pretrain_step=31250;
done
