#!/bin/bash

GROUP_NAME="baseline_abs";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [l|h]"
  echo "options:"
  echo "a     LR: a positive float."
  echo "h     Print this Help."
  echo
}

while getopts "a:h" option; do
  case $option in
    a)
      LR=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

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
        --config.optim_rule="abs_mup";
done
