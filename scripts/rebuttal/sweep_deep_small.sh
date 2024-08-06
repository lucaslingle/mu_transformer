#!/bin/bash

GROUP_NAME="rebuttal_deep_small";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [l|h]"
  echo "options:"
  echo "l     negative learning rate exponent: -log2(LR)."
  echo "h     Print this Help."
  echo
}

while getopts "l:h" option;
do
  case $option in
    l)
      LR_IDX=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac;
done;

LR=$(bc -l <<< "2 ^(-$LR_IDX)");
SIZE="small";
~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
  --experiment_group="$GROUP_NAME" \
  --config="mu_transformer/configs/$SIZE.py" \
  --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
  --mode="train" \
  --wb_enabled=True \
  --config.is_sweep=True \
  --config.force_download=False \
  --config.n_ds_shard=16 \
  --config.lr_base="$LR" \
  --config.n_layer=96 \
  --config.r_init="zero" \
  --config.dtype="bfloat16";
