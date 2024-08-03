#!/bin/bash

GROUP_NAME="rebuttal_qinit";
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
for RNG_SEED in 0 1 2 3 4 5 6 7 8 9;
do
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --seed="$RNG_SEED" \
        --config="mu_transformer/configs/$SIZE.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="train" \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.force_download=False \
        --config.n_ds_shard=16 \
        --config.lr_base="$LR" \
        --config.dtype="bfloat16" \
        --config.q_init="zero";
done;
