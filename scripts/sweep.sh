#!/bin/bash

Help() {
  echo "Syntax: sweep.sh [l|c|h]"
  echo "options:"
  echo "l     -log2(LR): a positive integer."
  echo "c     config name."
  echo "h     Print this Help."
  echo
}

while getopts "l:c:h" option; do
  case $option in
    l)
      LR_IDX=$OPTARG;;
    c)
      CONFIG=$OPTARG;;
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
  --config="mu_transformer/configs/$CONFIG.py" \
  --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
  --mode="train" \
  --wb_enabled=True \
  --config.is_sweep=True \
  --config.force_download=False \
  --config.n_ds_shard=16 \
  --config.lr_base="$LR" \
  --config.dtype="bfloat16";
