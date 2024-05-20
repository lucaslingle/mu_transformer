#!/bin/bash

GROUP_NAME="wd_flex";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [l|w|t|h]"
  echo "options:"
  echo "l     -log2(LR): a positive integer."
  echo "w     WD: a non-negative float."
  echo "t     T: a positive integer."
  echo "h     Print this Help."
  echo
}

while getopts "l:w:t:h" option; do
  case $option in
    l)
      LR_IDX=$OPTARG;;
      LR=$(bc -l <<< "2 ^(-$LR_IDX)");
    w)
      WD=$OPTARG;;
    t)
      T=$OPTARG;;
      WU=$(bc -l <<< "0.08 * $T / 1");
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
        --config.wd="$WD" \
        --config.n_warmup_step="$WU" \
        --config.n_pretrain_step="$T" \
        --config.act_name="relu" \
        --config.act_square=True \
        --config.q_init="zero";
done
