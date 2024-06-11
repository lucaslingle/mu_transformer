#!/bin/bash

Help() {
  echo "Syntax: sweep_lion.sh [l|h]"
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

LR=$(bc -l <<< "2 ^(-$LR_IDX)");
for b1 in 0.91 0.92 0.93 0.94 0.95;
do
    for b2 in 0.95 0.96 0.97 0.98 0.99;
    do
        GROUP_NAME="lion_$b1\_$b2";
        ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
            --experiment_group="$GROUP_NAME" \
            --config="mu_transformer/configs/small.py" \
            --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
            --mode="train" \
            --wb_enabled=True \
            --config.is_sweep=True \
            --config.force_download=False \
            --config.n_ds_shard=16 \
            --config.lr_base="$LR" \
            --config.dtype="bfloat16" \
            --config.d_head=32 \
            --config.qk_scale=0.03125 \
            --config.optim_name="lion" \
            --config.optim_beta1="$b1" \
            --config.optim_beta2="$b2" \
            --config.optim_eps=0.0 \
            --config.wd=0.0;
    done;
done;
