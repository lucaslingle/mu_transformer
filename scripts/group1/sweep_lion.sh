#!/bin/bash

Help() {
  echo "Syntax: sweep_lion.sh [l|t|h]"
  echo "options:"
  echo "l     -log2(LR): a positive integer."
  echo "t     total steps: a positive integer."
  echo "h     Print this Help."
  echo
}

while getopts "l:t:h" option; do
  case $option in
    l)
      LR_IDX=$OPTARG;;
    t)
      TOTAL_STEPS=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

LR=$(bc -l <<< "2 ^(-$LR_IDX)");

for BETA1 in 0.91 0.92 0.93 0.94 0.95;
do
  for BETA2 in 0.95 0.96 0.97 0.98 0.99;
  do
    for WD_LAM in 0.3162 0.1 0.03162 0.01 0.0;
    do
      for WARMUP_FRAC in 0.0 0.1 0.5 0.9 1.0;
      do
        for DECAY_FRAC in 0.0 0.1 0.5 0.9 1.0;
        do
          GROUP_NAME="optim-v2";
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
            --config.optim_beta1="$BETA1" \
            --config.optim_beta2="$BETA2" \
            --config.optim_eps=0.0 \
            --config.wd="$WD_LAM" \
            --config.n_warmup_frac="$WARMUP_FRAC" \
            --config.n_total_step="$TOTAL_STEPS" \
            --config.n_decay_frac="$DECAY_FRAC";
        done;
      done;
    done;
  done;
done;
