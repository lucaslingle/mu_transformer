#!/bin/bash

Help() {
  echo "Syntax: sweep_bsz.sh [s|l|h]"
  echo "options:"
  echo "s     Size of model (small, medium, large)."
  echo "l     -Log2 of starting lr in sweep, defaults to zero."
  echo "h     Print this Help."
  echo
}


GROUP_NAME="bsz";
while getopts "s:l:h" option; do
  case $option in
    s)
      SIZE_NAME=$OPTARG;;
    l)
      LR_IDX_START=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done


for i in $(seq "$LR_IDX_START" 1 10);
do
    LR=$(bc -l <<< "2 ^(-$i)");
    BSZ=$(bc -l <<< "2 ^(16)");
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/$SIZE_NAME.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="train" \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.lr_base="$LR" \
        --config.force_download=False \
        --config.tokens_per_global_batch="$BSZ" \
        --config.n_save_step=8000 \
        --config.n_warmup_step=32000 \
        --config.n_pretrain_step=384000;
done
