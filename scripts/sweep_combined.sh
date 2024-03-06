#!/bin/bash

Help() {
  echo "Syntax: sweep_combined.sh [s|l|h]"
  echo "options:"
  echo "s     Size of model (small, medium, large)."
  echo "l     -Log2 of starting lr in sweep, defaults to zero."
  echo "h     Print this Help."
  echo
}


GROUP_NAME="combined";
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
    echo "VERIFY EVERYTHING IN THE CONFIG PRINTED!!!"
    LR=$(bc -l <<< "2 ^(-$i)");
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/$SIZE_NAME.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="train" \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.lr_base="$LR" \
        --config.force_download=False \
        --config.n_layer=12 \
        --config.e_norm=True \
        --config.act_square=True \
        --config.n_pretrain_step=180000 \
        --config.n_warmup_step=15000 \
        --config.tokens_per_global_batch=1048576 \
        --config.optim_beta1=0.9 \
        --config.optim_beta2=0.95 \
        --config.optim_eps=0.00000001 \
        --config.wd=0.1;
done
