#!/bin/bash

Help() {
  echo "Syntax: sweep_huge_proxy_qkscale.sh [l|h]"
  echo "options:"
  echo "l     -Log2 of starting lr in sweep, defaults to zero."
  echo "h     Print this Help."
  echo
}


GROUP_NAME="huge_proxy_qkscale";
while getopts "l:h" option; do
  case $option in
    l)
      QKSCALE_IDX_START=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done


for i in $(seq "$QKSCALE_IDX_START" 1 14);
do
    QK_SCALE=$(bc -l <<< "2 ^(-$i)");
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/small.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="train" \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.lr_base=0.015625 \
        --config.force_download=False \
        --config.dtype=bfloat16 \
        --config.d_base=256 \
        --config.d_model=256 \
        --config.d_head=256 \
        --config.qk_scale="$QK_SCALE" \
        --config.act_square=True \
        --config.wd=0.0001 \
        --config.n_warmup_step=15000 \
        --config.n_pretrain_step=180000;
done
