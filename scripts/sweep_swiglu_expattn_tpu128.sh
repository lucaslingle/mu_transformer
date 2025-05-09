#!/bin/bash

GROUP_NAME="swigluexpattn";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [l|a|h]"
  echo "options:"
  echo "l     -log2(LR): a positive integer."
  echo "a     attn nonlin: one of softmax, sqrelu, elu, norm_sqrelu, norm_elu."
  echo "h     Print this Help."
  echo
}

while getopts "l:a:h" option; do
  case $option in
    l)
      LR_IDX=$OPTARG;;
    a)
      ATTN_NONLIN=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

LR=$(bc -l <<< "2 ^(-$LR_IDX)");
for size in "dm128" "dm512" "dm2048";
do
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/$size.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="train" \
        --rng_seed=0 \
        --rng_fold=False \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.force_download=False \
        --config.n_ds_shard=16 \
        --config.lr_base="$LR" \
        --config.ff_act_name="swiglu" \
        --config.ff_multiple=2.5 \
        --config.attn_act_name="$ATTN_NONLIN";
done
