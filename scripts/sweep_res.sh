#!/bin/bash

Help() {
  echo "Syntax: sweep_res.sh [g|r|s|m|h]"
  echo "options:"
  echo "r     Rule for scaling (sp, mup, spectral)."
  echo "s     Size of model (small, medium, large)."
  echo "m     Mode (train, validation)."
  echo "h     Print this Help."
  echo
}


GROUP_NAME="res";
while getopts "r:s:m:h" option; do
  case $option in
    r)
      RULE_NAME=$OPTARG;;
    s)
      SIZE_NAME=$OPTARG;;
    m)
      MODE_NAME=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done


for i in $(seq 0 2 10);
do
    LR=$(bc -l <<< "2 ^(-$i)");
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/$SIZE_NAME.py" \
        --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
        --mode="$MODE_NAME" \
        --wb_enabled=True \
        --config.is_sweep=True \
        --config.lr_base="$LR" \
        --config.force_download=False \
        --config.parameterization="$RULE_NAME" \
        --config.parallel_res=True;
done
