#!/bin/bash

GROUP_NAME="swiglu";
for size in "small" "medium" "large";
do
    for i in $(seq 2 2 10)
    do
        LR=$(bc -l <<< "2 ^(-$i)");
        ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
            --experiment_group="$GROUP_NAME" \
            --config="mu_transformer/configs/$size.py" \
            --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
            --mode="train" \
            --wb_enabled=True \
            --config.is_sweep=True \
            --config.lr_base="$LR" \
            --config.force_download=False \
            --config.ff_multiple=5.3 \
            --config.act_name="swiglu" \
            --config.act_square=False;
    done
done
