#!/bin/bash

GROUP_NAME="qkscale";
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
            --config.force_download=False \
            --config.n_ds_shard=16 \
            --config.lr_base="$LR" \
            --config.qk_scale=0.08838834764831845;
    done
done
