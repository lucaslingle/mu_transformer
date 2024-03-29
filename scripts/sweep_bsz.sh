#!/bin/bash

GROUP_NAME="bsz";
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
            --config.dtype=bfloat16 \
            --config.tokens_per_global_batch=1048576 \
            --config.optim_beta1=0.9 \
            --config.optim_beta2=0.95 \
            --config.optim_eps=0.00000001 \
            --config.n_warmup_step=2400 \
            --config.n_pretrain_step=30000;
    done
done