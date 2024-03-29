#!/bin/bash

GROUP_NAME="combined";
for size in "small" "medium" "large";
do
    for i in $(seq 4 2 8)
    do
        echo "VERIFY EVERYTHING IN THE CONFIG PRINTED!!!"
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
            --config.n_layer=12 \
            --config.q_init="zero" \
            --config.act_square=True \
            --config.n_save_step=500 \
            --config.n_pretrain_step=90000 \
            --config.n_warmup_step=7000 \
            --config.tokens_per_global_batch=2097152 \
            --config.optim_beta1=0.9 \
            --config.optim_beta2=0.95 \
            --config.optim_eps=0.00000001 \
            --config.wd=0.1;
    done
done
