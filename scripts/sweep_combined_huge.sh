#!/bin/bash

GROUP_NAME="combined";
SIZE="huge";

for i in $(seq 2 1 10)
do
    echo "VERIFY EVERYTHING IN THE CONFIG PRINTED!!!"
    LR=$(bc -l <<< "2 ^(-$i)");
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
        --experiment_group="$GROUP_NAME" \
        --config="mu_transformer/configs/$SIZE.py" \
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
