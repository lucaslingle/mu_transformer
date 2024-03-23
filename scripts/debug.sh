#!/bin/bash

GROUP_NAME="brrformer";
~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
    --experiment_group="$GROUP_NAME" \
    --config="mu_transformer/configs/small.py" \
    --workdir="gs://tpu_persist_bucket/brrformer/" \
    --mode="train" \
    --wb_enabled=True \
    --config.is_sweep=True \
    --config.force_download=False \
    --config.n_ds_shard=8 \
    --config.lr_base=0.01625 \
    --config.dtype=bfloat16 \
    --config.n_mesh_rows=32 \
    --config.n_mesh_cols=1;
