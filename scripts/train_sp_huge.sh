~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
    --experiment_group="huge_sp" \
    --config="mu_transformer/configs/huge.py" \
    --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
    --mode="train" \
    --wb_enabled=True \
    --load_suffix="qkscale_$QK_SCALE" \
    --save_suffix="qkscale_$QK_SCALE" \
    --config.is_sweep=True \
    --config.lr_base=0.00048828125 \
    --config.force_download=False \
    --config.dtype=bfloat16 \
    --config.d_base=256 \
    --config.d_head=256 \
    --config.qk_scale=0.0625 \
    --config.u_init=vs \
    --config.act_square=True \
    --config.adam_rule=sp \
    --config.wd=0.0001 \
    --config.n_warmup_step=15000 \
    --config.n_pretrain_step=180000 \
    --config.n_mesh_rows=64 \
    --config.n_mesh_cols=8;
