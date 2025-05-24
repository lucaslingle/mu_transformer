#!/bin/bash

GROUP_NAME="nemotron";
Help() {
  echo "Syntax: run_nemotron.sh [l|h]"
  echo ""
  echo "Trains a nemotron-style model with 4 billion parameters for 100 billion tokens."
  echo "Intended for larger-scale ablation studies."
  echo "Diffs vs llama: mha; d_ff = 4.5 * d_model; wd on embs & gains."
  echo ""
  echo "Options:"
  echo "l     learning rate."
  echo "h     Print this Help."
  echo
}

while getopts "l:h" option; do
  case $option in
    l)
      LR=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
  --experiment_group="$GROUP_NAME" \
  --config="mu_transformer/configs/dm4096.py" \
  --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
  --mode="train" \
  --rng_seed=0 \
  --rng_fold=False \
  --wb_enabled=True \
  --config.is_sweep=False \
  --config.force_download=False \
  --config.n_ds_shard=16 \
  --config.lr_base="$LR" \
  --config.d_model=3584 \
  --config.n_layer=28 \
  --config.u_init="sp" \
  --config.qk_scale=0.08838834764831845 \
  --config.ff_act_name="sqrelu" \
  --config.ff_multiple=4.5 \
  --config.norm_eps=1e-6 \
  --config.norm_gains=True \
  --config.tokens_per_global_batch=1048576 \
  --config.lr_schedule_name="cosine" \
  --config.lr_schedule_end_frac=0.1 \
  --config.optim_rule="sp" \
  --config.wd=0.1 \
  --config.use_iwd=False \
  --config.n_warmup_step=2000 \
  --config.n_pretrain_step=100000;
