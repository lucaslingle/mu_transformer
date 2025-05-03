#!/bin/bash

GROUP_NAME="llama_scaling";
Help() {
  echo "Syntax: run_llama_scaling.sh [n|d|h]"
  echo ""
  echo "Options:"
  echo "n     param count N."
  echo "d     token budget D."
  echo "h     Print this Help."
  echo
}

while getopts "n:d:h" option; do
  case $option in
    n)
      N=$OPTARG;;
    d)
      D=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

N_LAYER=$(python -c "import math; print(int(math.ceil(($N / (13 * 128 * 128)) ** 0.33)))");
D_MODEL=$(python -c "print($N_LAYER * 128)");
LR=$(bc -l <<< "2 ^(-$LR_IDX)");

for BSZ in 65536 131072 262144 524288 1048576 2097152 4194304;
do
  for LR in 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125;
  do
    N_STEP=$(python -c "print($D // $BSZ)");
    N_WARMUP=$(python -c "import math; print(math.ceil($N_STEP * 0.02))");
    ~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
      --experiment_group="$GROUP_NAME" \
      --config="mu_transformer/configs/dm2048.py" \
      --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
      --mode="train" \
      --rng_seed=0 \
      --rng_fold=False \
      --wb_enabled=True \
      --config.is_sweep=True \
      --config.force_download=False \
      --config.n_ds_shard=16 \
      --config.lr_base="$LR" \
      --config.d_model="$D_MODEL" \
      --config.n_layer="$N_LAYER" \
      --config.u_init="sp" \
      --config.qk_scale=0.08838834764831845 \
      --config.ff_act_name="swiglu" \
      --config.ff_multiple=3.0 \
      --config.norm_eps=1e-6 \
      --config.norm_gains=True \
      --config.tokens_per_global_batch="$BSZ" \
      --config.lr_schedule_name="cosine" \
      --config.lr_schedule_end_frac=0.1 \
      --config.optim_rule="sp" \
      --config.wd=0.1 \
      --config.use_iwd=False \
      --config.n_warmup_step="$N_WARMUP" \
      --config.n_pretrain_step="$N_STEP";
  done;
done;
