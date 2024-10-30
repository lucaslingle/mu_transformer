#!/bin/bash

GROUP_NAME="general";
Help() {
  echo "Syntax: sweep_$GROUP_NAME.sh [t|d|b|a|w|m|l|n|o|r|s|p|h]"
  echo "options:"
  echo "t     trial seed: note this git branch automatically folds in d_model, n_layer."
  echo "d     dtype: activation and gradient dtype (use one of bfloat16, float32)."
  echo "b     batch size: the global batch size, in tokens."
  echo "a     -log2(alpha): negative of base-2 logarithm of learning rate."
  echo "w     -log2(lambda): negative of base-2 logarithm of indep weight decay."
  echo "m     d_model: residual stream width."
  echo "l     n_layer: number of transformer blocks."
  echo "n     nonlinearity: one of relu, gelu, swiglu, sqrelu."
  echo "o     optimizer: one of adamw or lion."
  echo "r     rule: one of abs_mup, mup, sp."
  echo "s     schedule: one of linear, cosine, wsd, piecewise."
  echo "p     pretrain steps: number of training steps (this script uses 1% warmup)."
  echo "h     Print this help."
  echo
}

while getopts "t:d:b:a:w:m:l:n:o:r:s:p:h" option; do
  case $option in
    t)
      RNG_SEED=$OPTARG;;
    d)
      DTYPE=$OPTARG;;
    b)
      BATCH_SIZE=$OPTARG;;
    a)
      LR_IDX=$OPTARG;;
    w)
      WD_IDX=$OPTARG;;
    m)
      D_MODEL=$OPTARG;;
    l)
      N_LAYER=$OPTARG;;
    n)
      ACT_NAME=$OPTARG;;
    o)
      OPT_NAME=$OPTARG;;
    r)
      RULE_NAME=$OPTARG;;
    s)
      SCHED_NAME=$OPTARG;;
    p)
      PRETRAIN_STEPS=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done

N_ROWS=$(python3 -c "print(128 if $D_MODEL < 1024 else 32)");
N_COLS=$(python3 -c "print(1 if $D_MODEL < 1024 else 4)");
OPT_ALPHA=$(bc -l <<< "2 ^(-$LR_IDX)");
OPT_LAMBDA=$(bc -l <<< "2 ^(-$WD_IDX)");
OPT_BETA1=$(python3 -c "print(0.9 if '$OPT_NAME' == 'adamw' else 0.95)");
OPT_BETA2=$(python3 -c "print(0.95 if '$OPT_NAME' == 'adamw' else 0.98)");
WARMUP_STEPS=$(python3 -c "print(int(0.01 * $PRETRAIN_STEPS))");

~/.local/bin/poetry run python3 mu_transformer/jax_impl/launch.py \
  --experiment_group="$GROUP_NAME" \
  --config="mu_transformer/configs/base.py" \
  --workdir="gs://tpu_persist_bucket/mu_transformer_scaling/" \
  --mode="train" \
  --wb_enabled=True \
  --config.is_sweep=True \
  --config.force_download=False \
  --config.n_ds_shard=16 \
  --config.n_mesh_rows="$N_ROWS" \
  --config.n_mesh_cols="$N_COLS" \
  --seed="$RNG_SEED" \
  --config.dtype="$DTYPE" \
  --config.tokens_per_global_batch="$BATCH_SIZE" \
  --config.lr_base="$OPT_ALPHA" \
  --config.wd="$OPT_LAMBDA" \
  --config.d_model="$D_MODEL" \
  --config.n_layer="$N_LAYER" \
  --config.act_name="$ACT_NAME" \
  --config.optim_name="$OPT_NAME" \
  --config.optim_rule="$RULE_NAME" \
  --config.optim_beta1="$OPT_BETA1" \
  --config.optim_beta2="$OPT_BETA2" \
  --config.lr_schedule_name="$SCHED_NAME" \
  --config.n_pretrain_step="$PRETRAIN_STEPS" \
  --config.n_warmup_step="$WARMUP_STEPS" \
  --config.use_iwd=True \
  --config.use_eps_scaling=False;
