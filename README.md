# Mu-Transformer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pre-commit: enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://wandb.ai/lucaslingle/mu_transformer/workspace?workspace=user-lucaslingle">
    <img src="https://img.shields.io/badge/Weights_&_Biases-black?style=for-the-badge&logo=WeightsAndBiases&logoColor=yellow"  height="20" />
</a>

Transformer decoder with [Mu-Parameterization](https://arxiv.org/abs/2203.03466) in Jax/Flax.

- Passes the correctness tests: *wider is better throughout training* and *coordinate checking*.
- Supports any HuggingFace text dataset and tokenizer.
- Supports distributed training on TPU pod slices. 
- Supports data parallelism and tensor parallelism.
- Supports mixed precision training, performing forward/backward in bfloat16.
- Simple, flexible configuration.

## Installation

To proceed, install [Pipx](https://github.com/pypa/pipx) and [Poetry](https://github.com/python-poetry/poetry). Then run
```
git clone https://github.com/lucaslingle/mu_transformer.git;
cd mu_transformer;
poetry install --with cpu  # CPU
poetry install --with tpu  # Cloud TPU VM
```
- On Cloud TPU VMs, Pipx and Poetry can be installed via ```./tpu_setup.sh```.
- On Cloud TPU VMs, you may need to write ```~/.local/bin/poetry``` when invoking Poetry. 

## Basics

### Default config

To train a tiny model on OpenWebText for 10K steps, you can run
```
poetry run python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=/tmp/workdir;
```
A series of model configs, each increasing model size by about 4x, are provided.   
Settings follow ```base.py```, with exception of width and mesh sizes.

The tiny model should be a sufficient proxy for hyperparameter search. 
However, you may wish to change the batch size, sequence length, number of layers, and training steps to match your target setting. 

### CLI overrides

Setting overrides are supported via command line. Here are some examples:
```
poetry run python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=/tmp/workdir \
    --config.tokens_per_global_batch=131072 \
    --config.sequence_len=1024 \
    --config.n_layer=24 \
    --config.n_pretrain_step=125000 \
    --config.n_warmup_step=10000 \
    --config.lr_max=0.03;
```
You may also need to override the ```n_mesh_rows```, ```n_mesh_cols```, ```n_mesh_planes``` settings so that their product matches the total number of available devices. These three settings correspond to sharding directions for the batch, residual and hidden axes, respectively.  

### Data and tokenizer

This project supports any HuggingFace text dataset and tokenizer out-of-the-box.  
For instance, to use the [T5 tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5TokenizerFast) and [C4 dataset](https://huggingface.co/datasets/c4), you can write
```
poetry run python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=/tmp/workdir \
    --config.hftr_tokenizer_name=T5TokenizerFast \
    --config.hftr_tokenizer_shortname=t5-base \
    --config.hfds_identifier=c4 \
    --config.hfds_config=en \
    --config.hfds_datacol=text;
```

### Logging to w&b

To enable logging to [Weights and Biases](https://wandb.ai/), you can run with ```--wb_enabled=True```.

### Training daemon

You can use [tmux](https://github.com/tmux/tmux) to allow training on cloud hosts without staying logged in.

## Acknowledgements

This project was supported by Cloud TPUs from Google's TPU Research Cloud (TRC).
