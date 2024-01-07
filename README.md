# Mu-Transformer

Transformer decoder with [Mu-Parameterization](https://arxiv.org/abs/2203.03466) in Jax/Flax.

- Passes the correctness tests: *wider is better throughout training* and *coordinate checking*.
- Supports any HuggingFace text dataset and tokenizer.
- Supports distributed training on TPU pod slices. 
- Supports data parallelism and tensor parallelism.
- Supports mixed precision training, performing forward/backward in bfloat16.
- Simple, flexible configuration.

## Installation

### Installation via pip

We require Python 3.9 for compatibility with all dependencies.   
On TPU VM, it can be installed as follows:
```
sudo apt update;
sudo apt install python3-venv -y;
sudo apt install python3.8-venv -y;
sudo apt install python3.9-venv -y;
```

General installation can then be performed with
```
pip3 install --upgrade pip;
git clone https://github.com/lucaslingle/mu_transformer.git;
cd mu_transformer;

#### CPU-Only
pip3 install -e '.[cpu]'; 

#### Cloud TPU VM
pip3 install -e '.[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```

### Installation via poetry

You need to install [Pipx](https://github.com/pypa/pipx) and [Poetry](https://github.com/python-poetry/poetry) if they're not installed. 

On a Cloud TPU VM, you can do this via
```
python3 -m pip install --user pipx;
python3 -m pipx ensurepath;
~/.local/bin/pipx install poetry;
```

General installation can then be performed with
```
git clone https://github.com/lucaslingle/mu_transformer.git;
cd mu_transformer;

### CPU
~/.local/bin/poetry install --with cpu;

### Cloud TPU VM
~/.local/bin/poetry install --with tpu;
```

## Examples

To train with the default config applied to a tiny model, you can run
```
python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=workdir;
```
A series of model configs, each increasing model size by about 4x, are provided.  
Settings follow ```base.py```, with exception of width and mesh sizes. 

Settings overrides are supported via command line
```
python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=workdir \
    --config.lr_max=0.123;
```

This project supports any HuggingFace text dataset and tokenizer out-of-the-box.  
For instance, to use the [T5 tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5TokenizerFast) and [C4 dataset](https://huggingface.co/datasets/c4), you can write
```
python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=workdir \
    --config.hftr_tokenizer_name=T5TokenizerFast \
    --config.hftr_tokenizer_shortname=t5-base \
    --config.hfds_identifier=c4 \
    --config.hfds_config=en \
    --config.hfds_datacol=text;
```

## Acknowledgements

This project was supported by Cloud TPUs from Google's TPU Research Cloud (TRC).
