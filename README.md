# Mu-Transformer

Transformer decoder with [Mu-Parameterization](https://arxiv.org/abs/2203.03466) in Jax/Flax.

- Passes the correctness tests: *wider is better throughout training* and *coordinate checking*.
- Supports any HuggingFace text dataset and tokenizer.
- Supports distributed training on TPU pod slices. 
- Supports data parallelism and tensor parallelism.
- Supports mixed precision training, performing forward/backward in bfloat16.
- Simple, flexible configuration.

## Installation

We require Python 3.9 for compatibility with all dependencies. 

### CPU
Installation can be performed with
```
git clone https://github.com/lucaslingle/mu_transformer.git;
cd mu_transformer;
pip3 install --upgrade pip;
pip3 install -e '.[cpu]';  #### CPU-Only 
```

### TPU
On a Cloud TPU VM, we recommend installing via Poetry as follows: 
```
git clone https://github.com/lucaslingle/mu_transformer.git;
cd mu_transformer;
source tpu_setup.sh;
```
All Python commands should then be prefaced with ```~/.local/bin/poetry run```.

## Basics

### Default config

To train a tiny model on OpenWebText for 10K steps, you can run
```
python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=workdir;
```
A series of model configs, each increasing model size by about 4x, are provided.   
Settings follow ```base.py```, with exception of width and mesh sizes. 

### CLI overrides

Setting overrides are supported via command line
```
python3 mu_transformer/launch.py \
    --config=mu_transformer/configs/tiny.py \
    --mode=train \
    --workdir=workdir \
    --config.lr_max=0.123;
```
You may need to override the ```n_mesh_rows```, ```n_mesh_cols```, ```n_mesh_planes``` settings so that their product matches the total number of available devices. 

### Data and tokenizer

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
