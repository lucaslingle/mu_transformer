# Mu-Transformer

Transformer decoder with [Mu-Parameterization](https://arxiv.org/abs/2203.03466) in Jax/Flax.

- Supports distributed training on TPU pod slices and GPU clusters. 
- Supports data parallelism and tensor parallelism.
- Supports any HuggingFace dataset and any tokenizer.
- Simple, flexible configuration.

## Installation

It is recommended to install the python dependencies in a virtual environment such as venv, pipenv, or miniconda. 
After activating the environment, install as follows:
```
pip3 install --upgrade pip;
git clone https://github.com/lucaslingle/mu_transformer.git;
cd mu_transformer;

#### CPU-Only
pip3 install -e '.[cpu]'; 

#### Nvidia GPU, CUDA 11
pip3 install -e '.[cuda11]' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html;

#### Cloud TPU VM
pip3 install -e '.[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```
To install via pipenv, write ```pipenv install ...```. 

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
