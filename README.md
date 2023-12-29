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
pip3 install '.[cpu]'; 

#### Nvidia GPU, CUDA 11
pip3 install '.[cuda11]' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html;

#### Cloud TPU VM
pip3 install '.[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```

To install in editable mode, write ```pip3 install -e ...```.  
To install via pipenv, write ```pipenv install ...```. 

## Examples

To train with the default hyperparameters and tiny config:
```
python3 scripts/launch.py \
    --config=scripts/configs/config_tiny.py \
    --mode=train \
    --workdir=workdir;
```

You can also override the configs. For example, to use the T5 tokenizer and C4 dataset, write
```
python3 scripts/launch.py \
    --config=scripts/configs/config_tiny.py \
    --mode=train \
    --workdir=workdir \
    --config.hftr_tokenizer_name=T5TokenizerFast \
    --config.hftr_tokenizer_shortname=t5-base \
    --config.hfds_identifier=c4 \
    --config.hfds_config=en \
    --config.hfds_datacol=text;
```
