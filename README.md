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
