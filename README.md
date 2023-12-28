# mu_transformer

Transformer decoder with [Mu-Parameterization](https://arxiv.org/abs/2203.03466) in Jax/Flax.

- Supports distributed training on TPU pod slices and GPU clusters. 
- Supports data parallelism and tensor parallelism.
- Supports any HuggingFace dataset and any tokenizer.
- Simple, flexible configuration.
