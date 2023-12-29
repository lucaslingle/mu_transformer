from setuptools import find_packages
from setuptools import setup


setup(
    name="mu_transformer",
    version="0.0.1",
    url="https://github.com/lucaslingle/mu_transformer/",
    license="Apache 2.0",
    author="Lucas D. Lingle",
    description="Transformer with Mu-Transfer, implemented in Jax.",
    packages=find_packages(where=".:src"),
    package_dir={"": "src"},
    platforms="any",
    python_requires=">=3.8",
    install_requires=[
        "absl-py==2.0.0",
        "etils==1.3.0",
        "ml_collections",
        "chex>=0.1.7",
        "transformers==4.35.2",
        "datasets==2.14.6",
        "huggingface_hub==0.19.3",
        "jaxlib>=0.4.9",
        "flax==0.7.2",
        "numpy>=1.22.0",
        "optax>=0.1.5",
        "orbax-checkpoint>=0.1.7",
        "requests>=2.28.1",
        "tensorflow==2.12.1",
        "tqdm>=4.65.0",
        "wandb<0.15.0",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
        ],
        "cpu": [
            "jax>=0.4.9",
        ],
        "tpu": [
            "jax[tpu]>=0.4.9",
        ],
    },
)
