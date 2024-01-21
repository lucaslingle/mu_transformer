# Copyright 2024 Lucas Dax Lingle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import posixpath
import re
import sys
import time
from collections import namedtuple

import torch
import torch.cuda as cuda
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.optim as optim
import wandb
from absl import app
from absl import flags
from absl import logging
from etils.etree import py as etree
from ml_collections import config_flags
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa

from mu_transformer.data import count_batches
from mu_transformer.data import get_batch
from mu_transformer.data import get_dataset
from mu_transformer.data import get_tokenizer
from mu_transformer.pytorch_impl.model import Transformer
from mu_transformer.pytorch_impl.model import TransformerConfig


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration file", lock_config=False)
flags.DEFINE_string("workdir", None, "Working directory")
flags.DEFINE_enum("mode", None, ["train", "validation", "test"], "Mode")
flags.DEFINE_integer("seed", 0, "Experiment seed")
flags.DEFINE_boolean("wb_enabled", False, "Log to W&B")
flags.DEFINE_string("wb_run", None, "W&B run id, for resuming with continuity")
flags.DEFINE_string("loading_name", None, "Model name to load; None = use autogen")
flags.DEFINE_string("saving_name", None, "Model name to save; None = use autogen")
flags.mark_flags_as_required(["config", "workdir", "mode"])


TrainState = namedtuple("StateTuple", field_names=["config", "model", "opt", "sched"])


@functools.lru_cache(maxsize=1)
def device_factory():
    return torch.device("cpu" if not cuda.is_available() else f"cuda:{dist.get_rank()}")


@functools.lru_cache(maxsize=1)
def tokenizer_factory():
    return get_tokenizer(
        FLAGS.config.hftr_tokenizer_name,
        FLAGS.config.hftr_tokenizer_shortname,
    )


@functools.lru_cache(maxsize=1)
def transformer_config_factory():
    return TransformerConfig(
        **vars(FLAGS.config)["_fields"],
        n_vocab=tokenizer_factory().vocab_size,
        bos_token_id=tokenizer_factory().bos_token_id,
        eos_token_id=tokenizer_factory().eos_token_id,
        pad_token_id=tokenizer_factory().pad_token_id,
        device=device_factory(),
    )


def model_factory():
    config = transformer_config_factory()
    model = Transformer(config)
    model = DDP(model, device_ids=device_factory())
    return model


def filter_params(model, suffix):
    return [p for n, p in model.named_parameters() if n.endswith(suffix)]


def optimizer_factory(model):
    kws = dict(
        betas=(FLAGS.config.adam_b1, FLAGS.config.adam_b2),
        eps=FLAGS.config.adam_eps,
    )
    lr = FLAGS.config.lr_max
    dm = FLAGS.config.d_model
    dff = FLAGS.config.d_model * FLAGS.config.ff_multiple
    # adam optimizer for standard parametrization
    if not FLAGS.config.use_mup:
        return optim.Adam(model.params(), lr=lr, **kws)
    # adam optimizer for mu-parametrization
    lr_groups = [
        dict(params=filter_params(model, "w_ei"), lr=lr),  # table 3 col 1
        dict(params=filter_params(model, "w_eo"), lr=lr / dm),  # table 3 col2
        dict(params=filter_params(model, "w_aq"), lr=lr / dm),  # table 3 col3
        dict(params=filter_params(model, "w_ak"), lr=lr / dm),  # table 3 col3
        dict(params=filter_params(model, "w_av"), lr=lr / dm),  # table 3 col3
        dict(params=filter_params(model, "w_ao"), lr=lr / dm),  # table 3 col3; dm=nh*dh
        dict(params=filter_params(model, "w_fi"), lr=lr / dm),  # table 3 col3
        dict(params=filter_params(model, "w_fo"), lr=lr / dff),  # table 3 col3
    ]
    return optim.Adam(lr_groups, **kws)


def global_batch_size_factory():
    assert FLAGS.config.tokens_per_global_batch % FLAGS.config.sequence_len == 0
    global_bsz = FLAGS.config.tokens_per_global_batch // FLAGS.config.sequence_len
    assert global_bsz % dist.get_world_size() == 0
    return global_bsz


def automatic_modelname_factory():
    dataset_name = FLAGS.config.hfds_identifier.split("/")[-1].lower()
    assert re.search(r"^[a-zA-Z0-9-_]+$", dataset_name) is not None  # ^=start, $=end.
    lr = str(FLAGS.config.lr_max).split(".")
    assert len(lr) == 2
    parts = [
        "pytorch",
        "mutransformer",
        "mup" if FLAGS.config.use_mup else "sp",
        dataset_name,
        FLAGS.config.model_size,
        f"a{lr[0]}point{lr[1]}",
        f"b{global_batch_size_factory()}",
        f"t{FLAGS.config.sequence_len}",
        f"s{FLAGS.config.n_pretrain_step}",
        f"r{FLAGS.seed}",
    ]
    return "_".join(parts)


def modelname_factory(option):
    if option == "save":
        return FLAGS.saving_name or automatic_modelname_factory()
    if option == "load":
        return FLAGS.loading_name or automatic_modelname_factory()
    raise NotImplementedError(f"Unrecognized option {option}")


def modeldir_factory(option, suffix):
    return posixpath.join(FLAGS.workdir, modelname_factory(option), suffix)


def get_loss_mask(batch, *, pad_token_id, eos_token_id):
    # loss mask that allows training on first occurring eos/pad token as a target,
    # even if eos_token_id == pad_token_id
    loss_mask = torch.logical_or(
        torch.eq(batch, pad_token_id),
        torch.eq(batch, eos_token_id),
    )
    loss_mask = torch.logical_not(loss_mask)
    loss_mask = F.pad(loss_mask[:, 0:-1], (1, 0), value=True)
    loss_mask = torch.cumprod(loss_mask, dim=-1)  # mask everything after the first eos
    return loss_mask


def loss_fn(config, model, batch):
    logits, intermediates = model(batch)  # tokens shifted internally by model
    mask = get_loss_mask(
        batch, pad_token_id=config.pad_token_id, eos_token_id=config.eos_token_id
    )
    loss_term_avg = F.cross_entropy(
        weight=mask, input=logits, target=batch, reduction="mean"
    )
    metrics = dict(
        loss_term_avg=loss_term_avg.detach(),
        loss_mask_avg=mask.mean().detach(),
        **intermediates,
    )
    return loss_term_avg, metrics


def treesz(x):
    return etree.tree_reduce(lambda a, b: a + b.numel(), x, initializer=0)


def treel2(x):
    return (
        etree.tree_reduce(lambda a, b: a + torch.sum(b**2), x, initializer=0) ** 0.5
    )


def global_mean(x):
    return dist.all_reduce(x) / dist.get_world_size()


def train_step(state, batch):
    state.opt.zero_grad(set_to_none=True)
    loss_term_avg, metrics = loss_fn(state.config, state.model, batch)
    loss_term_avg.backward()  # global mean of grads obtained automatically via DDP

    metrics = etree.tree_map(lambda x: global_mean(x), metrics)
    if FLAGS.config.sow_intermediates:
        metrics["param_count"] = treesz(state.model.params())
        metrics["param_count"] = treel2(state.model.params())
        metrics["grad_norm"] = treel2(etree.map(lambda p: p.grad, state.model.params()))
        metrics["loss_avg"] = metrics["loss_term_avg"] / metrics["loss_mask_avg"]

    nn.utils.clip_grad_norm_(state.model.params(), FLAGS.config.grad_clip)
    state.optimizer.step()
    state.scheduler.step()
    return state, metrics


def train_loop():
    logging.info("Entering train loop function...")
    logging.info("Creating W&B connection...")

    if dist.get_rank() == 0:
        wandb.init(
            project="mu_transformer_private",
            config=vars(FLAGS.config)["_fields"],
            resume="never" if FLAGS.wb_run is None else "must",
            mode="online" if FLAGS.wb_enabled else "disabled",
            id=FLAGS.wb_run,
        )
    logging.info("Creating RNGs...")
    # todo

    logging.info("Creating train state...")
    start_step = None  # todo

    batch_size = global_batch_size_factory() // dist.get_world_size()
    dataset_shard = get_dataset(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hfds_buffer_size=FLAGS.config.hfds_buffer_size,
        hftr_tokenizer=tokenizer_factory(),
        split_name="train",
        batch_size=batch_size,
        sequence_len=FLAGS.config.sequence_len,
        pcount=dist.get_world_size(),
        pindex=dist.get_rank(),
        workdir=FLAGS.workdir,
    )

    logging.info("Starting training loop...")
    best_val_loss = float("inf")
    val_metrics = dict()
    log_level_is_debug = logging.get_verbosity() == 1
    start_time = time.perf_counter()

    # the user should set n_finetune_step > 0 if and only if currently fine-tuning.
    n_total_step = FLAGS.config.n_pretrain_step + FLAGS.config.n_finetune_step
    for step in range(start_step, n_total_step + 1):
        logging.debug(f"Training step {step}...")
        batch = get_batch(
            dataset_shard,
            batch_size=batch_size,
            sequence_len=FLAGS.config.sequence_len,
            step=step,
        )
        logging.debug("Got batch...")
        logging.debug(f"Batch shape (local): {batch.shape}")

        # run a training step
        logging.debug("Starting train step...")
        state, metrics = train_step(state=state, batch=batch)
        logging.debug("Finished train step...")
