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
import os
import posixpath
import re
import sys
import time
from collections import namedtuple

import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.optim as optim
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

from mu_transformer.data import get_batch
from mu_transformer.data import get_dataset
from mu_transformer.data import get_tokenizer
from mu_transformer.pytorch_impl.model import Transformer
from mu_transformer.pytorch_impl.model import TransformerConfig

# import jax.tree_util as jtu
# from mu_transformer.data import count_batches

#  import torch.distributed as dist
#  from torch.nn.parallel import DistributedDataParallel as DDP  # noqa


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


TrainState = namedtuple("TrainState", field_names=["model", "optim", "sched"])


@functools.lru_cache(maxsize=1)
def device_factory():
    # rank = dist.get_rank() # todo
    rank = 0  # todo
    return torch.device("cpu" if not cuda.is_available() else f"cuda:{rank}")


@functools.lru_cache(maxsize=1)
def tokenizer_factory():
    return get_tokenizer(
        FLAGS.config.hftr_tokenizer_name,
        FLAGS.config.hftr_tokenizer_shortname,
    )


@functools.lru_cache(maxsize=1)
def transformer_config_factory():
    return TransformerConfig.create(
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
    # model = DDP(model, device_ids=device_factory())  # todo
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
        return optim.Adam(model.parameters(), lr=lr, **kws)
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


def scheduler_factory(optimizer):
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0001,
        end_factor=1.0,
        total_iters=FLAGS.config.n_warmup_step,
    )
    decay = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=FLAGS.config.n_pretrain_step - FLAGS.config.n_warmup_step,
    )
    return optim.lr_scheduler.ChainedScheduler([warmup, decay])


def train_state_factory():
    model = model_factory()
    optimizer = optimizer_factory(model)
    scheduler = scheduler_factory(optimizer)
    return TrainState(model=model, optim=optimizer, sched=scheduler)


def global_batch_size_factory():
    assert FLAGS.config.tokens_per_global_batch % FLAGS.config.sequence_len == 0
    global_bsz = FLAGS.config.tokens_per_global_batch // FLAGS.config.sequence_len
    # assert global_bsz % dist.get_world_size() == 0  # todo
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


def load_state(state):
    model_path = modeldir_factory("load", "checkpoints/model.pth")
    optim_path = modeldir_factory("load", "checkpoints/optimizer.pth")
    sched_path = modeldir_factory("load", "checkpoints/schedule.pth")

    if os.path.exists(model_path):
        state.model.load_state_dict(torch.load(model_path))
    if os.path.exists(model_path):
        state.optim.load_state_dict(torch.load(optim_path))
    if os.path.exists(model_path):
        state.sched.load_state_dict(torch.load(sched_path))
    return state


def save_state(state):
    model_path = modeldir_factory("save", "checkpoints/model.pth")
    optim_path = modeldir_factory("save", "checkpoints/optimizer.pth")
    sched_path = modeldir_factory("save", "checkpoints/schedule.pth")

    torch.save(state.model.state_dict(), model_path)
    torch.save(state.optim.state_dict(), optim_path)
    torch.save(state.sched.state_dict(), sched_path)
    return state


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


def loss_fn(model, batch):
    outputs = model(batch)  # tokens shifted internally by model
    mask = get_loss_mask(
        batch=batch,
        pad_token_id=model.hps.pad_token_id,
        eos_token_id=model.hps.eos_token_id,
    )
    loss_terms = (
        -1.0
        * mask
        * torch.squeeze(
            input=torch.gather(outputs["logprobs"], dim=-1, index=batch[..., None]),
            dim=-1,
        )
    )
    metrics = dict(
        loss_term_avg=loss_terms.mean(),
        loss_mask_avg=mask.to(loss_terms.dtype).mean(),
        **outputs["intermediates"].to_dict(),
    )
    return metrics["loss_term_avg"], metrics


# def treesz(x):
#     return jtu.tree_reduce(lambda a, b: a + b.numel(), x, initializer=0)
#
#
# def treel2(x):
#     return jtu.tree_reduce(lambda a, b: a + torch.sum(b**2), x, initializer=0) ** 0.5


def global_mean(x):
    # return dist.all_reduce(x) / dist.get_world_size()  # todo
    return x


@functools.partial(torch.compile, mode="reduce-overhead")
def train_step(state, batch):
    state.optim.zero_grad(set_to_none=True)
    loss_term_avg, metrics = loss_fn(state.model, batch)
    loss_term_avg.backward()  # global mean of grads obtained automatically via DDP

    gnorm = nn.utils.clip_grad_norm_(state.model.parameters(), FLAGS.config.grad_clip)
    state.optim.step()
    state.sched.step()

    # metrics = jtu.tree_map(lambda x: global_mean(x), metrics)  # todo
    metrics["grad_norm"] = gnorm
    metrics["loss_avg"] = metrics["loss_term_avg"] / metrics["loss_mask_avg"]
    return state, metrics


def train_loop():
    logging.info("Entering train loop function...")
    logging.info("Creating W&B connection...")
    if True:  # dist.get_rank() == 0:  # todo
        wandb.init(
            project="mu_transformer_private",
            config=vars(FLAGS.config)["_fields"],
            resume="never" if FLAGS.wb_run is None else "must",
            mode="online" if FLAGS.wb_enabled else "disabled",
            id=FLAGS.wb_run,
        )

    logging.info("Creating train state...")
    state = train_state_factory()
    state = load_state(state)
    start_step = 0  # todo: parse from checkpoint name

    batch_size = global_batch_size_factory()  # // dist.get_world_size()  # todo
    dataset_shard = get_dataset(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hfds_buffer_size=FLAGS.config.hfds_buffer_size,
        hftr_tokenizer=tokenizer_factory(),
        split_name="train",
        batch_size=batch_size,
        sequence_len=FLAGS.config.sequence_len,
        pcount=1,  # dist.get_world_size(),  # todo
        pindex=0,  # dist.get_rank(),  # todo
        workdir=FLAGS.workdir,
    )

    logging.info("Starting training loop...")
    best_val_loss = float("inf")
    val_metrics = dict(loss_avg=torch.tensor([best_val_loss], dtype=torch.float32))
    start_time = time.perf_counter()
    # the user should set n_finetune_step > 0 if and only if currently fine-tuning.
    n_total_step = FLAGS.config.n_pretrain_step + FLAGS.config.n_finetune_step
    for step in range(start_step, n_total_step + 1):
        batch = get_batch(
            dataset_shard,
            batch_size=batch_size,
            sequence_len=FLAGS.config.sequence_len,
            step=step,
            out_dtype=np.int64,
        )
        state, metrics = train_step(
            state=state,
            batch=torch.from_numpy(batch).to(state.model.hps.device),
        )
        # occasionally print metrics
        if step % FLAGS.config.n_print_step == 0:
            logging.debug("Starting print action...")
            if cuda.is_available():
                cuda.synchronize()
            end_time = time.perf_counter()
            sec_per_step = (end_time - start_time) / FLAGS.config.n_print_step
            essentials = {
                "step": step,
                "sec_per_step": sec_per_step,
                "loss_avg": metrics.get("loss_avg").item(),
                "val_loss_avg": val_metrics.get("loss_avg").item(),
                "grad_norm": metrics.get("grad_norm").item(),
            }
            logging.info(essentials)
            if True:  # jax.process_index() == 0: # todo
                metrics.update(essentials)
                wandb.log(metrics)
            start_time = end_time
            logging.debug("Done with print action...")

        # occasionally evaluate
        if (step % FLAGS.config.n_save_step == 0) or step == n_total_step:
            logging.debug("Starting evaluation action...")
            val_metrics = eval_loop(state.model, n_eval_step=FLAGS.config.n_eval_step)
            if best_val_loss > val_metrics["loss_avg"].item():
                logging.info("Validation loss improved...")
                save_state(state)
                best_val_loss = val_metrics["loss_avg"].item()
            logging.debug("Done with evaluation action...")


def eval_loop(model, n_eval_step):
    return dict(loss_avg=torch.tensor([float("inf")], dtype=torch.float32))


def main(argv):
    del argv
    logging.info("=== Start of main() ===")
    logging.info(f"Python version: {sys.version.__repr__()}")
    logging.info("=== Flags: ===")
    logging.info(f"workdir: {FLAGS.workdir}")
    logging.info(f"mode: {FLAGS.mode}")
    logging.info(f"seed: {FLAGS.seed}")
    logging.info(f"wb_enabled: {FLAGS.wb_enabled}")
    logging.info(f"wb_run: {FLAGS.wb_run}")
    logging.info(f"loading_name: {FLAGS.loading_name}")
    logging.info(f"saving_name: {FLAGS.saving_name}")
    logging.info(f"verbosity: {FLAGS.verbosity}")
    logging.info("=== Config: ===")
    for k, v in vars(FLAGS.config)["_fields"].items():
        logging.info(f"{k}: {v}")
    assert global_batch_size_factory() % FLAGS.config.n_mesh_rows == 0
    assert FLAGS.config.n_mesh_rows >= 1
    assert FLAGS.config.n_mesh_cols == 1
    assert FLAGS.config.n_mesh_planes == 1

    if FLAGS.mode == "train":
        train_loop()


if __name__ == "__main__":
    app.run(main)
