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
from typing import Set

import blobfile
import flax
import flax.linen as nn
import jax
import jax.experimental.mesh_utils as jmu
import jax.experimental.multihost_utils as jmhu
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
import wandb
from absl import app
from absl import flags
from absl import logging
from flax import traverse_util
from flax.training import orbax_utils
from flax.training import train_state as train_utils
from ml_collections import config_flags

from mu_transformer.data import count_batches
from mu_transformer.data import get_batch
from mu_transformer.data import get_dataset
from mu_transformer.data import get_tokenizer
from mu_transformer.jax_impl.model import MESH_AXES
from mu_transformer.jax_impl.model import Transformer
from mu_transformer.jax_impl.model import TransformerConfig
from mu_transformer.jax_impl.samplers import apply_nucleus
from mu_transformer.jax_impl.samplers import apply_temp
from mu_transformer.jax_impl.samplers import apply_topk
from mu_transformer.jax_impl.shard import get_namedsharding
from mu_transformer.jax_impl.shard import sharding_constraint
from mu_transformer.jax_impl.shard import to_global_array
from mu_transformer.jax_impl.sow import split_and_name

MODES = ["train", "validation", "test", "sample"]
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration file", lock_config=False)
flags.DEFINE_string("experiment_group", None, "Experiment group name")
flags.DEFINE_string("workdir", None, "Working directory (GCS or local)")
flags.DEFINE_enum("mode", None, MODES, "Mode")
flags.DEFINE_integer("rng_seed", 0, "Experiment rng seed")
flags.DEFINE_boolean("rng_fold", False, "Fold bsz, train steps, width, depth into rng")
flags.DEFINE_boolean("wb_enabled", False, "Log to W&B")
flags.DEFINE_string("wb_run", None, "W&B run id, for resuming with continuity")
flags.DEFINE_string("load_suffix", None, "Suffix of model to load; prefix=autogen")
flags.DEFINE_string("save_suffix", None, "Suffix of model to save; prefix=autogen")
flags.mark_flags_as_required(["config", "experiment_group", "workdir", "mode"])


@functools.lru_cache(maxsize=1)
def tokenizer_factory():
    return get_tokenizer(
        FLAGS.config.hftr_tokenizer_name,
        FLAGS.config.hftr_tokenizer_instance,
    )


@functools.lru_cache(maxsize=3)
def transformer_config_factory(is_train, is_decoding):
    return TransformerConfig.create(
        **vars(FLAGS.config)["_fields"],
        n_vocab=tokenizer_factory().vocab_size,
        bos_token_id=tokenizer_factory().bos_token_id,
        eos_token_id=tokenizer_factory().eos_token_id,
        pad_token_id=tokenizer_factory().pad_token_id,
        is_train=is_train,
        is_decoding=is_decoding,
    )


@functools.lru_cache(maxsize=1)
def global_mesh_factory():
    return jax.sharding.Mesh(
        devices=jmu.create_device_mesh(
            mesh_shape=(
                FLAGS.config.n_mesh_rows,
                FLAGS.config.n_mesh_cols,
            ),
            devices=jax.devices(),
        ),
        axis_names=("X", "Y"),  # using 2D-finalized from GSPMD paper
    )


def params_factory(rng, model_cls, config, global_mesh):
    inputs = jnp.ones(dtype=jnp.int32, shape=[1, config.sequence_len])
    params = model_cls(config, global_mesh).init({"params": rng}, inputs)["params"]
    return params


def param_label_fn(params):
    flat = traverse_util.flatten_dict(params)
    flat_labels = {k: k[-1] for k, v in flat.items()}
    return traverse_util.unflatten_dict(flat_labels)


def schedule_factory():
    warm_steps = FLAGS.config.n_warmup_step
    cool_steps = FLAGS.config.n_pretrain_step - FLAGS.config.n_warmup_step
    zero_ = 1e-4
    warmup = optax.linear_schedule(zero_, 1.0, transition_steps=warm_steps)
    if FLAGS.config.lr_schedule_name == "linear":
        cooldown = optax.linear_schedule(1.0, zero_, transition_steps=cool_steps)
        return optax.join_schedules([warmup, cooldown], boundaries=[warm_steps])
    elif FLAGS.config.lr_schedule_name == "cosine":
        cooldown = optax.cosine_decay_schedule(1.0, alpha=zero_, decay_steps=cool_steps)
        return optax.join_schedules([warmup, cooldown], boundaries=[warm_steps])
    else:
        raise NotImplementedError(f"Unsupported sched: {FLAGS.config.lr_schedule_name}")


def get_standard_scaling(lr):
    return {
        # embeddings
        "g_e": lr,
        "w_e": lr,
        # attention
        "g_a": lr,
        "g_aq": lr,
        "g_ak": lr,
        "w_aq": lr,
        "w_ak": lr,
        "w_av": lr,
        "w_ao": lr,
        "b_aq": lr,
        "b_ak": lr,
        "b_av": lr,
        "b_ao": lr,
        # feed-forward
        "g_f": lr,
        "w_fi": lr,
        "w_fo": lr,
        "b_fi": lr,
        "b_fo": lr,
        # unembedding
        "g_u": lr,
        "w_u": lr,
        "b_u": lr,
    }


def get_rel_mup_scaling(lr):
    wm = FLAGS.config.d_model // FLAGS.config.d_base  # width multiple
    return {
        # embeddings
        "g_e": lr,
        "w_e": lr,
        # attention
        "g_a": lr,
        "g_aq": lr,
        "g_ak": lr,
        "w_aq": lr / wm,
        "w_ak": lr / wm,
        "w_av": lr / wm,
        "w_ao": lr / wm,
        "b_aq": lr,
        "b_ak": lr,
        "b_av": lr,
        "b_ao": lr,
        # feed-forward
        "g_f": lr,
        "w_fi": lr / wm,
        "w_fo": lr / wm,
        "b_fi": lr,
        "b_fo": lr,
        # unembedding
        "g_u": lr,
        "w_u": lr / wm,
        "b_u": lr,
    }


def get_abs_mup_scaling(lr):
    dm = FLAGS.config.d_model
    dff = FLAGS.config.d_model * FLAGS.config.ff_multiple
    return {
        # embeddings
        "g_e": lr,
        "w_e": lr,
        # attention
        "g_a": lr,
        "g_aq": lr,
        "g_ak": lr,
        "w_aq": lr / dm,
        "w_ak": lr / dm,
        "w_av": lr / dm,
        "w_ao": lr / dm,
        "b_aq": lr,
        "b_ak": lr,
        "b_av": lr,
        "b_ao": lr,
        # feed-forward
        "g_f": lr,
        "w_fi": lr / dm,
        "w_fo": lr / dff,
        "b_fi": lr,
        "b_fo": lr,
        # unembedding
        "g_u": lr,
        "w_u": lr / dm,
        "b_u": lr,
    }


def get_lrs():
    p = FLAGS.config.optim_rule
    if p == "sp":
        return get_standard_scaling(FLAGS.config.lr_base)
    if p == "mup":
        return get_rel_mup_scaling(FLAGS.config.lr_base)
    if p == "abs_mup":
        return get_abs_mup_scaling(FLAGS.config.lr_base)
    raise NotImplementedError(f"Unrecognized optim_rule: {p}")


def get_epss():
    p = FLAGS.config.optim_rule
    es = FLAGS.config.use_eps_scaling
    if es is False:
        return get_standard_scaling(FLAGS.config.optim_eps)
    if p == "sp":
        raise NotImplementedError("Eps scaling not supported for optim_rule sp.")
    if p == "mup":
        return get_rel_mup_scaling(FLAGS.config.optim_eps)
    if p == "abs_mup":
        return get_abs_mup_scaling(FLAGS.config.optim_eps)
    raise NotImplementedError(f"Unrecognized optim_rule: {p}")


def optimizer_factory():
    kws = dict(
        b1=FLAGS.config.optim_beta1,
        b2=FLAGS.config.optim_beta2,
        mu_dtype=FLAGS.config.dtype,
        weight_decay=0.0 if FLAGS.config.use_iwd else FLAGS.config.wd,
    )
    if FLAGS.config.optim_name == "adamw":
        lrs = get_lrs()
        epss = get_epss()
        return optax.multi_transform(
            {name: optax.adamw(lr, eps=epss[name], **kws) for name, lr in lrs.items()},
            param_labels=param_label_fn,
        )
    if FLAGS.config.optim_name == "lion":
        lrs = get_lrs()
        return optax.multi_transform(
            {name: optax.lion(lr, **kws) for name, lr in lrs.items()},
            param_labels=param_label_fn,
        )
    raise NotImplementedError(f"Unsupported optimizer: {FLAGS.config.optim_name}")


def grad_transform_factory():
    chain = []
    if FLAGS.config.grad_clip > 0.0:
        chain.append(optax.clip_by_global_norm(FLAGS.config.grad_clip))
    chain.append(optimizer_factory())
    if FLAGS.config.use_iwd:
        chain.append(optax.add_decayed_weights(-FLAGS.config.wd))
    chain.append(optax.scale_by_schedule(schedule_factory()))
    tx = optax.chain(*chain)
    if FLAGS.config.grad_acc_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=FLAGS.config.grad_acc_steps)
    return tx


def init_fn(rng, model_cls, optim_cls, config, global_mesh):
    return train_utils.TrainState.create(
        apply_fn=None,
        params=params_factory(
            rng=rng,
            model_cls=model_cls,
            config=config,
            global_mesh=global_mesh,
        ),
        tx=optim_cls,
    )


def train_state_factory(rng_init):
    # based on https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#the-output-s-sharding  # noqa
    model_cls = Transformer
    optimizer_cls = grad_transform_factory()
    config = transformer_config_factory(is_train=True, is_decoding=False)
    global_mesh = global_mesh_factory()

    prng_sharding = get_namedsharding(axis_names=(None,), device_mesh=global_mesh)
    abstract_variables = jax.eval_shape(
        functools.partial(
            init_fn,
            model_cls=model_cls,
            optim_cls=optimizer_cls,
            config=config,
            global_mesh=global_mesh,
        ),
        rng_init,
    )
    state_sharding = nn.get_sharding(abstract_variables, global_mesh)
    jit_init_fn = jax.jit(
        init_fn,
        static_argnums=(1, 2, 3, 4),
        in_shardings=(prng_sharding,),
        out_shardings=state_sharding,
    )
    initialized_state = jit_init_fn(
        rng_init, model_cls, optimizer_cls, config, global_mesh
    )
    return initialized_state


def global_batch_size_factory():
    assert FLAGS.config.tokens_per_global_batch % FLAGS.config.sequence_len == 0
    global_bsz = FLAGS.config.tokens_per_global_batch // FLAGS.config.sequence_len
    assert global_bsz % jax.process_count() == 0
    return global_bsz


def automatic_modelname_factory():
    dataset_name = FLAGS.config.hfds_identifier.split("/")[-1].lower()
    assert re.search(r"^[a-zA-Z0-9-_]+$", dataset_name) is not None  # ^=start, $=end.
    lr = str(FLAGS.config.lr_base).split(".")
    assert len(lr) == 2
    parts = [
        "mutransformer",
        dataset_name,
        FLAGS.experiment_group,
        f"t{FLAGS.rng_seed}",
        f"d{FLAGS.config.dtype}",
        f"b{FLAGS.config.tokens_per_global_batch}",
        f"a{FLAGS.config.lr_base}",
        f"w{FLAGS.config.wd}",
        f"m{FLAGS.config.d_model}",
        f"l{FLAGS.config.n_layer}",
        f"n{FLAGS.config.ff_act_name}",
        f"o{FLAGS.config.optim_name}",
        f"r{FLAGS.config.optim_rule}",
        f"s{FLAGS.config.lr_schedule_name}",
        f"p{FLAGS.config.n_pretrain_step}",
    ]
    return "_".join(parts)


def modelname_factory(option):
    name = automatic_modelname_factory()
    if option == "save":
        if FLAGS.save_suffix is not None:
            name += "_" + FLAGS.save_suffix
    elif option == "load":
        if FLAGS.load_suffix is not None:
            name += "_" + FLAGS.load_suffix
    else:
        raise NotImplementedError(f"Unrecognized option {option}")
    return name


def modeldir_factory(option, suffix):
    return posixpath.join(FLAGS.workdir, modelname_factory(option), suffix)


def checkpoint_manager_factory(option):
    return ocp.CheckpointManager(
        directory=modeldir_factory(option, "checkpoints"),
        checkpointers=ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=1,
            save_interval_steps=1,
            step_prefix="state",
        ),
    )


def do_restore(mgr, target):
    return mgr.restore(
        step=mgr.latest_step(),
        items=target,
        restore_kwargs=dict(
            restore_args=orbax_utils.restore_args_from_target(target, mesh=None),
        ),
    )


def do_save(mgr, step, target):
    mgr.save(
        step=step,
        items=target,
        save_kwargs=dict(
            save_args=orbax_utils.save_args_from_target(target),
        ),
    )


def size_pytree(x):
    return jtu.tree_reduce(lambda a, b: a + b.size, x, initializer=0.0)


def l2norm_pytree(x):
    return jtu.tree_reduce(lambda a, b: a + jnp.sum(b**2), x, initializer=0.0) ** 0.5


def get_loss_mask(batch, *, pad_token_id, eos_token_id):
    # loss mask that allows training on first occurring eos/pad token as a target,
    # even if eos_token_id == pad_token_id
    loss_mask = jnp.logical_or(
        jnp.equal(batch, pad_token_id),
        jnp.equal(batch, eos_token_id),
    )
    loss_mask = jnp.logical_not(loss_mask)
    loss_mask = jnp.pad(loss_mask[:, 0:-1], ((0, 0), (1, 0)), constant_values=True)
    loss_mask = jnp.cumprod(loss_mask, axis=-1)  # mask everything after the first eos
    return loss_mask


def maybe_untuple(maybe_tuple):
    if isinstance(maybe_tuple, tuple):
        if len(maybe_tuple) == 1:
            return maybe_tuple[0]
    return maybe_tuple


def maybe_unbox(tensor):
    if isinstance(tensor, flax.core.meta.Partitioned):
        tensor = tensor.value
    return tensor


def clean_and_flatten(pytree, split_filter: Set[str]):
    pytree = traverse_util.flatten_dict(pytree)
    pytree = {k: maybe_untuple(v) for k, v in pytree.items()}
    pytree = {k: maybe_unbox(v) for k, v in pytree.items()}
    pytree = {
        k: (
            split_and_name(k[-1], v)
            if any([s in k[-1] for s in split_filter])
            else {k[-1]: v}
        )
        for k, v in pytree.items()
    }
    pytree = traverse_util.flatten_dict(pytree)
    pytree = {k[-1]: v for k, v in pytree.items()}
    return pytree


def loss_fn(params, batch, config, global_mesh):
    bos, eos, pad = config.bos_token_id, config.eos_token_id, config.pad_token_id
    inp = jnp.pad(batch[:, 0:-1], ((0, 0), (1, 0)), constant_values=bos)
    inp = sharding_constraint(inp, MESH_AXES["XN"], global_mesh)
    tgt = sharding_constraint(batch, MESH_AXES["XN"], global_mesh)
    init_args = [config, global_mesh]
    apply_args = [{"params": params}, inp]
    if FLAGS.config.sow_intermediates:
        outp, mv = Transformer(*init_args).apply(*apply_args, mutable="intermediates")
        sown = clean_and_flatten(mv["intermediates"], split_filter={""})  # split all
    else:
        outp = Transformer(*init_args).apply(*apply_args)
        sown = dict()
    logits = outp.get("logits")
    terms = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=tgt)
    terms = sharding_constraint(terms, MESH_AXES["XN"], global_mesh)
    mask = get_loss_mask(batch, pad_token_id=pad, eos_token_id=eos)
    mask = sharding_constraint(mask, MESH_AXES["XN"], global_mesh)
    metrics = dict(
        loss_term_avg=jnp.mean(mask * terms),
        loss_mask_avg=jnp.mean(mask),
    )
    return metrics["loss_term_avg"], dict(**metrics, **sown)


def get_current_lr(name, step):
    name_without_layer = "_".join(name.split("_")[0:2])
    tensor_lr = get_lrs()[name_without_layer]
    schedule_now = schedule_factory()(step)
    return tensor_lr * schedule_now


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params,
        batch,
        transformer_config_factory(is_train=True, is_decoding=False),
        global_mesh_factory(),
    )
    # No extra mean anywhere, already have the sharded all-device mean gradient & loss.
    # Estimate ce loss for global batch: sum of non-masked ce terms / sum of mask values
    # Equivalently,
    metrics["loss_avg"] = metrics["loss_term_avg"] / metrics["loss_mask_avg"]
    # Compute param count
    metrics["param_count_total"] = size_pytree(state.params)
    if FLAGS.config.sow_param_info:
        # Maybe save update coordinate size (here, mean abs), scaled by current lr
        step = state.step
        p_old = state.params
        state = state.apply_gradients(grads=grads)  # do update
        p_new = state.params
        p_old = clean_and_flatten(p_old, split_filter={"w_a", "w_f", "g_a", "g_f"})
        p_new = clean_and_flatten(p_new, split_filter={"w_a", "w_f", "g_a", "g_f"})
        info = jtu.tree_map(lambda a, b: jnp.mean(jnp.abs(a - b)), p_new, p_old)
        info = {f"uu_{k}": v / get_current_lr(k, step) for k, v in info.items()}
    else:
        state = state.apply_gradients(grads=grads)  # do update
        info = dict()
    return state, dict(**metrics, **info)


def get_tpuv3_mfu(param_count, sec_per_step):
    """Estimate model flops utilization (MFU)"""
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    # get throughput estimate in tokens per second
    tokens_per_sec = FLAGS.config.tokens_per_global_batch / sec_per_step
    # get flop count estimate per token using analytical accounting
    n = param_count
    l = FLAGS.config.n_layer
    h = FLAGS.config.d_model // FLAGS.config.d_head
    q = FLAGS.config.d_head
    t = FLAGS.config.sequence_len
    flop_per_token = (6 * n) + (12 * l * h * q * t)
    # get estimated flop count per second (flop/s)
    flop_per_second_analytic = flop_per_token * tokens_per_sec
    # get peak theoretical flop count per second for the tpu v3 pod slice.
    #   formula: TPU v3 flop/s per chip * 4 chips per host * num hosts:
    flop_per_second_peak = 123e12 * 4 * jax.process_count()
    mfu = flop_per_second_analytic / flop_per_second_peak
    return mfu * 100  # a percentage


def get_scalar_on_host(tensor):
    tensor = jax.device_get(tensor)
    return tensor.item()


def get_rngs():
    rng_init = jax.random.PRNGKey(FLAGS.rng_seed)
    if FLAGS.rng_fold:
        # fold in B, S, M, L
        rng_init = jax.random.fold_in(rng_init, FLAGS.config.tokens_per_global_batch)
        rng_init = jax.random.fold_in(rng_init, FLAGS.config.n_pretrain_step)
        rng_init = jax.random.fold_in(rng_init, FLAGS.config.d_model)
        rng_init = jax.random.fold_in(rng_init, FLAGS.config.n_layer)
    rng_init, rng_stoch = jax.random.split(rng_init)
    return rng_init, rng_stoch


def train_loop():
    logging.info("Entering train loop function...")
    logging.info("Creating RNGs...")
    rng_init, _ = get_rngs()

    logging.info("Creating train state...")
    state = train_state_factory(rng_init)
    if not FLAGS.config.no_checkpoint:
        load_checkpoint_mgr = checkpoint_manager_factory(option="load")
        if load_checkpoint_mgr.latest_step() is not None:
            state = do_restore(load_checkpoint_mgr, state)
        start_step = load_checkpoint_mgr.latest_step() or 0
        del load_checkpoint_mgr
        if start_step == FLAGS.config.n_pretrain_step + FLAGS.config.n_finetune_step:
            return
    else:
        start_step = 0

    logging.info("Creating dataset...")
    n_host = jax.process_count()
    host_id = jax.process_index()
    n_shard = FLAGS.config.n_ds_shard or n_host
    n_host_per_shard = n_host // n_shard  # n_ds_shard = n_host on smallest runs
    global_batch_size = global_batch_size_factory()
    batch_size_per_host = global_batch_size // n_host
    shard_id = host_id // n_host_per_shard
    subshard_id = host_id % n_host_per_shard

    train_ds_shard = get_dataset(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hfds_buffer_size=FLAGS.config.hfds_buffer_size,
        hftr_tokenizer=tokenizer_factory(),
        split_name="train",
        batch_size=batch_size_per_host,
        sequence_len=FLAGS.config.sequence_len,
        n_shard=n_shard,
        shard_id=shard_id,
        workdir=FLAGS.workdir,
        force_download=FLAGS.config.force_download,
    )
    val_ds_shard = get_dataset(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hfds_buffer_size=FLAGS.config.hfds_buffer_size,
        hftr_tokenizer=tokenizer_factory(),
        split_name="validation",
        batch_size=batch_size_per_host,
        sequence_len=FLAGS.config.sequence_len,
        n_shard=n_shard,
        shard_id=shard_id,
        workdir=FLAGS.workdir,
        force_download=FLAGS.config.force_download,
    )

    logging.info("Starting training loop...")
    best_val_loss = float("inf")
    val_metrics = dict()
    global_mesh = global_mesh_factory()
    save_checkpoint_mgr = checkpoint_manager_factory(option="save")
    log_level_is_debug = logging.get_verbosity() == 1
    start_time = time.perf_counter()
    # the user should set n_finetune_step > 0 if and only if currently fine-tuning.
    n_total_step = FLAGS.config.n_pretrain_step + FLAGS.config.n_finetune_step
    for step in range(start_step, n_total_step + 1):
        # occasionally perform an evaluation and save a checkpoint on improvement
        if (step % FLAGS.config.n_save_step == 0) or step == n_total_step:
            state = jax.block_until_ready(state)
            # stop profiler
            if jax.process_index() == 0 and step == 2 * FLAGS.config.n_save_step:
                logging.info("Stopping profiler trace...")
                try:
                    jax.profiler.stop_trace()
                except RuntimeError as e:
                    # we ignore an error that occurs if restoring at step = 2 * n_save
                    if e.args[0] != "No profile started":
                        raise RuntimeError(e)
            logging.debug("Starting evaluation action...")
            val_metrics = eval_loop(
                state.params,
                ds_shard=val_ds_shard,
                n_eval_step=FLAGS.config.n_eval_step,
            )
            if best_val_loss > val_metrics["loss_avg"]:
                logging.info("Validation loss improved...")
                if not FLAGS.config.no_checkpoint:
                    do_save(save_checkpoint_mgr, step, state)
                best_val_loss = val_metrics["loss_avg"]
            # start profiler
            if jax.process_index() == 0 and step == FLAGS.config.n_save_step:
                assert FLAGS.config.n_save_step > FLAGS.config.n_print_step
                logging.info("Starting profiler trace...")
                jax.profiler.start_trace(
                    log_dir=modeldir_factory("save", "logging"),
                    # create_perfetto_trace=True,  # write extra trace file for perfetto
                )
            logging.debug("Done with evaluation action...")

        # do the training step
        logging.debug(f"Training step {step}...")
        batch = get_batch(
            shard=train_ds_shard,
            n_subshard=n_host_per_shard,
            subshard_id=subshard_id,
            batch_size=batch_size_per_host,
            sequence_len=FLAGS.config.sequence_len,
            step=step,
        )
        logging.debug("Got batch...")
        logging.debug(f"Batch shape (local): {batch.shape}")
        # distribute local batch arrays to global batch arrays
        logging.debug("Distributing batch to global array...")
        batch = to_global_array(batch, global_mesh)
        batch = jax.block_until_ready(batch) if log_level_is_debug else batch
        logging.debug("Finished distributing batch to global array...")
        logging.debug(f"Batch shape (global): {batch.shape}")
        # run a training step
        logging.debug("Starting train step...")
        state, metrics = train_step(state, batch=batch)
        state = jax.block_until_ready(state) if log_level_is_debug else state
        metrics = jax.block_until_ready(metrics) if log_level_is_debug else metrics
        logging.debug("Finished train step...")

        # occasionally print metrics
        if step % FLAGS.config.n_print_step == 0:
            state = jax.block_until_ready(state)
            metrics = {k: get_scalar_on_host(v) for k, v in metrics.items()}
            metrics = jax.block_until_ready(metrics)
            logging.debug("Starting print action...")
            end_time = time.perf_counter()
            sec_per_step = (end_time - start_time) / FLAGS.config.n_print_step
            essentials = {
                "step": step,
                "sec_per_step": sec_per_step,
                "loss_avg": metrics.get("loss_avg"),
                "val_loss_avg": val_metrics.get("loss_avg"),
                "tpuv3_mfu": get_tpuv3_mfu(metrics["param_count_total"], sec_per_step),
            }
            logging.info(essentials)
            if jax.process_index() == 0:
                metrics.update(essentials)
                wandb.log(metrics)
            start_time = end_time
            logging.debug("Done with print action...")


@jax.jit
def eval_step(params, batch):
    _, metrics = loss_fn(
        params,
        batch,
        transformer_config_factory(is_train=False, is_decoding=False),
        global_mesh_factory(),
    )
    return metrics


def eval_loop(params, ds_shard=None, n_eval_step=None, mode=None):
    logging.info("Entering eval loop function...")
    if params is None:
        rng_init, _ = get_rngs()
        logging.info("Creating params...")
        state = train_state_factory(rng_init)
        if not FLAGS.config.no_checkpoint:
            load_checkpoint_mgr = checkpoint_manager_factory(option="load")
            if load_checkpoint_mgr.latest_step() is not None:
                state = do_restore(load_checkpoint_mgr, state)
            del load_checkpoint_mgr
        state = jax.block_until_ready(state)
        params = state.params
        del state

    logging.info("Creating dataset...")
    n_host = jax.process_count()
    host_id = jax.process_index()
    n_shard = FLAGS.config.n_ds_shard
    n_host_per_shard = n_host // n_shard  # n_ds_shard = n_host on smallest runs
    global_batch_size = global_batch_size_factory()
    batch_size_per_host = global_batch_size // n_host
    shard_id = host_id // n_host_per_shard
    subshard_id = host_id % n_host_per_shard
    if ds_shard is None:
        ds_shard = get_dataset(
            hfds_identifier=FLAGS.config.hfds_identifier,
            hfds_config=FLAGS.config.hfds_config,
            hfds_datacol=FLAGS.config.hfds_datacol,
            hfds_buffer_size=FLAGS.config.hfds_buffer_size,
            hftr_tokenizer=tokenizer_factory(),
            split_name=mode,
            batch_size=batch_size_per_host,
            sequence_len=FLAGS.config.sequence_len,
            n_shard=n_shard,
            shard_id=shard_id,
            workdir=FLAGS.workdir,
            force_download=FLAGS.config.force_download,
        )
    n_batch_per_subshard = count_batches(
        shard=ds_shard,
        n_subshard=n_host_per_shard,
        batch_size=batch_size_per_host,
        sequence_len=FLAGS.config.sequence_len,
    )

    global_mesh = global_mesh_factory()
    acc = None
    log_level_is_debug = logging.get_verbosity() == 1
    start_time = time.perf_counter()
    for i in range(n_batch_per_subshard):
        logging.info(f"eval step {i}...")
        batch = get_batch(
            shard=ds_shard,
            n_subshard=n_host_per_shard,
            subshard_id=subshard_id,
            batch_size=batch_size_per_host,
            sequence_len=FLAGS.config.sequence_len,
            step=i,
        )
        logging.debug("Got batch...")
        logging.debug(f"Batch shape (local): {batch.shape}")

        # distribute local batch arrays to global batch arrays
        logging.debug("Distributing batch to global array...")
        batch = to_global_array(batch, global_mesh)
        batch = jax.block_until_ready(batch) if log_level_is_debug else batch
        logging.debug("Finished distributing batch to global array...")
        logging.debug(f"Batch shape (global): {batch.shape}")

        logging.debug("Starting eval step...")
        stats = eval_step(params=params, batch=batch)
        stats = jtu.tree_map(lambda a: jax.device_get(a).item(), stats)
        stats = jax.block_until_ready(stats)  # slows a bit, but makes printout accurate
        logging.debug("Finished eval step...")

        logging.debug("Starting accumulation step...")
        if acc is not None:
            acc = jtu.tree_map(
                lambda a, b: (i / (i + 1)) * a + (1 / (i + 1)) * b, acc, stats
            )
        else:
            acc = stats
        if logging.get_verbosity() == 1:
            acc = jax.block_until_ready(acc)
            logging.debug("Finished accumulation step...")

        if n_eval_step is not None:
            if i + 1 == n_eval_step:
                break

    logging.debug("Eval loop finished...")
    logging.debug("Computing eval metrics...")
    acc = jax.block_until_ready(acc)
    end_time = time.perf_counter()
    eval_metrics = dict(
        loss_avg=acc["loss_term_avg"] / acc["loss_mask_avg"],
        secs_per_step=(end_time - start_time) / (i + 1),
        **acc,
    )
    logging.debug("Finished computing eval metrics...")
    return eval_metrics


def apply_sampler(logits):
    if FLAGS.config.sampling_method == "nucleus":
        return apply_nucleus(logits, FLAGS.config.sampling_nucleus)
    if FLAGS.config.sampling_method == "topk":
        return apply_topk(logits, FLAGS.config.sampling_topk)
    if FLAGS.config.sampling_method == "temp":
        return apply_temp(logits, FLAGS.config.sampling_temp)
    raise NotImplementedError


def sample_step(carry, _):
    config = transformer_config_factory(is_train=False, is_decoding=True)
    global_mesh = global_mesh_factory()
    params = carry["params"]
    prev_token = carry["prev_token"]
    cache = carry["cache"]
    rng = carry["rng"]
    rng_new, rng_step = jax.random.split(rng)
    out = Transformer(config, global_mesh).apply({"params": params}, prev_token, cache)
    logits = apply_sampler(out["logits"])
    curr_token = jax.random.categorical(rng_step, logits, axis=-1)
    carry_new = dict(
        params=params,
        prev_token=curr_token,
        cache=out["kv_cache"],
        rng=rng_new,
    )
    return carry_new, curr_token


@jax.jit
def sample_sequence(rng, params, prompts):
    cfg = transformer_config_factory(is_train=False, is_decoding=False)
    # worked example:
    # seqlen = 5, npad = 2.
    # the prompt "a b c <pad> <pad>".
    # becomes "<bos> a b c <pad>".
    prefill = Transformer(cfg, global_mesh_factory()).apply(
        {"params": params},
        jnp.pad(prompts[:, :-1], ((0, 0), (1, 0)), constant_values=cfg.bos_token_id),
    )
    # worked example, continued:
    # the prefill step returns pos_id = slen - npad = 4 (since 1 non-pad tokens at end).
    # we need to sample one token, the output for "c", to init the main sampling loop.
    # this requires extracting the logits from index pos_id - 1 = 3, and sampling those.
    rng, rng_sample = jax.random.split(rng)
    first_logits = jnp.take_along_axis(
        arr=prefill["logits"],  # [B, C, V]
        indices=prefill["kv_cache"]["pos_ids"][0][..., None, None] - 1,  # [B, 1, 1]
        axis=1,
    )
    first_logits = apply_sampler(first_logits)
    first_output_token = jax.random.categorical(
        key=rng_sample,
        logits=first_logits,
        axis=-1,
    )
    init = dict(
        params=params,
        prev_token=first_output_token,
        cache=prefill["kv_cache"],
        rng=rng,
    )
    # and now for the main loop
    _, tokens = jax.lax.scan(
        f=sample_step,
        init=init,
        xs=jnp.arange(FLAGS.config.sampling_max_len - 1),
        length=FLAGS.config.sampling_max_len - 1,
        unroll=1,
    )
    # then we concatenate the samples together and reshape to correct shape,
    # since jax.lax.scan (unlike flax.linen.scan) can only be applied along leading axis
    tokens = jnp.squeeze(tokens, -1)
    tokens = jnp.transpose(tokens, (1, 0))
    tokens = jnp.concatenate([first_output_token, tokens], axis=-1)
    # lastly, we overwrite with '<pad>' every token slot at/following generated pad/eos
    keep = jnp.cumprod(
        jnp.logical_and(
            jnp.not_equal(tokens, cfg.eos_token_id),
            jnp.not_equal(tokens, cfg.pad_token_id),
        ),
        axis=-1,
    )
    tokens = keep * tokens + (1 - keep) * cfg.pad_token_id
    return tokens, keep


def sampling_loop():
    rng_init, rng_stoch = get_rngs()

    logging.info("Creating train state...")
    state = train_state_factory(rng_init)
    load_checkpoint_mgr = checkpoint_manager_factory(option="load")
    if load_checkpoint_mgr.latest_step() is not None:
        state = do_restore(load_checkpoint_mgr, state)
    tokenizer = tokenizer_factory()

    n_host = jax.process_count()
    host_id = jax.process_index()
    n_shard = FLAGS.config.n_ds_shard
    n_host_per_shard = n_host // n_shard  # n_ds_shard = n_host on smallest runs
    global_batch_size = global_batch_size_factory()
    batch_size_per_host = global_batch_size // n_host
    shard_id = host_id // n_host_per_shard
    subshard_id = host_id % n_host_per_shard
    ds_shard = get_dataset(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hfds_buffer_size=FLAGS.config.hfds_buffer_size,
        hftr_tokenizer=tokenizer,
        split_name="validation",
        batch_size=batch_size_per_host,
        sequence_len=FLAGS.config.sequence_len,
        n_shard=n_shard,
        shard_id=shard_id,
        workdir=FLAGS.workdir,
        force_download=FLAGS.config.force_download,
    )
    batch = get_batch(
        shard=ds_shard,
        n_subshard=n_host_per_shard,
        subshard_id=subshard_id,
        batch_size=batch_size_per_host,
        sequence_len=FLAGS.config.sequence_len,
        step=0,
    )
    slen, plen = FLAGS.config.sequence_len, FLAGS.config.sampling_prompt_len
    batch = jnp.pad(
        array=batch[:, 0:plen],
        pad_width=((0, 0), (0, slen - plen)),
        constant_values=tokenizer_factory().pad_token_id,
    )
    batch = to_global_array(batch, global_mesh_factory())

    out, _ = sample_sequence(rng_stoch, state.params, batch)
    out = jmhu.process_allgather(out)
    out_text = [tokenizer.decode(out[i].tolist()) for i in range(global_batch_size)]

    batch = jmhu.process_allgather(batch)
    batch_text = [tokenizer.decode(batch[i].tolist()) for i in range(global_batch_size)]

    for i in range(len(out_text)):
        print("-" * 80)
        print("PROMPT:")
        print(batch_text[i])
        print("CONTINUATION:")
        print(out_text[i])
        # print("")
        # print("PROMPT TOKEN IDS:")
        # print(batch[i])
        # print("CONTINUATION TOKEN IDS:")
        # print(out[i])
        # print("DONE MASK IDS:")
        # print(done_mask[i])
        # print("")
    # todo: we got high-quality samples, need to write to cloud still


def save_eval_loss():
    eval_loss = eval_loop(params=None, mode="validation")["loss_avg"]
    logging.info(f"Eval loss: {eval_loss}")
    if jax.process_index() == 0:
        table = wandb.Table(
            columns=[
                "Group",
                "RNG_Seed",
                "RNG_Fold",
                "Dtype",
                "Bsz",
                "LR",
                "WD",
                "Width",
                "Depth",
                "Nonlin",
                "Optim",
                "Rule",
                "Sched",
                "Steps",
                "Loss",
            ],
            data=[
                [
                    FLAGS.experiment_group,
                    FLAGS.rng_seed,
                    FLAGS.rng_fold,
                    FLAGS.config.dtype,
                    FLAGS.config.tokens_per_global_batch,
                    FLAGS.config.lr_base,
                    FLAGS.config.wd,
                    FLAGS.config.d_model,
                    FLAGS.config.n_layer,
                    FLAGS.config.ff_act_name,
                    FLAGS.config.optim_name,
                    FLAGS.config.optim_rule,
                    FLAGS.config.lr_schedule_name,
                    FLAGS.config.n_pretrain_step,
                    eval_loss,
                ],
            ],
        )
        wandb.log({"sweep_table": table})


def main(argv):
    del argv
    logging.info("=== Start of main() ===")
    logging.info(f"Python version: {sys.version.__repr__()}")
    jax.distributed.initialize()
    logging.info(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    logging.info("=== Flags: ===")
    logging.info(f"experiment_group: {FLAGS.experiment_group}")
    logging.info(f"workdir: {FLAGS.workdir}")
    logging.info(f"mode: {FLAGS.mode}")
    logging.info(f"rng_seed: {FLAGS.rng_seed}")
    logging.info(f"rng_fold: {FLAGS.rng_fold}")
    logging.info(f"wb_enabled: {FLAGS.wb_enabled}")
    logging.info(f"wb_run: {FLAGS.wb_run}")
    logging.info(f"load_suffix: {FLAGS.load_suffix}")
    logging.info(f"save_suffix: {FLAGS.save_suffix}")
    logging.info(f"verbosity: {FLAGS.verbosity}")
    logging.info("=== Config: ===")
    for k, v in vars(FLAGS.config)["_fields"].items():
        logging.info(f"{k}: {v}")
    assert FLAGS.config.d_model >= FLAGS.config.d_head
    assert FLAGS.config.d_model % FLAGS.config.d_head == 0

    n_host = jax.process_count()
    n_ds_shard = FLAGS.config.n_ds_shard
    assert n_host >= n_ds_shard
    assert n_host % n_ds_shard == 0

    n_device = jax.device_count()
    n_example = global_batch_size_factory()
    d_model = FLAGS.config.d_model
    d_ff = FLAGS.config.d_model * FLAGS.config.ff_multiple
    n_head = FLAGS.config.d_model // FLAGS.config.d_head
    n_row = FLAGS.config.n_mesh_rows
    n_col = FLAGS.config.n_mesh_cols
    assert n_row * n_col == n_device

    # weight shape constraints
    assert d_model % n_row == 0
    assert n_head % n_col == 0
    assert d_model % n_row == 0
    assert d_ff % n_col == 0

    # activation shape constraints
    assert n_example >= n_device  # dataloader quirk
    assert n_example % n_row == 0  # parallelize batch across rows
    assert d_model % n_col == 0  # parallelize residuals across columns
    assert n_head % n_col == 0  # parallelize heads across columns
    assert d_ff % n_col == 0  # parallelize mlp hiddens across columns

    logging.info("Creating W&B connection...")
    if jax.process_index() == 0:
        wandb.init(
            project="mu_transformer_experimental_improvements",
            group=FLAGS.experiment_group,
            config={
                **vars(FLAGS.config)["_fields"],
                "rng_seed": FLAGS.rng_seed,
                "rng_fold": FLAGS.rng_fold,
            },
            resume="never" if FLAGS.wb_run is None else "must",
            mode="online" if FLAGS.wb_enabled else "disabled",
            id=FLAGS.wb_run,
        )

    if FLAGS.mode == "train":
        if FLAGS.config.is_sweep:
            done_fn = "done.txt"
            done_fp = posixpath.join(modeldir_factory("load", "checkpoints"), done_fn)
            local_done_fp = f"/tmp/{done_fn}"
            if not blobfile.exists(done_fp):
                if os.path.exists(local_done_fp):
                    os.remove(local_done_fp)
                train_loop()
                save_eval_loss()
                os.mknod(local_done_fp)
                blobfile.copy(local_done_fp, done_fp, overwrite=True)
                os.remove(local_done_fp)
        else:
            train_loop()
    elif FLAGS.mode in {"validation", "test"}:
        eval_metrics = eval_loop(params=None, n_eval_step=None, mode=FLAGS.mode)
        eval_loss = eval_metrics["loss_avg"]
        logging.info(f"Eval loss: {eval_loss}")
    elif FLAGS.mode == "sample":
        sampling_loop()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
