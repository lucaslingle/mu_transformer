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
import re
import time

import flax.linen as nn
import jax
import jax.experimental.mesh_utils as jmu
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
import wandb
from absl import app
from absl import flags
from absl import logging
from etils import epath
from flax import traverse_util
from flax.training import orbax_utils
from flax.training import train_state as train_utils
from ml_collections import config_flags

from mu_transformer.data import get_dataset
from mu_transformer.data import get_tokenizer
from mu_transformer.model import Transformer
from mu_transformer.model import TransformerConfig
from mu_transformer.shard import get_namedsharding
from mu_transformer.shard import MESH_AXES
from mu_transformer.shard import sharding_constraint
from mu_transformer.shard import to_global_array
from mu_transformer.sow import split_coord_checks


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration file", lock_config=False)
flags.DEFINE_string("workdir", None, "Working directory (local or GCS)")
flags.DEFINE_enum("mode", None, ["train", "val", "test"], "Mode")
flags.DEFINE_integer("seed", 0, "Experiment seed")
flags.DEFINE_boolean("wb_enabled", False, "Log to W&B")
flags.DEFINE_string("wb_run", None, "W&B run id, for resuming with continuity")
flags.DEFINE_string("loading_name", None, "Model name to load; None = use autogen")
flags.DEFINE_string("saving_name", None, "Model name to save; None = use autogen")
flags.mark_flags_as_required(["config", "workdir", "mode"])


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
    )


@functools.lru_cache(maxsize=1)
def global_mesh_factory():
    return jax.sharding.Mesh(
        devices=jmu.create_device_mesh(
            mesh_shape=(
                FLAGS.config.n_mesh_rows,
                FLAGS.config.n_mesh_cols,
                FLAGS.config.n_mesh_planes,
            ),
            devices=jax.devices(),
        ),
        axis_names=("rows", "columns", "planes"),
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
    warmup_steps = FLAGS.config.n_warmup_step
    decay_steps = FLAGS.config.n_pretrain_step - FLAGS.config.n_warmup_step  # const aft
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, end_value=1.0, transition_steps=warmup_steps),
            optax.linear_schedule(1.0, end_value=0.1, transition_steps=decay_steps),
        ],
        boundaries=[warmup_steps],
    )


def grad_transform_factory():
    kws = dict(
        b1=FLAGS.config.adam_b1,
        b2=FLAGS.config.adam_b2,
        eps=FLAGS.config.adam_eps,
        mu_dtype=FLAGS.config.param_dtype,
        weight_decay=0.0,  # done later
    )
    lr = FLAGS.config.lr_max
    wd = FLAGS.config.wd_lam
    dm = FLAGS.config.d_model
    dff = FLAGS.config.d_model * FLAGS.config.ff_multiple
    return optax.chain(
        optax.clip_by_global_norm(FLAGS.config.grad_clip),
        optax.multi_transform(
            {
                # embeddings and de-embeddings
                "w_ei": optax.adamw(lr, **kws),  # table 3, col 1
                "w_eo": optax.adamw(lr / dm, **kws),  # table 3, col2
                # attention projections
                "w_aq": optax.adamw(lr / dm, **kws),  # table 3, col3
                "w_ak": optax.adamw(lr / dm, **kws),  # table 3, col3
                "w_av": optax.adamw(lr / dm, **kws),  # table 3, col3
                "w_ao": optax.adamw(lr / dm, **kws),  # table 3, col3; assumes dm=nh*dh
                # feed-forward projections
                "w_fi": optax.adamw(lr / dm, **kws),  # table 3, col3
                "w_fo": optax.adamw(lr / dff, **kws),  # table 3, col3
            },
            param_labels=param_label_fn,
        ),
        optax.add_decayed_weights(-lr * wd),  # mult by master lr * schedule; not width
        optax.scale_by_schedule(schedule_factory()),
    )


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
    config = transformer_config_factory()
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
    parts = [
        "mutransformer",
        dataset_name,
        FLAGS.config.model_size,
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


def checkpoint_manager_factory(option):
    model_name = modelname_factory(option)
    return ocp.CheckpointManager(
        directory=epath.Path(FLAGS.workdir) / model_name / "checkpoints",
        checkpointers=ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=5,
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


def loss_fn(params, batch, config, global_mesh):
    init_args = [config, global_mesh]
    apply_args = [{"params": params}, batch["inputs"]]
    if FLAGS.config.sow_intermediates:
        logits, mv = Transformer(*init_args).apply(*apply_args, mutable="intermediates")
        sown = traverse_util.flatten_dict(mv["intermediates"])
        sown = {k[-1]: split_coord_checks(k[-1], v[0]) for k, v in sown.items()}
        sown = traverse_util.flatten_dict(sown)
        sown = {k[-1]: v for k, v in sown.items()}
    else:
        logits = Transformer(*init_args).apply(*apply_args)
        sown = dict()
    terms = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=sharding_constraint(batch["targets"], MESH_AXES["RN"], global_mesh),
    )
    metrics = dict(
        loss_term_avg=jnp.mean(batch["loss_mask"] * terms),
        loss_mask_avg=jnp.mean(batch["loss_mask"]),
    )
    return metrics["loss_term_avg"], dict(**metrics, **sown)


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params,
        batch=batch,
        config=transformer_config_factory(),
        global_mesh=global_mesh_factory(),
    )
    # no extra mean anywhere, we already have the sharded all-device mean gradient!
    metrics["param_count"] = size_pytree(state.params)  # so it's always visible
    metrics["param_norm"] = l2norm_pytree(state.params)
    metrics["grad_norm"] = l2norm_pytree(grads)
    metrics["grad_nan"] = jnp.isnan(metrics["grad_norm"]).astype(jnp.int32)
    state = state.apply_gradients(grads=grads)
    # Estimate ce loss for global batch: sum of unmasked ce terms / sum of mask values.
    # Equivalently,
    loss_avg = metrics["loss_term_avg"] / metrics["loss_mask_avg"]
    return state, dict(loss_avg=loss_avg, **metrics)


def train_loop():
    log_level = logging.INFO
    logging.log(log_level, "Entering train loop function...")
    logging.log(log_level, "Creating W&B connection...")
    if jax.process_index() == 0:
        wandb.init(
            project="mu_transformer",
            config=vars(FLAGS.config)["_fields"],
            resume="never" if FLAGS.wb_run is None else "must",
            mode="online" if FLAGS.wb_enabled else "disabled",
            id=FLAGS.wb_run,
        )
    logging.log(log_level, "Creating RNGs...")
    rng_init, rng_stoch = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    rng_stoch = jax.random.fold_in(rng_stoch, jax.process_index())

    logging.log(log_level, "Creating train state...")
    load_checkpoint_mgr = checkpoint_manager_factory(option="load")
    state = train_state_factory(rng_init)
    if load_checkpoint_mgr.latest_step() is not None:
        state = do_restore(load_checkpoint_mgr, state)
    start_step = load_checkpoint_mgr.latest_step() or 0
    del load_checkpoint_mgr

    logging.log(log_level, "Creating dataset...")
    batch_iter_kwargs = dict(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hftr_tokenizer=tokenizer_factory(),
        split_name="train",
        batch_size=global_batch_size_factory() // jax.process_count(),
        sequence_len=FLAGS.config.sequence_len,
        shuffle=True,
    )
    batch_iter = get_dataset(start_step=start_step, **batch_iter_kwargs)

    logging.log(log_level, "Starting training loop...")
    best_val_loss = float("inf")
    val_metrics = dict()
    global_mesh = global_mesh_factory()
    save_checkpoint_mgr = checkpoint_manager_factory(option="save")
    start_time = time.perf_counter()
    # the user should set n_finetune_step > 0 if and only if currently fine-tuning.
    n_total_step = FLAGS.config.n_pretrain_step + FLAGS.config.n_finetune_step
    for step in range(start_step, n_total_step + 1):
        # get next batch, and reset at epoch end.
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = get_dataset(start_step=step, **batch_iter_kwargs)
            batch = next(batch_iter)

        # distribute local batch arrays to global batch arrays
        batch = jtu.tree_map(lambda y: to_global_array(y, global_mesh), batch)
        # run a training step
        state, metrics = train_step(state, batch=batch)

        # occasionally print metrics
        if step % FLAGS.config.n_print_step == 0:
            state = jax.block_until_ready(state)
            metrics = jtu.tree_map(lambda a: jax.device_get(a).item(), metrics)
            metrics = jax.block_until_ready(metrics)
            end_time = time.perf_counter()
            sec_per_step = (end_time - start_time) / FLAGS.config.n_print_step
            essentials = {
                "step": step,
                "sec_per_step": sec_per_step,
                "loss_avg": metrics.get("loss_avg"),
                "val_loss_avg": val_metrics.get("loss_avg"),
            }
            logging.info(essentials)
            if jax.process_index() == 0:
                metrics.update(essentials)
                wandb.log(metrics)
            start_time = end_time

        # occasionally perform an evaluation and save a checkpoint on improvement
        if (step % FLAGS.config.n_save_step == 0) or step == n_total_step:
            state = jax.block_until_ready(state)
            val_metrics = eval_loop(state.params, n_eval_step=FLAGS.config.n_eval_step)
            if best_val_loss > val_metrics["loss_avg"]:
                do_save(save_checkpoint_mgr, step, state)
                best_val_loss = val_metrics["loss_avg"]


@jax.jit
def eval_step(params, batch):
    _, metrics = loss_fn(
        params=params,
        batch=batch,
        config=transformer_config_factory(),
        global_mesh=global_mesh_factory(),
    )
    return metrics


def eval_loop(params, n_eval_step=None):
    logging.info("Entering eval loop function...")
    if params is None:
        rng_init, _ = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
        logging.info("Creating params...")
        load_checkpoint_mgr = checkpoint_manager_factory(option="load")
        state = train_state_factory(rng_init)
        if load_checkpoint_mgr.latest_step() is not None:
            state = do_restore(load_checkpoint_mgr, state)
        del load_checkpoint_mgr
        state = jax.block_until_ready(state)
        params = state.params
        del state

    logging.info("Creating dataset...")
    batch_iter = get_dataset(
        hfds_identifier=FLAGS.config.hfds_identifier,
        hfds_config=FLAGS.config.hfds_config,
        hfds_datacol=FLAGS.config.hfds_datacol,
        hftr_tokenizer=tokenizer_factory(),
        split_name="val" if FLAGS.mode == "train" else FLAGS.mode,
        batch_size=global_batch_size_factory() // jax.process_count(),
        sequence_len=FLAGS.config.sequence_len,
        start_step=0,
        shuffle=False,
    )

    start_time = time.perf_counter()
    acc = None
    for i, batch in enumerate(batch_iter):
        logging.info(f"eval step {i}...")
        stats = eval_step(params=params, batch=batch)
        stats = jtu.tree_map(lambda a: jax.device_get(a).item(), stats)
        stats = jax.block_until_ready(stats)  # slows a bit, but makes printout accurate
        if acc is not None:
            acc = jtu.tree_map(
                lambda a, b: (i / (i + 1)) * a + (1 / (i + 1)) * b, acc, stats
            )
        else:
            acc = stats
        if n_eval_step is not None:
            if i + 1 == n_eval_step:
                break

    acc = jax.block_until_ready(acc)
    end_time = time.perf_counter()
    eval_metrics = dict(
        loss_avg=acc["loss_term_avg"] / acc["loss_mask_avg"],
        secs_per_step=(end_time - start_time) / (i + 1),
        **acc,
    )
    return eval_metrics


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)

    logging.info("=== Start of main() ===")
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("=== Flags: ===")
    logging.info(f"workdir: {FLAGS.workdir}")
    logging.info(f"mode: {FLAGS.mode}")
    logging.info(f"seed: {FLAGS.seed}")
    logging.info(f"wb_enabled: {FLAGS.wb_enabled}")
    logging.info(f"wb_run: {FLAGS.wb_run}")
    logging.info(f"loading_name: {FLAGS.loading_name}")
    logging.info(f"saving_name: {FLAGS.saving_name}")
    logging.info("=== Config: ===")
    for k, v in vars(FLAGS.config)["_fields"].items():
        logging.info(f"{k}: {v}")
    assert FLAGS.config.d_model >= FLAGS.config.d_head
    assert FLAGS.config.d_model % FLAGS.config.d_head == 0
    n_device = jax.device_count()
    n_example = global_batch_size_factory()
    n_head = FLAGS.config.d_model // FLAGS.config.d_head
    d_model = FLAGS.config.d_model
    n_row = FLAGS.config.n_mesh_rows
    n_col = FLAGS.config.n_mesh_cols
    n_plane = FLAGS.config.n_mesh_planes
    assert n_row * n_col * n_plane == n_device
    assert n_example >= n_device  # dataloader quirk
    assert n_example % n_row == 0  # parallelize batch across rows
    assert d_model >= n_col  # parallelize residual activations across columns
    assert d_model % n_col == 0
    assert n_head >= n_plane  # parallelize hidden activations across planes
    assert n_head % n_plane == 0
    try:
        jax.distributed.initialize()
    except Exception:
        logging.warning("Jax distributed did not init successfully.")

    if FLAGS.mode == "train":
        train_loop()
    elif FLAGS.mode in {"val", "test"}:
        eval_metrics = eval_loop(params=None, n_eval_step=None)
        eval_loss = eval_metrics["loss_avg"]
        logging.info(f"Eval metrics: {eval_metrics}")
        logging.info(f"Eval loss: {eval_loss}")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
