import functools
import time

import flax.core
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
from mu_transformer.sharding import get_namedsharding
from mu_transformer.sharding import to_global_array

MODES = ["train", "val", "test"]
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration file", lock_config=False)
flags.DEFINE_string("workdir", None, "Working directory (local or GCS)")
flags.DEFINE_boolean("wb_enabled", False, "Log to W&B")
flags.DEFINE_string("wb_run", None, "W&B run id, for resuming with continuity")
flags.DEFINE_enum("mode", None, MODES, "Mode")
flags.DEFINE_string("load_name", None, "Model name to load; None = use autogen")
flags.DEFINE_string("save_name", None, "Model name to save; None = use autogen")
flags.DEFINE_integer("experiment_seed", 0, "Experiment seed")
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
            mesh_shape=(FLAGS.config.n_shard_data, FLAGS.config.n_shard_model),
            devices=jax.devices(),
        ),
        axis_names=("data", "model"),
    )


def params_factory(rng, model_cls):
    config = transformer_config_factory()
    global_mesh = global_mesh_factory()
    inputs = jnp.ones(dtype=jnp.int32, shape=[1, config.sequence_len])
    params = model_cls(config, global_mesh).init({"params": rng}, inputs)["params"]
    if isinstance(params, flax.core.FrozenDict):
        params = params.unfreeze()
    return params


def param_label_fn(params):
    flat = traverse_util.flatten_dict(params)
    flat_labels = {k: k[-1].split("_")[-1] for k, v in flat.items()}
    return traverse_util.unflatten_dict(flat_labels)


def schedule_factory():
    warmup_steps = FLAGS.config.n_warmup_step
    decay_steps = FLAGS.config.n_pretrain_step - FLAGS.config.n_warmup_step  # const aft
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, end_value=1.0, transition_steps=warmup_steps),
            optax.cosine_decay_schedule(1.0, alpha=0.1, decay_steps=decay_steps),
        ],
        boundaries=[warmup_steps],
    )


def grad_transform_factory():
    kwargs = dict(
        b1=FLAGS.config.adam_b1,
        b2=FLAGS.config.adam_b2,
        eps=FLAGS.config.adam_eps,
        mu_dtype=FLAGS.config.param_dtype,
        weight_decay=FLAGS.config.wd_lam,
    )
    return optax.chain(
        optax.clip_by_global_norm(FLAGS.config.grad_clip),
        optax.multi_transform(
            {
                "fi": optax.adamw(FLAGS.config.lr_max, **kwargs),
                "ii": optax.adamw(FLAGS.config.lr_max / FLAGS.config.d_model, **kwargs),
                "if": optax.adamw(FLAGS.config.lr_max / FLAGS.config.d_model, **kwargs),
            },
            param_labels=param_label_fn,
        ),
        optax.scale_by_schedule(schedule_factory()),
    )


def train_state_factory(rng_init):
    # based on https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#the-output-s-sharding  # noqa
    model_cls = Transformer
    optimizer_cls = grad_transform_factory()

    def init_fn(rng, model_cls_, optim_cls_):
        return train_utils.TrainState.create(
            apply_fn=None,
            params=params_factory(rng, model_cls_),
            tx=optim_cls_,
        )

    global_mesh = global_mesh_factory()
    prng_sharding = get_namedsharding(axis_names=(None,), device_mesh=global_mesh)
    abstract_variables = jax.eval_shape(
        functools.partial(init_fn, model_cls_=model_cls, optim_cls_=optimizer_cls),
        rng_init,
    )
    state_sharding = nn.get_sharding(abstract_variables, global_mesh)
    jit_init_fn = jax.jit(
        init_fn,
        static_argnums=(1, 2),
        in_shardings=(prng_sharding,),
        out_shardings=state_sharding,
    )
    initialized_state = jit_init_fn(rng_init, model_cls, optimizer_cls)
    return initialized_state


def global_batch_size_factory():
    assert FLAGS.config.tokens_per_global_batch % FLAGS.config.sequence_len == 0
    global_bsz = FLAGS.config.tokens_per_global_batch // FLAGS.config.sequence_len
    assert global_bsz % jax.process_count() == 0
    return global_bsz


def automatic_modelname_factory():
    dataset_name = FLAGS.config.hfds_identifier.split("/")[-1].lower()
    assert dataset_name.isalnum()
    parts = [
        "mutransformer",
        dataset_name,
        FLAGS.config.model_size,
        f"b{global_batch_size_factory()}",
        f"t{FLAGS.config.sequence_len}",
        f"s{FLAGS.config.n_pretrain_step}",
        f"r{FLAGS.experiment_seed}",
    ]
    return "_".join(parts)


def modelname_factory(option):
    if option == "save":
        return FLAGS.save_name or automatic_modelname_factory()
    if option == "load":
        return FLAGS.load_name or automatic_modelname_factory()
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


def loss_fn(params, batch):
    # todo: support sown intermediates
    # if FLAGS.config.sow_intermediates:
    #     logits, mvars = Transformer(*i_args).apply(*c_args, mutable="intermediates")
    #     intermediates = mvars["intermediates"]
    #     sown_metrics, _ = jtu.tree_flatten_with_path(intermediates["stack"])
    #     sown_metrics = {k[-2].key: jnp.mean(v) for k, v in sown_metrics}  # layer avg
    # else:
    #     logits = Transformer(i_args).apply(*c_args)
    #     sown_metrics = dict()
    config = transformer_config_factory()
    global_mesh = global_mesh_factory()
    logits = Transformer(config, global_mesh).apply({"params": params}, batch["inputs"])
    sown_metrics = dict()

    loss_terms = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch["targets"],
    )
    loss_metrics = dict(
        loss_term_avg=jnp.mean(batch["loss_mask"] * loss_terms),
        loss_mask_avg=jnp.mean(batch["loss_mask"]),
    )
    return loss_metrics["loss_term_avg"], dict(**loss_metrics, **sown_metrics)


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    # no extra mean anywhere, we already have the sharded all-device mean gradient!
    metrics["param_count"] = size_pytree(state.params)  # so it's always visible
    metrics["param_norm"] = l2norm_pytree(state.params)
    metrics["grad_norm"] = l2norm_pytree(grads)
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
            project=modelname_factory(option="save"),
            config=vars(FLAGS.config)["_fields"],
            resume="never" if FLAGS.wb_run is None else "must",
            mode="online" if FLAGS.wb_enabled else "disabled",
            id=FLAGS.wb_run,
        )
    logging.log(log_level, "Creating RNGs...")
    rng_init, rng_stoch = jax.random.split(jax.random.PRNGKey(FLAGS.experiment_seed))
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
    _, metrics = loss_fn(params, batch)
    return metrics


def eval_loop(params, n_eval_step=None):
    logging.info("Entering eval loop function...")
    if params is None:
        rng_init, _ = jax.random.split(jax.random.PRNGKey(FLAGS.experiment_seed))
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
        stats = jax.block_until_ready(stats)  # slows a bit, but makes printout accurate
        if acc is not None:
            acc = jtu.tree_map(lambda a, b: a + b, stats, acc)
        else:
            acc = stats
        if n_eval_step is not None:
            if i + 1 == n_eval_step:
                break

    acc = jtu.tree_map(lambda a: jax.device_get(a).item() / (i + 1), acc)
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
    logging.info(f"wb_enabled: {FLAGS.wb_enabled}")
    logging.info(f"wb_run: {FLAGS.wb_run}")
    logging.info(f"mode: {FLAGS.mode}")
    logging.info(f"load_name: {FLAGS.load_name}")
    logging.info(f"save_name: {FLAGS.save_name}")
    logging.info("=== Config: ===")
    for k, v in vars(FLAGS.config)["_fields"].items():
        logging.info(f"{k}: {v}")
    assert FLAGS.config.d_model >= 128
    assert FLAGS.config.d_model % 128 == 0
    assert FLAGS.n_shard_data * FLAGS.n_shard_model == jax.device_count()
    assert FLAGS.config.global_batch_size >= jax.device_count()  # dataloader quirk
    assert FLAGS.config.global_batch_size % FLAGS.n_shard_data == 0

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
