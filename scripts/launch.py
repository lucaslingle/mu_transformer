import functools
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
import wandb
from absl import app
from absl import flags
from absl import logging
from etils import epath
from flax import jax_utils
from flax import traverse_util
from flax.training import common_utils
from flax.training import orbax_utils
from flax.training import train_state
from ml_collections import config_flags

from mu_transformer.data import get_dataset
from mu_transformer.data import get_tokenizer
from mu_transformer.model import Transformer
from mu_transformer.model import TransformerConfig


MODES = ["train", "val", "test"]
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration file", lock_config=False)
flags.DEFINE_string("workdir", None, "Working directory (local or GCS)")
flags.DEFINE_boolean("wb_enabled", False, "Log to W&B")
flags.DEFINE_string("wb_run", None, "W&B run id, for resuming with continuity")
flags.DEFINE_enum("mode", None, MODES, "Mode")
flags.DEFINE_string("load_name", None, "Model name to load; None = use autogen")
flags.DEFINE_string("save_name", None, "Model name to save; None = use autogen")
flags.mark_flags_as_required(["config", "workdir", "mode"])


@functools.lru_cache(maxsize=1)
def tokenizer_factory():
    return get_tokenizer(
        FLAGS.config.hftr_tokenizer_name,
        FLAGS.config.hftr_tokenizer_shortname,
    )


@functools.lru_cache(maxsize=2)
def transformer_config_factory(is_train):
    return TransformerConfig.create(
        **vars(FLAGS.config)["_fields"],
        n_vocab=tokenizer_factory().vocab_size,
        is_train=is_train,
    )


def params_factory(rng, is_train):
    config = transformer_config_factory(is_train)
    inputs = jnp.ones(dtype=jnp.int32, shape=[1, config.sequence_len])
    params = Transformer(config).init({"params": rng}, inputs)["params"]
    params_count = sum(x.size for x in jtu.tree_leaves(params))
    logging.info(f"Param count: {params_count}")
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


def train_state_factory(rng, is_train):
    return train_state.TrainState.create(
        apply_fn=None,
        params=params_factory(rng, is_train),
        tx=grad_transform_factory(),
    )


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
        FLAGS.config.hfds_identifier.split("/")[-1].lower(),
        FLAGS.config.model_size,
        f"t{FLAGS.config.sequence_len}",
        f"s{FLAGS.config.n_pretrain_step}",
    ]
    return "_".join(parts)


def modelname_factory(option):
    if option == "save":
        if FLAGS.save_name is not None:
            return FLAGS.save_name
        else:
            return automatic_modelname_factory()
    elif option == "load":
        if FLAGS.load_name is not None:
            return FLAGS.load_name
        else:
            return automatic_modelname_factory()


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


def loss_fn(params, batch, is_train):
    config = transformer_config_factory(is_train)
    args = [{"params": params}, batch["inputs"]]
    if FLAGS.config.sow_intermediates:
        logits, mvars = Transformer(config).apply(*args, mutable="intermediates")
        intermediates = mvars["intermediates"]
        sown_metrics, _ = jax.tree_util.tree_flatten_with_path(intermediates["stack"])
        sown_metrics = {k[-2].key: jnp.mean(v) for k, v in sown_metrics}  # layer avg
    else:
        logits = Transformer(config).apply(*args)
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


@functools.partial(jax.pmap, donate_argnums=(0,), axis_name="devices")
def train_step(state, batch, rng):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params,
        batch=batch,
        is_train=True,
    )
    metrics, grads = jax.lax.pmean([metrics, grads], axis_name="devices")
    state = state.apply_gradients(grads=grads)
    metrics["loss_avg"] = metrics["loss_term_avg"] / metrics["loss_mask_avg"]
    return state, metrics


def train_loop(rng):
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
    rng_init, rng_stoch = jax.random.split(rng)
    rng_stoch = jax.random.fold_in(rng_stoch, jax.process_index())

    logging.log(log_level, "Creating train state...")
    load_checkpoint_mgr = checkpoint_manager_factory(option="load")
    state = train_state_factory(rng_init, is_train=True)
    if load_checkpoint_mgr.latest_step() is not None:
        state = do_restore(load_checkpoint_mgr, state)
    start_step = load_checkpoint_mgr.latest_step() or 0
    state = jax.block_until_ready(jax_utils.replicate(state))
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
    # use n_finetune_step > 0 if and only if currently fine-tuning.
    n_total_step = FLAGS.config.n_pretrain_step + FLAGS.config.n_finetune_step
    best_val_loss = float("inf")
    val_loss = None
    save_checkpoint_mgr = checkpoint_manager_factory(option="save")
    start_time = time.perf_counter()
    for step in range(start_step, n_total_step + 1):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = get_dataset(start_step=step, **batch_iter_kwargs)
            batch = next(batch_iter)
        state, metrics = train_step(
            state,
            common_utils.shard(batch),
            common_utils.shard_prng_key(jax.random.fold_in(rng_stoch, step)),
        )
        if step % FLAGS.config.n_print_step == 0:
            metrics = jax_utils.unreplicate(metrics)
            metrics = jax.block_until_ready(metrics)
            end_time = time.perf_counter()
            extras = dict(
                step=step,
                val_loss_avg=val_loss,
                sec_per_step=(end_time - start_time) / FLAGS.config.n_print_step,
            )
            metrics.update(extras)
            logging.info(metrics)
            if jax.process_index() == 0:
                wandb.log(metrics)
            start_time = end_time
        if (step % FLAGS.config.n_save_step == 0) or step == n_total_step:
            val_loss = eval_loop(
                params=state.params,
                rng=jax.random.fold_in(rng_stoch, step),
                step=step,
            )
            if best_val_loss > val_loss:
                do_save(save_checkpoint_mgr, step, jax_utils.unreplicate(state))
                best_val_loss = val_loss


@functools.partial(jax.pmap, axis_name="devices")
def eval_step(params, batch, rng):
    _, metrics = loss_fn(params, batch, is_train=False)
    return jax.lax.pmean(metrics, axis_name="devices")


def eval_loop(params, rng, step):
    logging.info("Entering eval loop function...")
    if params is None:
        rng, rng_init = jax.random.split(rng)
        logging.info("Creating params...")
        load_checkpoint_mgr = checkpoint_manager_factory(option="load")
        state = train_state_factory(rng_init, is_train=False)
        if load_checkpoint_mgr.latest_step() is not None:
            state = do_restore(load_checkpoint_mgr, state)
        del load_checkpoint_mgr
        state = jax.block_until_ready(jax_utils.replicate(state))
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
        start_step=step,
        shuffle=False,
    )

    logging.info("Starting eval loop...")
    loss_term_avg = 0.0
    loss_mask_avg = 0.0
    count = 0
    for batch in batch_iter:
        logging.info(f"eval step {count}...")
        if (FLAGS.mode == "train") and (count == FLAGS.config.n_eval_step):
            break
        metrics = eval_step(
            params,
            common_utils.shard(batch),
            common_utils.shard_prng_key(jax.random.fold_in(rng, count)),
        )
        loss_term_avg += (1 / (count + 1)) * (metrics["loss_term_avg"] - loss_term_avg)
        loss_mask_avg += (1 / (count + 1)) * (metrics["loss_mask_avg"] - loss_mask_avg)
        count += 1
    loss_avg = loss_term_avg / loss_mask_avg
    return jax.block_until_ready(jax_utils.unreplicate(loss_avg))


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

    try:
        jax.distributed.initialize()
    except Exception:
        logging.warning("Jax distributed did not init successfully.")

    if FLAGS.mode == "train":
        train_loop(jax.random.PRNGKey(0))
    elif FLAGS.mode in {"validation", "test"}:
        eval_loss = eval_loop(None, jax.random.PRNGKey(0), 0)
        logging.info(f"Eval loss: {eval_loss}")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
