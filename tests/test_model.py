import jax
import jax.experimental.mesh_utils as jmu
import jax.numpy as jnp
import numpy as np
import pytest

from mu_transformer.jax_impl.model import RotaryEncoding
from mu_transformer.jax_impl.model import RotaryEncodingV2
from mu_transformer.jax_impl.model import Transformer
from mu_transformer.jax_impl.model import TransformerConfig


def test_rope_equivalence():
    bsz = 32
    conf = TransformerConfig.create(
        param_dtype=jnp.float64,
        dtype=jnp.float64,
        sequence_len=100,
        d_model=512,
        d_head=128,
        ff_multiple=4,
        e_norm=False,
        q_init="vs",
        r_init="vs",
        u_init="sp",
        qk_scale=1 / 128,
        qk_norm=False,
        kv_mqa=False,
        rotary_base=10_000,
        act_name="relu",
        act_square=False,
        norm_eps=1e-5,
        norm_gains=False,
        norm_gains_type="vector",
        proj_biases=False,
        n_layer=3,
        n_vocab=10,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        is_train=True,
        is_decoding=False,
    )
    n_mesh_rows, n_mesh_cols = 1, 1
    mesh = jax.sharding.Mesh(
        devices=jmu.create_device_mesh(
            mesh_shape=(n_mesh_rows, n_mesh_cols),
            devices=jax.devices(),
        ),
        axis_names=("X", "Y"),  # using 2D-finalized from GSPMD paper
    )

    n_head = conf.d_model // conf.d_head
    q = jax.random.normal(
        jax.random.PRNGKey(0), shape=(bsz, n_head, conf.sequence_len, conf.d_head)
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), shape=(bsz, n_head, conf.sequence_len, conf.d_head)
    )

    # test equivalence with original implementation when no position offset is applied
    qr1 = RotaryEncoding(conf, mesh, is_keys=False).apply({"params": {}}, q)
    kr1 = RotaryEncoding(conf, mesh, is_keys=True).apply({"params": {}}, k)
    qr2 = RotaryEncodingV2(conf, mesh, is_keys=False).apply(
        {"params": {}}, q, pos_ids=jnp.zeros([bsz])
    )
    kr2 = RotaryEncodingV2(conf, mesh, is_keys=True).apply(
        {"params": {}}, k, pos_ids=jnp.zeros([bsz])
    )
    np.testing.assert_allclose(qr1, qr2)
    np.testing.assert_allclose(kr1, kr2)

    # test the shift equivariance property
    qr1 = RotaryEncoding(conf, mesh, is_keys=False).apply({"params": {}}, q)
    kr1 = RotaryEncoding(conf, mesh, is_keys=True).apply({"params": {}}, k)
    sr1 = jnp.einsum("bhid,bhjd->bhij", qr1, kr1) * conf.qk_scale
    shifts = jax.random.randint(
        jax.random.PRNGKey(2), shape=(bsz,), minval=0, maxval=conf.sequence_len * 100
    )
    qr2 = RotaryEncodingV2(conf, mesh, is_keys=False).apply(
        {"params": {}}, q, pos_ids=shifts
    )
    kr2 = RotaryEncodingV2(conf, mesh, is_keys=True).apply(
        {"params": {}}, k, pos_ids=shifts
    )
    sr2 = jnp.einsum("bhid,bhjd->bhij", qr2, kr2) * conf.qk_scale
    np.testing.assert_allclose(sr1, sr2, atol=1e-4, rtol=1e-4)


def test_decoding_equivalence():
    # test that parallel-mode and teacher-forced decoding produce equivalent logits
    bsz = 32
    common = dict(
        param_dtype=jnp.float64,
        dtype=jnp.float64,
        sequence_len=100,
        d_model=512,
        d_head=128,
        ff_multiple=4,
        e_norm=False,
        q_init="vs",
        r_init="vs",
        u_init="sp",
        qk_scale=1 / 128,
        qk_norm=False,
        kv_mqa=False,
        rotary_base=10_000,
        act_name="relu",
        act_square=False,
        norm_eps=1e-5,
        norm_gains=False,
        norm_gains_type="vector",
        proj_biases=False,
        n_layer=3,
        n_vocab=32000,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        is_train=True,
    )
    conf = TransformerConfig.create(**common, is_decoding=False)
    n_mesh_rows, n_mesh_cols = 1, 1
    mesh = jax.sharding.Mesh(
        devices=jmu.create_device_mesh(
            mesh_shape=(n_mesh_rows, n_mesh_cols),
            devices=jax.devices(),
        ),
        axis_names=("X", "Y"),  # using 2D-finalized from GSPMD paper
    )

    rng = jax.random.PRNGKey(0)
    rng_init, rng_tokens = jax.random.split(rng)
    tokens = jax.random.randint(
        rng_tokens, shape=[bsz, conf.sequence_len], minval=3, maxval=conf.n_vocab
    )
    params = Transformer(conf, mesh).init({"params": rng_init}, tokens)["params"]
    parallel_outputs = Transformer(conf, mesh).apply({"params": params}, tokens)
    parallel_logits = parallel_outputs["logits"]
    parallel_cache = parallel_outputs["kv_cache"]

    prefill_end = conf.sequence_len // 2
    prefill_outputs = Transformer(conf, mesh).apply(
        {"params": params}, tokens[:, 0:prefill_end]
    )
    prefill_logits = prefill_outputs["logits"]
    prefill_cache = prefill_outputs["kv_cache"]

    decode_conf = TransformerConfig.create(**common, is_decoding=True)
    decode_logits = []
    cache = prefill_cache
    for t in range(prefill_end, conf.sequence_len):
        out = Transformer(decode_conf, mesh).apply(
            {"params": params}, tokens[:, t : t + 1]
        )
        decode_logits.append(out["logits"])
        cache = out["kv_cache"]
    decode_logits = jnp.concatenate(decode_logits, axis=1)

    np.testing.assert_allclose(parallel_logits[:, :prefill_end, :], prefill_logits)
    np.testing.assert_allclose(parallel_logits[:, prefill_end:, :], decode_logits)
    np.testing.assert_allclose(parallel_cache["k"], cache["k"])
    np.testing.assert_allclose(parallel_cache["v"], cache["v"])
    np.testing.assert_allclose(parallel_cache["pos_ids"], cache["pos_ids"])
