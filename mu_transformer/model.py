from dataclasses import fields
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import struct
from flax.linen import partitioning as nn_partitioning

from mu_transformer.dims import Dimensions
from mu_transformer.shard import sharding_constraint
from mu_transformer.sow import coord_check_l1

HEAD_DIM = 128
FF_MULTIPLE = 4
INFTY_APPROX = 1e30


@struct.dataclass
class TransformerConfig:
    param_dtype: Any
    dtype: Any
    sequence_len: int
    d_model: int
    n_layer: int
    n_vocab: int
    rotary_base: int
    rotary_interp_q: bool
    rotary_interp_k: bool
    act_name: str
    act_square: bool

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        return cls(**filtered)


class RMSLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + 1e-8)[..., None]


class RotaryEncoder(nn.Module):
    rotary_base: float
    interpretable: bool

    @nn.compact
    def __call__(self, x):
        length = x.shape[-2]
        width = x.shape[-1]
        positions = jnp.arange(length)
        positions = positions[..., None]  # expand along width axis

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        ang_freqs = jnp.power(self.rotary_base, -dimensions / (width // 2))
        ang_freqs = ang_freqs[None, ...]  # expand along length axis

        # expand along leading axes, such as batch and head.
        while positions.ndim < x.ndim:
            positions = positions[None, ...]
            ang_freqs = ang_freqs[None, ...]

        radians = positions * ang_freqs
        cos = jnp.cos(radians).astype(x.dtype)
        sin = jnp.sin(radians).astype(x.dtype)
        if self.interpretable:
            # treat 'even' as all ones, 'odd' as all zeros.
            r_even = cos
            r_odd = sin
        else:
            # rotate coordinate pairs as usual
            even, odd = jnp.split(x, 2, axis=-1)
            r_even = even * cos - odd * sin
            r_odd = even * sin + odd * cos
        r = jnp.concatenate([r_even, r_odd], axis=-1)
        if self.interpretable:
            r = jnp.broadcast_to(r, x.shape)
        chex.assert_shape(r, x.shape)
        return r


class FractionalRotaryEncoder(nn.Module):
    rotary_base: float
    interpretable: bool

    @nn.compact
    def __call__(self, x):
        rotary, skip = jnp.split(x, 2, axis=-1)
        rotary = RotaryEncoder(self.rotary_base, self.interpretable)(rotary)
        return jnp.concatenate([rotary, skip], axis=-1)


class CausalMask(nn.Module):
    length: int

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.length)[..., None]
        j = jnp.arange(self.length)[None, ...]
        mask = jnp.less(i, j)  # keep lower triangular
        while mask.ndim < x.ndim:
            mask = mask[None, ...]
        return x - jnp.array([INFTY_APPROX], dtype=x.dtype) * mask


class MultiheadSelfAttention(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        shapes = Dimensions(
            B=x.shape[0],
            T=self.hps.sequence_len,
            M=self.hps.d_model,
            D=HEAD_DIM,
            H=self.hps.d_model // HEAD_DIM,
        )
        sharding = Dimensions(
            B="data",
            T=None,
            M=None,
            D=None,
            H="model",
        )
        chex.assert_shape(x, shapes["BTM"])
        x = sharding_constraint(x, sharding["BTM"], self.global_mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        q_init = init.zeros  # zero init; appdx d.2
        kv_init = init.normal(self.hps.d_model**-0.5)  # normal, var 1/fan_in; table 3
        o_init = init.normal(self.hps.d_model**-0.5)  # normal, var 1/fan_in = 1 / m
        wq = self.param(
            "w_aq",
            nn.with_partitioning(q_init, sharding["MHD"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wk = self.param(
            "w_ak",
            nn.with_partitioning(kv_init, sharding["MHD"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wv = self.param(
            "w_av",
            nn.with_partitioning(kv_init, sharding["MHD"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wo = self.param(
            "w_ao",
            nn.with_partitioning(o_init, sharding["HDM"], self.global_mesh),
            shapes["HDM"],
            self.hps.param_dtype,
        )

        # todo: maybe use dot general instead of einsum? need to see if it's faster
        q = jnp.einsum("bim,mhd->bhid", x, wq.astype(self.hps.dtype))
        k = jnp.einsum("bim,mhd->bhid", x, wk.astype(self.hps.dtype))
        v = jnp.einsum("bim,mhd->bhid", x, wv.astype(self.hps.dtype))
        q = sharding_constraint(q, sharding["BHTD"], self.global_mesh)
        k = sharding_constraint(k, sharding["BHTD"], self.global_mesh)
        v = sharding_constraint(v, sharding["BHTD"], self.global_mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        q = FractionalRotaryEncoder(self.hps.rotary_base, self.hps.rotary_interp_q)(q)
        k = FractionalRotaryEncoder(self.hps.rotary_base, self.hps.rotary_interp_k)(k)
        q = sharding_constraint(q, sharding["BHTD"], self.global_mesh)
        k = sharding_constraint(k, sharding["BHTD"], self.global_mesh)
        self.sow("intermediates", "aqr_l1", coord_check_l1(q))
        self.sow("intermediates", "akr_l1", coord_check_l1(k))

        s = jnp.einsum("bhid,bhjd->bhij", q, k) / HEAD_DIM
        s = sharding_constraint(s, sharding["BHTT"], self.global_mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        s = CausalMask(length=self.hps.sequence_len)(s)
        s = sharding_constraint(s, sharding["BHTT"], self.global_mesh)

        p = jax.nn.softmax(s, axis=-1)
        p = sharding_constraint(p, sharding["BHTT"], self.global_mesh)
        self.sow("intermediates", "ap_l1", coord_check_l1(p))

        o = jnp.einsum("bhij,bhjd->bhid", p, v)
        o = sharding_constraint(o, sharding["BHTD"], self.global_mesh)
        self.sow("intermediates", "ao_l1", coord_check_l1(o))

        r = jnp.einsum("bhid,hdm->bim", o, wo.astype(self.hps.dtype))
        r = sharding_constraint(r, sharding["BTM"], self.global_mesh)
        self.sow("intermediates", "ar_l1", coord_check_l1(r))
        return r


class MultiLayerPerceptron(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        seqlen = self.hps.sequence_len
        d_model = self.hps.d_model
        d_ff = self.hps.d_model * FF_MULTIPLE
        shapes = Dimensions(B=x.shape[0], T=seqlen, M=d_model, F=d_ff)
        sharding = Dimensions(B="data", T=None, M=None, F="model")
        chex.assert_shape(x, shapes["BTM"])
        x = sharding_constraint(x, sharding["BTM"], self.global_mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        w1_init = init.normal(d_model**-0.5)  # normal w variance 1 / fan_in
        w2_init = init.normal(d_ff**-0.5)  # normal w variance 1 / fan_in
        w1 = self.param(
            "w_fi",
            nn.with_partitioning(w1_init, sharding["MF"], self.global_mesh),
            shapes["MF"],
            self.hps.param_dtype,
        )
        w2 = self.param(
            "w_fo",
            nn.with_partitioning(w2_init, sharding["FM"], self.global_mesh),
            shapes["FM"],
            self.hps.param_dtype,
        )

        # todo: maybe use dot general?
        x = jnp.einsum("btm,mf->btf", x, w1.astype(self.hps.dtype))
        x = sharding_constraint(x, sharding["BTF"], self.global_mesh)
        self.sow("intermediates", "fp_l1", coord_check_l1(x))

        x = getattr(jax.nn, self.hps.act_name)(x)
        x = sharding_constraint(x, sharding["BTF"], self.global_mesh)
        if self.hps.act_square:
            x = jnp.square(x)
            x = sharding_constraint(x, sharding["BTF"], self.global_mesh)
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        # todo: maybe use dot general?
        x = jnp.einsum("btf,fm->btm", x, w2.astype(self.hps.dtype))
        x = sharding_constraint(x, sharding["BTF"], self.global_mesh)
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        x += MultiheadSelfAttention(self.hps, self.global_mesh)(RMSLayerNorm()(x))
        x = sharding_constraint(x, ("data", None, None), self.global_mesh)
        x += MultiLayerPerceptron(self.hps, self.global_mesh)(RMSLayerNorm()(x))
        x = sharding_constraint(x, ("data", None, None), self.global_mesh)
        return x, None


class Transformer(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        nv = self.hps.n_vocab
        dm = self.hps.d_model
        chex.assert_shape(x, [None, self.hps.sequence_len])
        x = sharding_constraint(x, ("data", None), self.global_mesh)

        e_init = init.normal(1.0)  # appendix b.1
        o_init = init.zeros  # appendix d.2
        w_emb = self.param(
            "w_ei",
            nn.with_partitioning(e_init, (None, None), self.global_mesh),
            [nv, dm],
            self.hps.param_dtype,
        )
        w_out = self.param(
            "w_eo",
            nn.with_partitioning(o_init, (None, None), self.global_mesh),
            [dm, nv],
            self.hps.param_dtype,
        )

        # todo: sharding constraints
        w_emb = w_emb[None, ...]  # 1VD
        x = x[..., None]  # BT1
        x = sharding_constraint(x, ("data", None, None), self.global_mesh)
        x = jnp.take_along_axis(w_emb, x, axis=-2)
        x = x.astype(self.hps.dtype)
        x = sharding_constraint(x, ("data", None, None), self.global_mesh)

        x, _ = nn.scan(
            nn_partitioning.remat(TransformerBlock),
            length=self.hps.n_layer,
            variable_axes=dict(params=0, intermediates=0),
            variable_broadcast=False,
            split_rngs=dict(params=True),
            metadata_params={nn.PARTITION_NAME: None},
        )(hps=self.hps, global_mesh=self.global_mesh)(x, None)

        x = sharding_constraint(x, ("data", None, None), self.global_mesh)
        x = RMSLayerNorm()(x)
        x = sharding_constraint(x, ("data", None, None), self.global_mesh)

        # todo: dot general
        x = jnp.einsum("btd,dv->btv", x, w_out.astype(jnp.float32))
        x = sharding_constraint(x, ("data", None, None), self.global_mesh)
        return x
