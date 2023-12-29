from dataclasses import fields
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import struct

DTypeLike = Any


@struct.dataclass
class TransformerConfig:
    param_dtype: DTypeLike
    dtype: DTypeLike
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
        return x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1) + 1e-6)[..., None]


class RotaryEncoding(nn.Module):
    length: int
    width: int
    rotary_base: float
    interpretable: bool

    @nn.compact
    def __call__(self, x):
        positions = jnp.arange(self.length)
        positions = positions[..., None]  # expand along width axis

        dimensions = jnp.arange(self.width // 2)  # half each for sin and cos
        ang_freqs = jnp.power(self.rotary_base, -dimensions / (self.width // 2))
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
        chex.assert_shape(r, x.shape)
        return r


class FractionalRotaryEncoding(nn.Module):
    rotary_base: float
    interpretable: bool

    @nn.compact
    def __call__(self, x):
        rotary, skip = jnp.split(x, 2, axis=-1)
        rotary = RotaryEncoding(
            length=rotary.shape[-2],
            width=rotary.shape[-1],
            rotary_base=self.rotary_base,
            interpretable=self.interpretable,
        )(rotary)
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
        return x - 1e30 * mask.astype(x.dtype)


class MultiheadSelfAttention(nn.Module):
    hps: TransformerConfig

    @nn.compact
    def __call__(self, x):
        b = x.shape[0]
        t = self.hps.sequence_len
        dm = self.hps.d_model
        dh = 128
        nh = self.hps.d_model // dh
        chex.assert_shape(x, [b, t, dm])

        # todo: shard
        q_init = init.zeros  # zero init, per appdx d.2
        kv_init = init.normal(dm**-0.5)  # normal w variance 1 / fan_in, per table 3
        o_init = init.normal(dm**-0.5)  # same as kv since fan_in = nh * dh = dm
        wq = self.param("wq_ii", q_init, [nh, dm, dh], self.hps.param_dtype)
        wk = self.param("wk_ii", kv_init, [nh, dm, dh], self.hps.param_dtype)
        wv = self.param("wv_ii", kv_init, [nh, dm, dh], self.hps.param_dtype)
        wo = self.param("wo_ii", o_init, [nh, dh, dm], self.hps.param_dtype)

        # todo: use dot general instead of einsum to avoid transposes
        q = jnp.einsum("bim,hmd->bhid", x, wq.astype(self.hps.dtype))
        k = jnp.einsum("bim,hmd->bhid", x, wk.astype(self.hps.dtype))
        v = jnp.einsum("bim,hmd->bhid", x, wv.astype(self.hps.dtype))

        q = FractionalRotaryEncoding(self.hps.rotary_base, self.hps.rotary_interp_q)(q)
        k = FractionalRotaryEncoding(self.hps.rotary_base, self.hps.rotary_interp_k)(k)
        q, k = map(lambda y: y * (dh**-0.5), [q, k])  # def 4.1
        self.sow("intermediates", "q_norm_m1", jnp.mean(jnp.linalg.norm(q, axis=-1)))
        self.sow("intermediates", "k_norm_m1", jnp.mean(jnp.linalg.norm(k, axis=-1)))

        s = jnp.einsum("bhik,bhjk->bhij", q, k)
        s = CausalMask(length=t)(s)
        p = jax.nn.softmax(s, axis=-1)
        o = jnp.einsum("bhij,bhjd->bihd", p, v)
        return jnp.einsum("bihd,hdm->bim", o, wo.astype(self.hps.dtype))


class MultiLayerPerceptron(nn.Module):
    hps: TransformerConfig

    @nn.compact
    def __call__(self, x):
        dm = self.hps.d_model
        dff = 4 * self.hps.d_model

        # todo: shard
        w1_init = init.normal(dm**-0.5)  # normal w variance 1 / fan_in
        w2_init = init.normal(dff**-0.5)  # normal w variance 1 / fan_in
        w1 = self.param("w1_ii", w1_init, [dm, dff], self.hps.param_dtype)
        w2 = self.param("w2_ii", w2_init, [dff, dm], self.hps.param_dtype)

        # todo: dot general
        x = jnp.einsum("btd,df->btf", x, w1.astype(self.hps.dtype))

        x = getattr(jax.nn, self.hps.act_name)(x)
        if self.hps.act_square:
            x = jnp.square(x)

        # todo: dot general
        x = jnp.einsum("btf,fd->btd", x, w2.astype(self.hps.dtype))
        return x


class TransformerBlock(nn.Module):
    hps: TransformerConfig

    @nn.compact
    def __call__(self, x):
        # todo: sharding constraints
        x += MultiheadSelfAttention(self.hps)(RMSLayerNorm()(x))
        x += MultiLayerPerceptron(self.hps)(RMSLayerNorm()(x))
        return x


class Transformer(nn.Module):
    hps: TransformerConfig

    @nn.compact
    def __call__(self, x):
        nv = self.hps.n_vocab
        dm = self.hps.d_model

        # todo: shard
        e_init = init.normal(1.0)  # appendix b.1
        o_init = init.zeros  # appendix d.2
        w_emb = self.param("we_fi", e_init, [nv, dm], self.hps.param_dtype)
        w_out = self.param("wd_if", o_init, [dm, nv], self.hps.param_dtype)

        # todo: sharding constraints
        w_emb = w_emb[None, ...]  # 1VD
        x = x[..., None]  # BT1
        x = jnp.take_along_axis(w_emb, x, axis=-2)
        x = x.astype(self.hps.dtype)
        x = nn.remat_scan(TransformerBlock, lengths=(self.hps.n_layer, 1))(
            hps=self.hps, name="stack"
        )(x)
        x = RMSLayerNorm()(x)

        # todo: dot general
        x = jnp.einsum("btd,dv->btv", x, w_out)
        x = x.astype(jnp.float32)
        return x
