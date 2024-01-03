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
SHARDING = Dimensions(S="devices", R=None)


@struct.dataclass
class TransformerConfig:
    param_dtype: Any
    dtype: Any
    sequence_len: int
    d_model: int
    n_layer: int
    n_vocab: int
    rotary_base: int
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
        even, odd = jnp.split(x, 2, axis=-1)
        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        r = jnp.concatenate([r_even, r_odd], axis=-1)
        chex.assert_shape(r, x.shape)
        return r


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
        chex.assert_shape(x, shapes["BTM"])
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        q_init = init.zeros  # zero init; appdx d.2
        kv_init = init.normal(self.hps.d_model**-0.5)  # normal, var 1/fan_in; table 3
        o_init = init.normal(self.hps.d_model**-0.5)  # normal, var 1/fan_in = 1 / m
        wq = self.param(
            "w_aq",
            nn.with_partitioning(q_init, SHARDING["SRR"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wk = self.param(
            "w_ak",
            nn.with_partitioning(kv_init, SHARDING["SRR"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wv = self.param(
            "w_av",
            nn.with_partitioning(kv_init, SHARDING["SRR"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wo = self.param(
            "w_ao",
            nn.with_partitioning(o_init, SHARDING["RRS"], self.global_mesh),
            shapes["HDM"],
            self.hps.param_dtype,
        )

        # todo: maybe use dot general instead of einsum? need to see if it's faster
        q = jnp.einsum("bim,mhd->bhid", x, wq.astype(self.hps.dtype))
        k = jnp.einsum("bim,mhd->bhid", x, wk.astype(self.hps.dtype))
        v = jnp.einsum("bim,mhd->bhid", x, wv.astype(self.hps.dtype))
        q = sharding_constraint(q, SHARDING["SRRR"], self.global_mesh)
        k = sharding_constraint(k, SHARDING["SRRR"], self.global_mesh)
        v = sharding_constraint(v, SHARDING["SRRR"], self.global_mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        q = RotaryEncoder(self.hps.rotary_base)(q)
        k = RotaryEncoder(self.hps.rotary_base)(k)
        q = sharding_constraint(q, SHARDING["SRRR"], self.global_mesh)
        k = sharding_constraint(k, SHARDING["SRRR"], self.global_mesh)
        self.sow("intermediates", "aqr_l1", coord_check_l1(q))
        self.sow("intermediates", "akr_l1", coord_check_l1(k))

        s = jnp.einsum("bhid,bhjd->bhij", q, k) / HEAD_DIM
        s = sharding_constraint(s, SHARDING["SRRR"], self.global_mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        s = CausalMask(length=self.hps.sequence_len)(s)
        s = sharding_constraint(s, SHARDING["SRRR"], self.global_mesh)

        p = jax.nn.softmax(s, axis=-1)
        p = sharding_constraint(p, SHARDING["SRRR"], self.global_mesh)
        self.sow("intermediates", "ap_l1", coord_check_l1(p))

        o = jnp.einsum("bhij,bhjd->bhid", p, v)
        o = sharding_constraint(o, SHARDING["SRRR"], self.global_mesh)
        self.sow("intermediates", "ao_l1", coord_check_l1(o))

        r = jnp.einsum("bhid,hdm->bim", o, wo.astype(self.hps.dtype))
        r = sharding_constraint(r, SHARDING["RRS"], self.global_mesh)
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
        chex.assert_shape(x, shapes["BTM"])
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        w1_init = init.normal(d_model**-0.5)  # normal w variance 1 / fan_in
        w2_init = init.normal(d_ff**-0.5)  # normal w variance 1 / fan_in
        w1 = self.param(
            "w_fi",
            nn.with_partitioning(w1_init, SHARDING["SR"], self.global_mesh),
            shapes["MF"],
            self.hps.param_dtype,
        )
        w2 = self.param(
            "w_fo",
            nn.with_partitioning(w2_init, SHARDING["RS"], self.global_mesh),
            shapes["FM"],
            self.hps.param_dtype,
        )

        # todo: maybe use dot general?
        x = jnp.einsum("btm,mf->btf", x, w1.astype(self.hps.dtype))
        x = sharding_constraint(x, SHARDING["SRR"], self.global_mesh)
        self.sow("intermediates", "fp_l1", coord_check_l1(x))

        x = getattr(jax.nn, self.hps.act_name)(x)
        x = sharding_constraint(x, SHARDING["SRR"], self.global_mesh)
        if self.hps.act_square:
            x = jnp.square(x)
            x = sharding_constraint(x, SHARDING["SRR"], self.global_mesh)
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        # todo: maybe use dot general?
        x = jnp.einsum("btf,fm->btm", x, w2.astype(self.hps.dtype))
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        x += MultiheadSelfAttention(self.hps, self.global_mesh)(RMSLayerNorm()(x))
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)
        x += MultiLayerPerceptron(self.hps, self.global_mesh)(RMSLayerNorm()(x))
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)
        return x, None


class Transformer(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        nv = self.hps.n_vocab
        dm = self.hps.d_model
        chex.assert_shape(x, [None, self.hps.sequence_len])
        x = sharding_constraint(x, SHARDING["SR"], self.global_mesh)

        e_init = init.normal(1.0)  # appendix b.1
        o_init = init.zeros  # appendix d.2
        w_emb = self.param(
            "w_ei",
            nn.with_partitioning(e_init, SHARDING["RS"], self.global_mesh),
            [nv, dm],
            self.hps.param_dtype,
        )
        w_out = self.param(
            "w_eo",
            nn.with_partitioning(o_init, SHARDING["SR"], self.global_mesh),
            [dm, nv],
            self.hps.param_dtype,
        )

        w_emb = w_emb[None, ...]  # 1VD
        x = x[..., None]  # BT1
        x = sharding_constraint(x, SHARDING["SRR"], self.global_mesh)
        x = jnp.take_along_axis(w_emb, x, axis=-2)
        x = x.astype(self.hps.dtype)
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)

        x, _ = nn.scan(
            nn_partitioning.remat(TransformerBlock),
            length=self.hps.n_layer,
            variable_axes=dict(params=0, intermediates=0),
            variable_broadcast=False,
            split_rngs=dict(params=True),
            metadata_params={nn.PARTITION_NAME: None},
        )(hps=self.hps, global_mesh=self.global_mesh)(x, None)
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)

        x = RMSLayerNorm()(x)
        x = sharding_constraint(x, SHARDING["RRS"], self.global_mesh)

        # todo: dot general
        x = jnp.einsum("btd,dv->btv", x, w_out.astype(jnp.float32))
        x = sharding_constraint(x, SHARDING["SRR"], self.global_mesh)
        return x
