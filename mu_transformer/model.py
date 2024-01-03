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
import numpy as np
from flax import struct
from flax.linen import partitioning as nn_partitioning

from mu_transformer.dims import Dimensions
from mu_transformer.shard import sharding_constraint
from mu_transformer.sow import coord_check_l1

INFTY_APPROX = 1e30
MESH_AXES = Dimensions(R="rows", C="columns", P="planes", N=None)


@struct.dataclass
class TransformerConfig:
    param_dtype: Any
    dtype: Any
    sequence_len: int
    d_model: int
    d_head: int
    ff_multiple: int
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
        eps = jnp.array([1e-8], dtype=x.dtype)
        ms = jnp.mean(jnp.square(x), axis=-1)
        rms = jnp.sqrt(ms + eps)
        return x / rms[..., None]


class RotaryEncoding(nn.Module):
    rotary_base: float
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        *_, length, width = x.shape

        positions = jnp.arange(length)
        # positions = sharding_constraint(positions, MESH_AXES["N"], self.global_mesh)
        positions = positions[..., None]  # expand along width axis
        # positions = sharding_constraint(positions, MESH_AXES["NN"], self.global_mesh)

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        # dimensions = sharding_constraint(dimensions, MESH_AXES["N"], self.global_mesh)
        ang_freqs = jnp.power(self.rotary_base, -dimensions / (width // 2))
        ang_freqs = ang_freqs[None, ...]  # expand along length axis
        # ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NN"], self.global_mesh)

        # expand along leading axes, such as batch and head.
        positions = positions[None, None, ...]
        ang_freqs = ang_freqs[None, None, ...]
        # positions = sharding_constraint(positions, MESH_AXES["NNNN"], self.global_mesh) # noqa
        # ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NNNN"], self.global_mesh) # noqa
        chex.assert_shape(positions, [1, 1, length, 1])
        chex.assert_shape(ang_freqs, [1, 1, 1, width // 2])

        radians = positions * ang_freqs
        # radians = sharding_constraint(radians, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(radians, [1, 1, length, width // 2])

        cos = jnp.cos(radians).astype(x.dtype)
        sin = jnp.sin(radians).astype(x.dtype)
        # cos = sharding_constraint(cos, MESH_AXES["NNNN"], self.global_mesh)
        # sin = sharding_constraint(sin, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(cos, [1, 1, length, width // 2])
        chex.assert_shape(sin, [1, 1, length, width // 2])

        even, odd = jnp.split(x, 2, axis=-1)
        # even = sharding_constraint(even, MESH_AXES["RPNN"], self.global_mesh)
        # odd = sharding_constraint(odd, MESH_AXES["RPNN"], self.global_mesh)

        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        # r_even = sharding_constraint(r_even, MESH_AXES["RPNN"], self.global_mesh)
        # r_odd = sharding_constraint(r_odd, MESH_AXES["RPNN"], self.global_mesh)

        r = jnp.concatenate([r_even, r_odd], axis=-1)
        # r = sharding_constraint(r, MESH_AXES["RPNN"], self.global_mesh)
        chex.assert_shape(r, x.shape)
        return r


class FractionalRotaryEncoding(nn.Module):
    rotary_base: float

    @nn.compact
    def __call__(self, x):
        # x = sharding_constraint(x, MESH_AXES["RPNN"], self.global_mesh)
        rotary, skip = jnp.split(x, np.array([x.shape[-1] // 4]), axis=-1)
        # rotary = sharding_constraint(rotary, MESH_AXES["RPNN"], self.global_mesh)
        # skip = sharding_constraint(skip, MESH_AXES["RPNN"], self.global_mesh)
        rotary = RotaryEncoding(self.rotary_base)(rotary)
        x = jnp.concatenate([rotary, skip], axis=-1)
        # x = sharding_constraint(x, MESH_AXES["RPNN"], self.global_mesh)
        return x


class CausalMask(nn.Module):
    length: int
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.length)[..., None]
        j = jnp.arange(self.length)[None, ...]
        # i = sharding_constraint(i, MESH_AXES["NN"], self.global_mesh)
        # j = sharding_constraint(j, MESH_AXES["NN"], self.global_mesh)
        mask = jnp.less(i, j)  # i.e., j > i, indicator masks out non-causal connections
        # mask = sharding_constraint(mask, MESH_AXES["NN"], self.global_mesh)
        mask = mask[None, None, ...]
        # mask = sharding_constraint(mask, MESH_AXES["NNNN"], self.global_mesh)
        x = x - jnp.array([INFTY_APPROX], dtype=x.dtype) * mask
        # x = sharding_constraint(x, MESH_AXES["RPNN"], self.global_mesh)
        return x


class MultiheadSelfAttention(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        shapes = Dimensions(
            B=x.shape[0],
            T=self.hps.sequence_len,
            M=self.hps.d_model,
            D=self.hps.d_head,
            H=self.hps.d_model // self.hps.d_head,
        )
        chex.assert_shape(x, shapes["BTM"])
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        q_init = init.zeros  # zero init; appdx d.2
        kv_init = init.normal(self.hps.d_model**-0.5)  # normal, var 1/fan_in; table 3
        o_init = init.normal(self.hps.d_model**-0.5)  # normal, var 1/fan_in = 1 / m
        wq = self.param(
            "w_aq",
            nn.with_partitioning(q_init, MESH_AXES["CPN"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wk = self.param(
            "w_ak",
            nn.with_partitioning(kv_init, MESH_AXES["CPN"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wv = self.param(
            "w_av",
            nn.with_partitioning(kv_init, MESH_AXES["CPN"], self.global_mesh),
            shapes["MHD"],
            self.hps.param_dtype,
        )
        wo = self.param(
            "w_ao",
            nn.with_partitioning(o_init, MESH_AXES["PNC"], self.global_mesh),
            shapes["HDM"],
            self.hps.param_dtype,
        )

        # todo: maybe use dot general instead of einsum? need to see if it's faster
        wq = wq.astype(self.hps.dtype)
        # wq = sharding_constraint(wq, MESH_AXES["CPN"], self.global_mesh)
        wk = wk.astype(self.hps.dtype)
        # wk = sharding_constraint(wk, MESH_AXES["CPN"], self.global_mesh)
        wv = wv.astype(self.hps.dtype)
        # wv = sharding_constraint(wv, MESH_AXES["CPN"], self.global_mesh)
        q = jnp.einsum("bim,mhd->bhid", x, wq)
        k = jnp.einsum("bim,mhd->bhid", x, wk)
        v = jnp.einsum("bim,mhd->bhid", x, wv)
        chex.assert_trees_all_equal_dtypes(
            q, k, v, jnp.array([0.0], dtype=self.hps.dtype)
        )
        # q = sharding_constraint(q, MESH_AXES["RPNN"], self.global_mesh)
        # k = sharding_constraint(k, MESH_AXES["RPNN"], self.global_mesh)
        # v = sharding_constraint(v, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        q = RotaryEncoding(self.hps.rotary_base, self.global_mesh)(q)
        k = RotaryEncoding(self.hps.rotary_base, self.global_mesh)(k)
        chex.assert_trees_all_equal_dtypes(
            q, k, v, jnp.array([0.0], dtype=self.hps.dtype)
        )
        # q = sharding_constraint(q, MESH_AXES["RPNN"], self.global_mesh)
        # k = sharding_constraint(k, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "aqr_l1", coord_check_l1(q))
        self.sow("intermediates", "akr_l1", coord_check_l1(k))

        s = jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.array([self.hps.d_head], q.dtype)
        chex.assert_trees_all_equal_dtypes(s, jnp.array([0.0], dtype=self.hps.dtype))
        # s = sharding_constraint(s, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        s = CausalMask(self.hps.sequence_len, self.global_mesh)(s)
        chex.assert_trees_all_equal_dtypes(s, jnp.array([0.0], dtype=self.hps.dtype))
        # s = sharding_constraint(s, MESH_AXES["RPNN"], self.global_mesh)

        p = jax.nn.softmax(s, axis=-1)
        chex.assert_trees_all_equal_dtypes(p, jnp.array([0.0], dtype=self.hps.dtype))
        # p = sharding_constraint(p, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "ap_l1", coord_check_l1(p))

        o = jnp.einsum("bhij,bhjd->bhid", p, v)
        chex.assert_trees_all_equal_dtypes(o, jnp.array([0.0], dtype=self.hps.dtype))
        # o = sharding_constraint(o, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "ao_l1", coord_check_l1(o))

        wo = wo.astype(self.hps.dtype)
        # wo = sharding_constraint(wo, MESH_AXES["PNC"], self.global_mesh)
        r = jnp.einsum("bhid,hdm->bim", o, wo)
        chex.assert_trees_all_equal_dtypes(r, jnp.array([0.0], dtype=self.hps.dtype))
        # r = sharding_constraint(r, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "ar_l1", coord_check_l1(r))
        return r


class MultiLayerPerceptron(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        seqlen = self.hps.sequence_len
        d_model = self.hps.d_model
        d_ff = self.hps.d_model * self.hps.ff_multiple
        shapes = Dimensions(B=x.shape[0], T=seqlen, M=d_model, F=d_ff)
        chex.assert_shape(x, shapes["BTM"])
        chex.assert_trees_all_equal_dtypes(x, jnp.array([0.0], dtype=self.hps.dtype))
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        w1_init = init.normal(d_model**-0.5)  # normal w variance 1 / fan_in
        w2_init = init.normal(d_ff**-0.5)  # normal w variance 1 / fan_in
        w1 = self.param(
            "w_fi",
            nn.with_partitioning(w1_init, MESH_AXES["CP"], self.global_mesh),
            shapes["MF"],
            self.hps.param_dtype,
        )
        w2 = self.param(
            "w_fo",
            nn.with_partitioning(w2_init, MESH_AXES["PC"], self.global_mesh),
            shapes["FM"],
            self.hps.param_dtype,
        )

        # todo: maybe use dot general?
        w1 = w1.astype(self.hps.dtype)
        # w1 = sharding_constraint(w1, MESH_AXES["CP"], self.global_mesh)
        x = jnp.einsum("btm,mf->btf", x, w1)
        # x = sharding_constraint(x, MESH_AXES["RNP"], self.global_mesh)
        chex.assert_shape(x, shapes["BTF"])
        chex.assert_trees_all_equal_dtypes(x, jnp.array([0.0], dtype=self.hps.dtype))
        self.sow("intermediates", "fp_l1", coord_check_l1(x))

        x = getattr(jax.nn, self.hps.act_name)(x)
        # x = sharding_constraint(x, MESH_AXES["RNP"], self.global_mesh)
        if self.hps.act_square:
            x = jnp.square(x)
            # x = sharding_constraint(x, MESH_AXES["RNP"], self.global_mesh)
        chex.assert_trees_all_equal_dtypes(x, jnp.array([0.0], dtype=self.hps.dtype))
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        # todo: maybe use dot general?
        w2 = w2.astype(self.hps.dtype)
        # w2 = sharding_constraint(w2, MESH_AXES["PC"], self.global_mesh)
        x = jnp.einsum("btf,fm->btm", x, w2)
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        chex.assert_shape(x, shapes["BTM"])
        chex.assert_trees_all_equal_dtypes(x, jnp.array([0.0], dtype=self.hps.dtype))
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        x += MultiheadSelfAttention(self.hps, self.global_mesh)(RMSLayerNorm()(x))
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        x += MultiLayerPerceptron(self.hps, self.global_mesh)(RMSLayerNorm()(x))
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        return x, None


class Transformer(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        nv = self.hps.n_vocab
        dm = self.hps.d_model
        chex.assert_shape(x, [None, self.hps.sequence_len])
        # x = sharding_constraint(x, MESH_AXES["RN"], self.global_mesh)

        e_init = init.normal(1.0)  # appendix b.1
        o_init = init.zeros  # appendix d.2
        w_emb = self.param(
            "w_ei",
            nn.with_partitioning(e_init, MESH_AXES["NN"], self.global_mesh),  # no shard
            [nv, dm],
            self.hps.param_dtype,
        )
        w_out = self.param(
            "w_eo",
            nn.with_partitioning(o_init, MESH_AXES["NN"], self.global_mesh),  # shard
            [dm, nv],
            self.hps.param_dtype,
        )

        w_emb = w_emb[None, ...]  # 1VD
        x = x[..., None]  # BT1
        # x = sharding_constraint(x, MESH_AXES["RNN"], self.global_mesh)
        x = jnp.take_along_axis(w_emb, x, axis=-2)
        x = x.astype(self.hps.dtype)
        # x = sharding_constraint(x, MESH_AXES["RNN"], self.global_mesh)

        x, _ = nn.scan(
            nn_partitioning.remat(TransformerBlock),
            length=self.hps.n_layer,
            variable_axes=dict(params=0, intermediates=0),
            variable_broadcast=False,
            split_rngs=dict(params=True),
            metadata_params={nn.PARTITION_NAME: None},
        )(hps=self.hps, global_mesh=self.global_mesh)(x, None)
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)

        x = RMSLayerNorm()(x)
        # x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)

        # todo: dot general
        x = jnp.einsum("btd,dv->btv", x, w_out.astype(jnp.float32))
        # x = sharding_constraint(x, MESH_AXES["RCN"], self.global_mesh)
        return x
