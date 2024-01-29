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
from flax.linen import partitioning as nnp

from mu_transformer.dims import Dimensions
from mu_transformer.jax_impl.shard import sharding_constraint
from mu_transformer.jax_impl.sow import coord_check_l1

INFTY_APPROX = 1e30
MESH_AXES = Dimensions(R="rows", C="columns", P="planes", N=None)


@struct.dataclass
class TransformerConfig:
    parameterization: str
    param_dtype: Any
    dtype: Any
    output_logits_dtype: Any
    sequence_len: int
    d_model: int
    d_head: int
    ff_multiple: int
    rotary_base: int
    act_name: str
    act_square: bool
    norm_eps: float
    n_layer: int
    n_vocab: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    is_train: bool

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        flt = {k: v for k, v in kwargs.items() if k in signature}
        flt.update({k: jnp.dtype(v) for k, v in flt.items() if k.endswith("_dtype")})
        return cls(**flt)


class RMSNorm(nn.Module):
    hps: TransformerConfig

    @nn.compact
    def __call__(self, x):
        eps = jnp.array([self.hps.norm_eps], dtype=x.dtype)
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
        positions = sharding_constraint(positions, MESH_AXES["N"], self.global_mesh)
        positions = positions[..., None]  # expand along width axis
        positions = sharding_constraint(positions, MESH_AXES["NN"], self.global_mesh)

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        dimensions = sharding_constraint(dimensions, MESH_AXES["N"], self.global_mesh)
        ang_freqs = jnp.power(self.rotary_base, -dimensions / (width // 2))
        ang_freqs = ang_freqs[None, ...]  # expand along length axis
        ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NN"], self.global_mesh)

        # expand along leading axes, such as batch and head.
        positions = positions[None, None, ...]
        ang_freqs = ang_freqs[None, None, ...]
        positions = sharding_constraint(positions, MESH_AXES["NNNN"], self.global_mesh)
        ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(positions, [1, 1, length, 1])
        chex.assert_shape(ang_freqs, [1, 1, 1, width // 2])

        radians = positions * ang_freqs
        radians = sharding_constraint(radians, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(radians, [1, 1, length, width // 2])

        cos = jnp.cos(radians).astype(x.dtype)
        sin = jnp.sin(radians).astype(x.dtype)
        cos = sharding_constraint(cos, MESH_AXES["NNNN"], self.global_mesh)
        sin = sharding_constraint(sin, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(cos, [1, 1, length, width // 2])
        chex.assert_shape(sin, [1, 1, length, width // 2])

        even, odd = jnp.split(x, 2, axis=-1)
        even = sharding_constraint(even, MESH_AXES["RPNN"], self.global_mesh)
        odd = sharding_constraint(odd, MESH_AXES["RPNN"], self.global_mesh)

        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        r_even = sharding_constraint(r_even, MESH_AXES["RPNN"], self.global_mesh)
        r_odd = sharding_constraint(r_odd, MESH_AXES["RPNN"], self.global_mesh)

        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, MESH_AXES["RPNN"], self.global_mesh)
        chex.assert_shape(r, x.shape)
        return r


class CausalMask(nn.Module):
    length: int
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.length)[..., None]
        j = jnp.arange(self.length)[None, ...]
        i = sharding_constraint(i, MESH_AXES["NN"], self.global_mesh)
        j = sharding_constraint(j, MESH_AXES["NN"], self.global_mesh)
        mask = jnp.less(i, j)  # i.e., j > i, indicator masks out non-causal connections
        mask = sharding_constraint(mask, MESH_AXES["NN"], self.global_mesh)
        mask = mask[None, None, ...]
        mask = sharding_constraint(mask, MESH_AXES["NNNN"], self.global_mesh)
        x = x - jnp.array([INFTY_APPROX], dtype=x.dtype) * mask
        x = sharding_constraint(x, MESH_AXES["RPNN"], self.global_mesh)
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
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        if self.hps.parameterization in {"sp"}:
            q_init = init.normal(self.hps.d_model**-0.5)
        elif self.hps.parameterization in {"sp++", "mup"}:
            q_init = init.zeros  # zero init; appdx d.2
        else:
            raise NotImplementedError

        # normal, var 1/fan_in; table 3
        kv_init = init.normal(self.hps.d_model**-0.5)
        # normal, var 1/fan_in; table 3 with nh*dh=dm.
        o_init = init.normal(self.hps.d_model**-0.5)
        wq = self.param(
            "w_aq",
            nn.with_partitioning(q_init, MESH_AXES["PCN"], self.global_mesh),
            shapes["HMD"],
            self.hps.param_dtype,
        )
        wk = self.param(
            "w_ak",
            nn.with_partitioning(kv_init, MESH_AXES["PCN"], self.global_mesh),
            shapes["HMD"],
            self.hps.param_dtype,
        )
        wv = self.param(
            "w_av",
            nn.with_partitioning(kv_init, MESH_AXES["PCN"], self.global_mesh),
            shapes["HMD"],
            self.hps.param_dtype,
        )
        wo = self.param(
            "w_ao",
            nn.with_partitioning(o_init, MESH_AXES["PNC"], self.global_mesh),
            shapes["HDM"],
            self.hps.param_dtype,
        )

        q = jnp.einsum("bim,hmd->bhid", x, wq.astype(self.hps.dtype))
        k = jnp.einsum("bim,hmd->bhid", x, wk.astype(self.hps.dtype))
        v = jnp.einsum("bim,hmd->bhid", x, wv.astype(self.hps.dtype))
        q = sharding_constraint(q, MESH_AXES["RPNN"], self.global_mesh)
        k = sharding_constraint(k, MESH_AXES["RPNN"], self.global_mesh)
        v = sharding_constraint(v, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        if self.hps.rotary_base > 0:
            q = RotaryEncoding(self.hps.rotary_base, self.global_mesh)(q)
            k = RotaryEncoding(self.hps.rotary_base, self.global_mesh)(k)
            q = sharding_constraint(q, MESH_AXES["RPNN"], self.global_mesh)
            k = sharding_constraint(k, MESH_AXES["RPNN"], self.global_mesh)
            self.sow("intermediates", "aqr_l1", coord_check_l1(q))
            self.sow("intermediates", "akr_l1", coord_check_l1(k))

        mult = jnp.array([self.hps.d_head**-0.5], dtype=self.hps.dtype)
        s = jnp.einsum("bhid,bhjd->bhij", q * mult, k * mult)
        s = sharding_constraint(s, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        s = CausalMask(self.hps.sequence_len, self.global_mesh)(s)
        s = sharding_constraint(s, MESH_AXES["RPNN"], self.global_mesh)

        p = jax.nn.softmax(s, axis=-1)
        p = sharding_constraint(p, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "ap_l1", coord_check_l1(p))

        o = jnp.einsum("bhij,bhjd->bhid", p, v)
        o = sharding_constraint(o, MESH_AXES["RPNN"], self.global_mesh)
        self.sow("intermediates", "ao_l1", coord_check_l1(o))

        r = jnp.einsum("bhid,hdm->bim", o, wo.astype(self.hps.dtype))
        r = sharding_constraint(r, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "ar_l1", coord_check_l1(r))
        return r


class MultiLayerPerceptron(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        shapes = Dimensions(
            B=x.shape[0],
            T=self.hps.sequence_len,
            M=self.hps.d_model,
            F=self.hps.d_model * self.hps.ff_multiple,
        )
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        # table 3
        i_init = init.normal(self.hps.d_model**-0.5)
        # table 3 with dff = dm * ff_multiple
        o_init = init.normal((self.hps.d_model * self.hps.ff_multiple) ** -0.5)
        w_fi = self.param(
            "w_fi",
            nn.with_partitioning(i_init, MESH_AXES["CP"], self.global_mesh),
            shapes["MF"],
            self.hps.param_dtype,
        )
        w_fo = self.param(
            "w_fo",
            nn.with_partitioning(o_init, MESH_AXES["PC"], self.global_mesh),
            shapes["FM"],
            self.hps.param_dtype,
        )

        x = jnp.einsum("btm,mf->btf", x, w_fi.astype(self.hps.dtype))
        x = sharding_constraint(x, MESH_AXES["RNP"], self.global_mesh)
        self.sow("intermediates", "fp_l1", coord_check_l1(x))

        x = getattr(jax.nn, self.hps.act_name)(x)
        x = sharding_constraint(x, MESH_AXES["RNP"], self.global_mesh)
        if self.hps.act_square:
            x = jnp.square(x)
            x = sharding_constraint(x, MESH_AXES["RNP"], self.global_mesh)
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        x = jnp.einsum("btf,fm->btm", x, w_fo.astype(self.hps.dtype))
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        x += MultiheadSelfAttention(self.hps, self.global_mesh)(RMSNorm(self.hps)(x))
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        x += MultiLayerPerceptron(self.hps, self.global_mesh)(RMSNorm(self.hps)(x))
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        return x, None


class Embedding(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        e_init = init.normal(1.0)  # appendix b.1
        w_emb = self.param(
            "w_ei",
            nn.with_partitioning(e_init, MESH_AXES["NN"], self.global_mesh),  # NC=slow
            [self.hps.n_vocab, self.hps.d_model],
            self.hps.param_dtype,
        )
        x = sharding_constraint(x, MESH_AXES["RN"], self.global_mesh)
        x = jnp.take_along_axis(
            w_emb.astype(self.hps.dtype)[None, ...],  # 1VM
            x[..., None],  # BT1
            axis=1,
        )
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        return x


class PredictionHead(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        if self.hps.parameterization in {"sp"}:
            o_init = init.normal(self.hps.d_model**-0.5)
        elif self.hps.parameterization in {"sp++", "mup"}:
            o_init = init.zeros  # zero init; appdx d.2
        else:
            raise NotImplementedError
        w_out = self.param(
            "w_eo",
            nn.with_partitioning(o_init, MESH_AXES["CN"], self.global_mesh),
            [self.hps.d_model, self.hps.n_vocab],
            self.hps.param_dtype,
        )
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        x = RMSNorm(self.hps)(x)
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        if self.hps.is_train:
            output_logits_dtype = self.hps.output_logits_dtype
        else:
            output_logits_dtype = self.hps.param_dtype
        x = jnp.einsum("btm,mv->btv", x, w_out.astype(output_logits_dtype))
        x = sharding_constraint(x, MESH_AXES["RNN"], self.global_mesh)
        return x


class Transformer(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x[:, 0:-1], ((0, 0), (1, 0)), constant_values=self.hps.bos_token_id)
        x = nnp.remat(Embedding)(self.hps, self.global_mesh)(x)
        x, _ = nn.scan(
            nnp.remat(TransformerBlock),
            length=self.hps.n_layer,
            variable_axes=dict(params=0, intermediates=0),
            variable_broadcast=False,
            split_rngs=dict(params=True),
            metadata_params={nn.PARTITION_NAME: None},
        )(hps=self.hps, global_mesh=self.global_mesh)(x, None)
        x = sharding_constraint(x, MESH_AXES["RNC"], self.global_mesh)
        x = nnp.remat(PredictionHead)(self.hps, self.global_mesh)(x)
        return x
