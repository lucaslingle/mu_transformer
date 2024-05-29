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
MESH_AXES = Dimensions(X="X", Y="Y", N=None)


@struct.dataclass
class TransformerConfig:
    param_dtype: Any
    dtype: Any
    sequence_len: int
    d_model: int
    d_head: int
    qkv_sepconv: bool
    qk_norm: bool
    qk_kernel: str
    v_gating: bool
    rotary_base: int
    act_name: str
    act_square: bool
    norm_eps: float
    norm_trainable: bool
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


class XLayerNorm(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str

    @nn.compact
    def __call__(self, x):
        m = self.hps.d_model
        eps = jnp.array([self.hps.norm_eps], dtype=x.dtype)
        one = jnp.array([1.0], dtype=x.dtype)
        x -= jnp.mean(x, axis=-1)[..., None]
        x /= jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + eps)[..., None]
        if self.hps.norm_trainable:
            x *= one + self.param(
                "g_" + self.suffix,
                nn.with_partitioning(init.zeros, MESH_AXES["Y"], self.global_mesh),
                [m],
                self.hps.param_dtype,
            )[None, None, ...].astype(self.hps.dtype)
            x += self.param(
                "b_" + self.suffix,
                nn.with_partitioning(init.zeros, MESH_AXES["Y"], self.global_mesh),
                [m],
                self.hps.param_dtype,
            )[None, None, ...].astype(self.hps.dtype)
        return x


class QKLayerNorm(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str

    @nn.compact
    def __call__(self, x):
        d = self.hps.d_head
        h = self.hps.d_model // self.hps.d_head
        eps = jnp.array([self.hps.norm_eps], dtype=x.dtype)
        one = jnp.array([1.0], dtype=x.dtype)
        x -= jnp.mean(x, axis=-1)[..., None]
        x /= jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + eps)[..., None]
        if self.hps.norm_trainable:
            x *= one + self.param(
                "g_" + self.suffix,
                nn.with_partitioning(init.zeros, MESH_AXES["YNN"], self.global_mesh),
                [h, 1, d],
                self.hps.param_dtype,
            )[None, ...].astype(self.hps.dtype)
            x += self.param(
                "b_" + self.suffix,
                nn.with_partitioning(init.zeros, MESH_AXES["YNN"], self.global_mesh),
                [h, 1, d],
                self.hps.param_dtype,
            )[None, ...].astype(self.hps.dtype)
        return x


class RotaryEncoding(nn.Module):
    hps: TransformerConfig
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
        ang_freqs = jnp.power(self.hps.rotary_base, -dimensions / (width // 2))
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
        even = sharding_constraint(even, MESH_AXES["XYNN"], self.global_mesh)
        odd = sharding_constraint(odd, MESH_AXES["XYNN"], self.global_mesh)

        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        r_even = sharding_constraint(r_even, MESH_AXES["XYNN"], self.global_mesh)
        r_odd = sharding_constraint(r_odd, MESH_AXES["XYNN"], self.global_mesh)

        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, MESH_AXES["XYNN"], self.global_mesh)
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
        x = sharding_constraint(x, MESH_AXES["XYNN"], self.global_mesh)
        return x


class MultiHeadAttention(nn.Module):
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
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        w_init = init.normal(self.hps.d_model**-0.5)
        i_init = nn.with_partitioning(w_init, MESH_AXES["XYN"], self.global_mesh)
        o_init = nn.with_partitioning(w_init, MESH_AXES["YNX"], self.global_mesh)

        wq = self.param("w_aq", i_init, shapes["MHD"], self.hps.param_dtype)
        wk = self.param("w_ak", i_init, shapes["MHD"], self.hps.param_dtype)
        wv = self.param("w_av", i_init, shapes["MHD"], self.hps.param_dtype)
        wo = self.param("w_ao", o_init, shapes["HDM"], self.hps.param_dtype)

        q = jnp.einsum("bim,mhd->bhid", x, wq.astype(self.hps.dtype))
        k = jnp.einsum("bim,mhd->bhid", x, wk.astype(self.hps.dtype))
        v = jnp.einsum("bim,mhd->bhid", x, wv.astype(self.hps.dtype))
        q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
        k = sharding_constraint(k, MESH_AXES["XYNN"], self.global_mesh)
        v = sharding_constraint(v, MESH_AXES["XYNN"], self.global_mesh)

        if self.hps.qkv_sepconv:
            c_init = init.normal(3 ** -0.5)
            c_init = nn.with_partitioning(c_init, MESH_AXES["YNN"], self.global_mesh)
            c_shape = shapes["PHD"]
            c_args = [c_init, c_shape, self.hps.param_dtype]
            cq = self.param("c_aq", *c_args).astype(self.hps.dtype)
            ck = self.param("c_ak", *c_args).astype(self.hps.dtype)
            cv = self.param("c_av", *c_args).astype(self.hps.dtype)
            q = jnp.pad(q, ((0, 0), (0, 0), (2, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, 0), (2, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, 0), (2, 0), (0, 0)))
            q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
            k = sharding_constraint(k, MESH_AXES["XYNN"], self.global_mesh)
            v = sharding_constraint(v, MESH_AXES["XYNN"], self.global_mesh)
            q = (
                q[:, :, 0:-2, :] * cq[None, 0, :, :] +
                q[:, :, 1:-1, :] * cq[None, 1, :, :] +
                q[:, :, 2:, :] * cq[None, 2, :, :]
            )
            k = (
                k[:, :, 0:-2, :] * ck[None, 0, :, :] +
                k[:, :, 1:-1, :] * ck[None, 1, :, :] +
                k[:, :, 2:, :] * ck[None, 2, :, :]
            )
            v = (
                v[:, :, 0:-2, :] * cv[None, 0, :, :] +
                v[:, :, 1:-1, :] * cv[None, 1, :, :] +
                v[:, :, 2:, :] * cv[None, 2, :, :]
            )
            q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
            k = sharding_constraint(k, MESH_AXES["XYNN"], self.global_mesh)
            v = sharding_constraint(v, MESH_AXES["XYNN"], self.global_mesh)

        if self.hps.qk_norm:
            q = QKLayerNorm(self.hps, self.global_mesh, "aq")(q)
            k = QKLayerNorm(self.hps, self.global_mesh, "ak")(k)
            q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
            k = sharding_constraint(k, MESH_AXES["XYNN"], self.global_mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))

        q = RotaryEncoding(self.hps, self.global_mesh)(q)
        k = RotaryEncoding(self.hps, self.global_mesh)(k)
        q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
        k = sharding_constraint(k, kv_mesh_axes, self.global_mesh)
        self.sow("intermediates", "aqr_l1", coord_check_l1(q))
        self.sow("intermediates", "akr_l1", coord_check_l1(k))

        if self.hps.v_gating:
            wg = self.param("w_ag", i_init, shapes["MHD"], self.hps.param_dtype)
            g = jnp.einsum("bim,mhd->bhid", x, wg.astype(self.hps.dtype))
            g = sharding_constraint(g, MESH_AXES["XYNN"], self.global_mesh)
            if self.hps.qkv_sepconv:
                c_init = init.normal(3 ** -0.5)
                c_init = nn.with_partitioning(c_init, MESH_AXES["YNN"], self.global_mesh)
                c_shape = shapes["PHD"]
                c_args = [c_init, c_shape, self.hps.param_dtype]
                cg = self.param("c_ag", *c_args).astype(self.hps.dtype)
                g = jnp.pad(g, ((0, 0), (0, 0), (2, 0), (0, 0)))
                g = (
                    g[:, :, 0:-2, :] * cg[None, 0, :, :] +
                    g[:, :, 1:-1, :] * cg[None, 1, :, :] +
                    g[:, :, 2:, :] * cg[None, 2, :, :]
                )
                g = sharding_constraint(g, MESH_AXES["XYNN"], self.global_mesh)
            v = v * jax.nn.silu(g)
            v = sharding_constraint(v, MESH_AXES["XYNN"], self.global_mesh)
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        mult = jnp.array([self.hps.qk_scale**0.5], dtype=self.hps.dtype)
        s = jnp.einsum("bhid,bhjd->bhij", q * mult, k * mult)
        s = sharding_constraint(s, MESH_AXES["XYNN"], self.global_mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        s = CausalMask(self.hps.sequence_len, self.global_mesh)(s)
        s = sharding_constraint(s, MESH_AXES["XYNN"], self.global_mesh)

        if self.hps.qk_kernel == "softmax":
            p = jax.nn.softmax(s, axis=-1)
        elif self.hps.qk_kernel == "sqrelu":
            p = jnp.square(jax.nn.relu(s))
        else:
            raise NotImplementedError("Unrecognized qk_kernel name")
        p = sharding_constraint(p, MESH_AXES["XYNN"], self.global_mesh)
        self.sow("intermediates", "ap_l1", coord_check_l1(p))

        o = jnp.einsum("bhij,bhjd->bhid", p, v)
        o = sharding_constraint(o, MESH_AXES["XYNN"], self.global_mesh)
        self.sow("intermediates", "ao_l1", coord_check_l1(o))

        r = jnp.einsum("bhid,hdm->bim", o, wo.astype(self.hps.dtype))
        r = sharding_constraint(r, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "ar_l1", coord_check_l1(r))
        return r


class MultiLayerPerceptron(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        ff_multiple = 3.5 if self.hps.v_gating else 4.0
        d_ff = int(ff_multiple * self.hps.d_model)

        shapes = Dimensions(
            B=x.shape[0],
            T=self.hps.sequence_len,
            M=self.hps.d_model,
            F=d_ff,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        i_init = init.normal(self.hps.d_model**-0.5)
        o_init = init.normal(d_ff_out**-0.5)

        wi = self.param(
            "w_fi",
            nn.with_partitioning(i_init, MESH_AXES["XY"], self.global_mesh),
            shapes["MF"],
            self.hps.param_dtype,
        )
        wo = self.param(
            "w_fo",
            nn.with_partitioning(o_init, MESH_AXES["YX"], self.global_mesh),
            shapes["FM"],
            self.hps.param_dtype,
        )

        x = jnp.einsum("btm,mf->btf", x, wi.astype(self.hps.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fh_l1", coord_check_l1(x))

        x = getattr(jax.nn, self.hps.act_name)(x)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        if self.hps.act_square:
            x = jnp.square(x)
            x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        x = jnp.einsum("btf,fm->btm", x, wo.astype(self.hps.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        kws = dict(hps=self.hps, global_mesh=self.global_mesh)

        x += MultiHeadAttention(**kws)(XLayerNorm(**kws, suffix="a")(x))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        x += MultiLayerPerceptron(**kws)(XLayerNorm(**kws, suffix="f")(x))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        return x, None


class Embedding(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        w_emb = self.param(
            "w_e",
            nn.with_partitioning(init.normal(1.0), MESH_AXES["NY"], self.global_mesh),
            [self.hps.n_vocab, self.hps.d_model],
            self.hps.param_dtype,
        )
        x = sharding_constraint(x, MESH_AXES["XN"], self.global_mesh)
        x = jnp.take_along_axis(
            w_emb.astype(self.hps.dtype)[None, ...],  # 1VM
            x[..., None],  # BT1
            axis=1,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        return x


class Unembedding(nn.Module):
    hps: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        x = XLayerNorm(self.hps, self.global_mesh, "u")(x)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        wu = self.param(
            "w_u",
            nn.with_partitioning(init.zeros, MESH_AXES["YN"], self.global_mesh),
            [self.hps.d_model, self.hps.n_vocab],
            self.hps.param_dtype,
        )

        out_dtype = self.hps.dtype if self.hps.is_train else self.hps.param_dtype
        x = jnp.einsum("btm,mv->btv", x, wu.astype(out_dtype))
        x = sharding_constraint(x, MESH_AXES["XNN"], self.global_mesh)
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
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        x = nnp.remat(Unembedding)(self.hps, self.global_mesh)(x)
        return x
