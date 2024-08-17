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
from typing import Dict
from typing import Optional

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
    ff_multiple: int
    e_norm: bool
    q_init: str
    r_init: str
    u_init: str
    qk_scale: float
    qk_norm: bool
    kv_mqa: bool
    rotary_base: int
    act_name: str
    act_square: bool
    norm_eps: float
    norm_gains: bool
    norm_gains_type: str
    proj_biases: bool
    n_layer: int
    n_vocab: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    is_train: bool
    is_decoding: bool

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        flt = {k: v for k, v in kwargs.items() if k in signature}
        flt.update({k: jnp.dtype(v) for k, v in flt.items() if k.endswith("_dtype")})
        return cls(**flt)


class RMSNorm(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str

    @nn.compact
    def __call__(self, x):
        eps = jnp.array([self.cfg.norm_eps], dtype=x.dtype)
        ms = jnp.mean(jnp.square(x), axis=-1)
        rms = jnp.sqrt(ms + eps)
        normed = x / rms[..., None]
        if self.cfg.norm_gains:
            g_is_scalar = self.cfg.norm_gains_type == "scalar"
            g_shape = [1] if g_is_scalar else [self.cfg.d_model]
            g_mesh = MESH_AXES["N"] if g_is_scalar else MESH_AXES["Y"]
            normed *= self.param(
                "g_" + self.suffix,
                nn.with_partitioning(init.ones, g_mesh, self.global_mesh),
                g_shape,
                self.cfg.param_dtype,
            ).astype(self.cfg.dtype)[None, None, ...]
        return normed


class RotaryEncoding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    is_keys: bool

    @nn.compact
    def __call__(self, x):
        *_, length, width = x.shape

        positions = jnp.arange(length)
        positions = sharding_constraint(positions, MESH_AXES["N"], self.global_mesh)
        positions = positions[..., None]  # expand along width axis
        positions = sharding_constraint(positions, MESH_AXES["NN"], self.global_mesh)

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        dimensions = sharding_constraint(dimensions, MESH_AXES["N"], self.global_mesh)
        ang_freqs = jnp.power(self.cfg.rotary_base, -dimensions / (width // 2))
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

        broadcast = self.cfg.kv_mqa and self.is_keys
        mesh_axes = MESH_AXES["XNNN"] if broadcast else MESH_AXES["XYNN"]

        even, odd = jnp.split(x, 2, axis=-1)
        even = sharding_constraint(even, mesh_axes, self.global_mesh)
        odd = sharding_constraint(odd, mesh_axes, self.global_mesh)

        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        r_even = sharding_constraint(r_even, mesh_axes, self.global_mesh)
        r_odd = sharding_constraint(r_odd, mesh_axes, self.global_mesh)

        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, mesh_axes, self.global_mesh)
        chex.assert_shape(r, x.shape)
        return r


class RotaryEncodingV2(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    is_keys: bool

    @nn.compact
    def __call__(self, x, pos_ids):
        bsz, _, length, width = x.shape

        positions = jnp.arange(length)
        positions = sharding_constraint(positions, MESH_AXES["N"], self.global_mesh)
        positions = positions[..., None]  # expand along width axis
        positions = sharding_constraint(positions, MESH_AXES["NN"], self.global_mesh)

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        dimensions = sharding_constraint(dimensions, MESH_AXES["N"], self.global_mesh)
        ang_freqs = jnp.power(self.cfg.rotary_base, -dimensions / (width // 2))
        ang_freqs = ang_freqs[None, ...]  # expand along length axis
        ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NN"], self.global_mesh)

        # expand along leading axes, such as batch and head.
        positions = positions[None, None, ...]
        ang_freqs = ang_freqs[None, None, ...]
        positions = sharding_constraint(positions, MESH_AXES["NNNN"], self.global_mesh)
        ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(positions, [1, 1, length, 1])
        chex.assert_shape(ang_freqs, [1, 1, 1, width // 2])

        # to supporting decoding with batch of variable-length prefills:
        positions = positions + pos_ids[..., None, None, None]
        positions = sharding_constraint(positions, MESH_AXES["XNNN"], self.global_mesh)

        # rest is same as before
        radians = positions * ang_freqs
        radians = sharding_constraint(radians, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(radians, [bsz, 1, length, width // 2])

        cos = jnp.cos(radians).astype(x.dtype)
        sin = jnp.sin(radians).astype(x.dtype)
        cos = sharding_constraint(cos, MESH_AXES["NNNN"], self.global_mesh)
        sin = sharding_constraint(sin, MESH_AXES["NNNN"], self.global_mesh)
        chex.assert_shape(cos, [bsz, 1, length, width // 2])
        chex.assert_shape(sin, [bsz, 1, length, width // 2])

        broadcast = self.cfg.kv_mqa and self.is_keys
        mesh_axes = MESH_AXES["XNNN"] if broadcast else MESH_AXES["XYNN"]

        even, odd = jnp.split(x, 2, axis=-1)
        even = sharding_constraint(even, mesh_axes, self.global_mesh)
        odd = sharding_constraint(odd, mesh_axes, self.global_mesh)

        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        r_even = sharding_constraint(r_even, mesh_axes, self.global_mesh)
        r_odd = sharding_constraint(r_odd, mesh_axes, self.global_mesh)

        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, mesh_axes, self.global_mesh)
        chex.assert_shape(r, x.shape)
        return r


class CausalMask(nn.Module):
    present_length: int
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.present_length)[..., None]
        j = jnp.arange(self.present_length)[None, ...]
        i = sharding_constraint(i, MESH_AXES["NN"], self.global_mesh)
        j = sharding_constraint(j, MESH_AXES["NN"], self.global_mesh)
        mask = jnp.less(i, j)  # i.e., j > i, indicator masks out non-causal connections
        mask = sharding_constraint(mask, MESH_AXES["NN"], self.global_mesh)
        mask = mask[None, None, ...]
        mask = sharding_constraint(mask, MESH_AXES["NNNN"], self.global_mesh)
        x = x - jnp.array([INFTY_APPROX], dtype=x.dtype) * mask
        x = sharding_constraint(x, MESH_AXES["XYNN"], self.global_mesh)
        return x


class CacheMask(nn.Module):
    cache_size: int
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, pos_ids):
        # - kv cache on timestep t contains the last c keys/values from 0 thru t-1.
        # - the most recent entry on step t-1 is written to (t-1) % c.
        # - pos_ids should start at 0 when inputting the bos token in prefill.
        #     so the smallest pos_ids seen by CacheMask will be 1.
        q_len, kv_len = x.shape[-2], self.cache_size
        i = jnp.arange(q_len)[None, None, ..., None] + pos_ids[..., None, None, None]
        j = jnp.arange(kv_len)[None, None, None, ...]
        i = sharding_constraint(i, MESH_AXES["XNNN"], self.global_mesh)
        j = sharding_constraint(j, MESH_AXES["NNNN"], self.global_mesh)
        mask = jnp.less_equal(i, j)  # masks out non-populated buffer elements
        mask = sharding_constraint(mask, MESH_AXES["XNNN"], self.global_mesh)
        x = x - jnp.array([INFTY_APPROX], dtype=x.dtype) * mask
        x = sharding_constraint(x, MESH_AXES["XYNN"], self.global_mesh)
        return x


class MultiHeadAttention(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, state):
        bsz = x.shape[0]
        shapes = Dimensions(
            B=bsz,
            T=self.cfg.sequence_len,
            M=self.cfg.d_model,
            D=self.cfg.d_head,
            H=self.cfg.d_model // self.cfg.d_head,
            I=1,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        stddev = self.cfg.d_model**-0.5
        q_init = {"zero": init.zeros, "vs": init.normal(stddev)}[self.cfg.q_init]
        kv_init = init.normal(stddev)
        o_init = {"zero": init.zeros, "vs": init.normal(stddev)}[self.cfg.r_init]
        b_init = init.zeros

        w_kv_mesh_axes = MESH_AXES["XNN"] if self.cfg.kv_mqa else MESH_AXES["XYN"]
        w_kv_shape = shapes["MID"] if self.cfg.kv_mqa else shapes["MHD"]
        wq = self.param(
            "w_aq",
            nn.with_partitioning(q_init, MESH_AXES["XYN"], self.global_mesh),
            shapes["MHD"],
            self.cfg.param_dtype,
        )
        wk = self.param(
            "w_ak",
            nn.with_partitioning(kv_init, w_kv_mesh_axes, self.global_mesh),
            w_kv_shape,
            self.cfg.param_dtype,
        )
        wv = self.param(
            "w_av",
            nn.with_partitioning(kv_init, w_kv_mesh_axes, self.global_mesh),
            w_kv_shape,
            self.cfg.param_dtype,
        )
        wo = self.param(
            "w_ao",
            nn.with_partitioning(o_init, MESH_AXES["YNX"], self.global_mesh),
            shapes["HDM"],
            self.cfg.param_dtype,
        )
        if self.cfg.proj_biases:
            b_kv_mesh_axes = MESH_AXES["NN"] if self.cfg.kv_mqa else MESH_AXES["YN"]
            b_kv_shape = shapes["ID"] if self.cfg.kv_mqa else shapes["HD"]
            bq = self.param(
                "b_aq",
                nn.with_partitioning(b_init, MESH_AXES["YN"], self.global_mesh),
                shapes["HD"],
                self.cfg.param_dtype,
            )
            bk = self.param(
                "b_ak",
                nn.with_partitioning(b_init, b_kv_mesh_axes, self.global_mesh),
                b_kv_shape,
                self.cfg.param_dtype,
            )
            bv = self.param(
                "b_av",
                nn.with_partitioning(b_init, b_kv_mesh_axes, self.global_mesh),
                b_kv_shape,
                self.cfg.param_dtype,
            )
            bo = self.param(
                "b_ao",
                nn.with_partitioning(b_init, MESH_AXES["Y"], self.global_mesh),
                shapes["M"],
                self.cfg.param_dtype,
            )

        kv_mesh_axes = MESH_AXES["XNNN"] if self.cfg.kv_mqa else MESH_AXES["XYNN"]
        q = jnp.einsum("bim,mhd->bhid", x, wq.astype(self.cfg.dtype))
        k = jnp.einsum("bim,mhd->bhid", x, wk.astype(self.cfg.dtype))
        v = jnp.einsum("bim,mhd->bhid", x, wv.astype(self.cfg.dtype))
        q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
        k = sharding_constraint(k, kv_mesh_axes, self.global_mesh)
        v = sharding_constraint(v, kv_mesh_axes, self.global_mesh)
        if self.cfg.proj_biases:
            q += jnp.expand_dims(bq.astype(self.cfg.dtype), (0, 2))  # noqa
            k += jnp.expand_dims(bk.astype(self.cfg.dtype), (0, 2))  # noqa
            v += jnp.expand_dims(bv.astype(self.cfg.dtype), (0, 2))  # noqa
            q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
            k = sharding_constraint(k, kv_mesh_axes, self.global_mesh)
            v = sharding_constraint(v, kv_mesh_axes, self.global_mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        if self.cfg.qk_norm:
            # maybe should not use with gains since the gains will be tied for all heads
            q = RMSNorm(self.cfg, self.global_mesh, "aq")(q)
            k = RMSNorm(self.cfg, self.global_mesh, "ak")(k)

        if self.cfg.rotary_base > 0:
            rope_kws = dict(cfg=self.cfg, global_mesh=self.global_mesh)
            if state is not None:
                pos_ids = state["pos_ids"]
            else:
                pos_ids = jnp.zeros([bsz], dtype=jnp.int32)
            q = RotaryEncodingV2(**rope_kws, is_keys=False)(q, pos_ids=pos_ids)
            k = RotaryEncodingV2(**rope_kws, is_keys=True)(k, pos_ids=pos_ids)
            q = sharding_constraint(q, MESH_AXES["XYNN"], self.global_mesh)
            k = sharding_constraint(k, kv_mesh_axes, self.global_mesh)
            self.sow("intermediates", "aqr_l1", coord_check_l1(q))
            self.sow("intermediates", "akr_l1", coord_check_l1(k))

        mult = jnp.array([self.cfg.qk_scale**0.5], dtype=self.cfg.dtype)
        s = jnp.einsum("bhid,bhjd->bhij", q * mult, k * mult)
        s = sharding_constraint(s, MESH_AXES["XYNN"], self.global_mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        if not self.cfg.is_decoding:
            s = CausalMask(self.cfg.sequence_len, self.global_mesh)(s)
            s = sharding_constraint(s, MESH_AXES["XYNN"], self.global_mesh)
            p = jax.nn.softmax(s, axis=-1)
            p = sharding_constraint(p, MESH_AXES["XYNN"], self.global_mesh)
            self.sow("intermediates", "ap_l1", coord_check_l1(p))
            o = jnp.einsum("bhij,bhjd->bhid", p, v)
            o = sharding_constraint(o, MESH_AXES["XYNN"], self.global_mesh)
            self.sow("intermediates", "ao_l1", coord_check_l1(o))
            # for prefill. pos_ids will be written at the end of Transformer.__call__.
            new_state = dict(k=k, v=v, pos_ids=None)
        else:
            # incremental decoding.
            mesh, slen = self.global_mesh, self.cfg.sequence_len
            s_cache = jnp.einsum("bhid,bhjd->bhij", q * mult, state["k"] * mult)
            s_cache = sharding_constraint(s_cache, MESH_AXES["XYNN"], mesh)
            s_cache = CacheMask(slen, mesh)(s_cache, pos_ids=state["pos_ids"])
            s_cache = sharding_constraint(s_cache, MESH_AXES["XYNN"], mesh)
            s = jnp.concatenate([s_cache, s], axis=-1)
            p = jax.nn.softmax(s, axis=-1)
            p = sharding_constraint(p, MESH_AXES["XYNN"], mesh)
            o = jnp.einsum(
                "bhij,bhjd->bhid", p, jnp.concatenate([state["v"], v], axis=-2)
            )
            o = sharding_constraint(o, MESH_AXES["XYNN"], mesh)
            write_weights = jax.nn.one_hot(
                jnp.mod(
                    state["pos_ids"],
                    jnp.full_like(state["pos_ids"], fill_value=slen),
                ),
                num_classes=slen,
                axis=-1,
                dtype=self.cfg.dtype,
            )
            write_weights = jnp.expand_dims(jnp.expand_dims(write_weights, 1), -1)
            new_state = dict(
                k=(1.0 - write_weights) * state["k"] + write_weights * k,
                v=(1.0 - write_weights) * state["v"] + write_weights * v,
                pos_ids=state["pos_ids"] + 1,
            )

        r = jnp.einsum("bhid,hdm->bim", o, wo.astype(self.cfg.dtype))
        r = sharding_constraint(r, MESH_AXES["XNY"], self.global_mesh)
        if self.cfg.proj_biases:
            r += bo.astype(self.cfg.dtype)[None, None, ...]  # noqa
            r = sharding_constraint(r, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "ar_l1", coord_check_l1(r))
        return r, new_state


class MultiLayerPerceptron(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        d_ff_in = int(self.cfg.ff_multiple * self.cfg.d_model)
        if self.cfg.act_name == "swiglu":
            d_ff_in = (d_ff_in // 2) * 2
            d_ff_out = d_ff_in // 2
        else:
            d_ff_out = d_ff_in

        shapes = Dimensions(
            B=x.shape[0],
            T=self.cfg.sequence_len,
            M=self.cfg.d_model,
            E=d_ff_in,
            F=d_ff_out,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        i_init = init.normal(self.cfg.d_model**-0.5)
        o_init = {
            "zero": init.zeros,
            "vs": init.normal(d_ff_out**-0.5),
        }[self.cfg.r_init]
        b_init = init.zeros

        wi = self.param(
            "w_fi",
            nn.with_partitioning(i_init, MESH_AXES["XY"], self.global_mesh),
            shapes["ME"],
            self.cfg.param_dtype,
        )
        wo = self.param(
            "w_fo",
            nn.with_partitioning(o_init, MESH_AXES["YX"], self.global_mesh),
            shapes["FM"],
            self.cfg.param_dtype,
        )
        if self.cfg.proj_biases:
            bi = self.param(
                "b_fi",
                nn.with_partitioning(b_init, MESH_AXES["Y"], self.global_mesh),
                shapes["E"],
                self.cfg.param_dtype,
            )
            bo = self.param(
                "b_fo",
                nn.with_partitioning(b_init, MESH_AXES["Y"], self.global_mesh),
                shapes["M"],
                self.cfg.param_dtype,
            )

        x = jnp.einsum("btm,me->bte", x, wi.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        if self.cfg.proj_biases:
            x += bi.astype(self.cfg.dtype)[None, None, ...]  # noqa
            x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fh_l1", coord_check_l1(x))

        if self.cfg.act_name == "swiglu":
            # a more communication-efficient implementation of swiglu would define
            # two separate projections for xg, xf with the same sharding.
            xg, xf = jnp.split(x, 2, axis=-1)
            x = jax.nn.silu(xg) * xf
        else:
            x = getattr(jax.nn, self.cfg.act_name)(x)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        if self.cfg.act_square:
            x = jnp.square(x)
            x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        x = jnp.einsum("btf,fm->btm", x, wo.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        if self.cfg.proj_biases:
            x += bo.astype(self.cfg.dtype)[None, None, ...]  # noqa
            x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, state):
        kws = dict(cfg=self.cfg, global_mesh=self.global_mesh)

        ao, state_new = MultiHeadAttention(**kws)(RMSNorm(**kws, suffix="a")(x), state)
        x += ao
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        fo = MultiLayerPerceptron(**kws)(RMSNorm(**kws, suffix="f")(x))
        x += fo
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        return x, state_new


class Embedding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        w_emb = self.param(
            "w_e",
            nn.with_partitioning(init.normal(1.0), MESH_AXES["NY"], self.global_mesh),
            [self.cfg.n_vocab, self.cfg.d_model],
            self.cfg.param_dtype,
        )
        x = sharding_constraint(x, MESH_AXES["XN"], self.global_mesh)
        x = jnp.take_along_axis(
            w_emb.astype(self.cfg.dtype)[None, ...],  # 1VM
            x[..., None],  # BT1
            axis=1,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        if self.cfg.e_norm:
            x = RMSNorm(self.cfg, self.global_mesh, "e")(x)
            x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        return x


class Unembedding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        x = RMSNorm(self.cfg, self.global_mesh, "u")(x)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)

        stddev = self.cfg.d_model**-0.5
        u_init = {
            "zero": init.zeros,
            "sp": init.normal(stddev),
            "mup": init.normal(stddev**2),
        }[self.cfg.u_init]
        b_init = init.zeros
        wu = self.param(
            "w_u",
            nn.with_partitioning(u_init, MESH_AXES["YN"], self.global_mesh),
            [self.cfg.d_model, self.cfg.n_vocab],
            self.cfg.param_dtype,
        )
        if self.cfg.proj_biases:
            bu = self.param(
                "b_u",
                nn.with_partitioning(b_init, MESH_AXES["N"], self.global_mesh),
                [self.cfg.n_vocab],
                self.cfg.param_dtype,
            )

        out_dtype = self.cfg.dtype if self.cfg.is_train else self.cfg.param_dtype
        x = jnp.einsum("btm,mv->btv", x, wu.astype(out_dtype))
        x = sharding_constraint(x, MESH_AXES["XNN"], self.global_mesh)
        if self.cfg.proj_biases:
            x += bu.astype(out_dtype)[None, None, ...]  # noqa
            x = sharding_constraint(x, MESH_AXES["XNN"], self.global_mesh)
        return x


class Transformer(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        kv_cache: Optional[Dict[str, jax.Array]] = None,
    ) -> Dict[str, Any]:
        # kv_cache is an optional dict with fields "k", "v", and "pos_ids".
        #     the k/v arrays have shape [n_layer, bsz, n_head, sequence_len, d_head].
        #     each of them is a circular buffer supporting sliding window attn.
        #
        # there are two modes of operation:
        #     parallel (kv_cache is None, is_decoding=False):
        #         input kv cache ignored, kv cache still outputted to facilitate prefill
        #     sequential (kv_cache given, is_decoding=True):
        #         input kv cache used in attn, new kv cache outputted for next decode
        #
        x = nnp.remat(Embedding)(self.cfg, self.global_mesh)(inputs)
        x, kv_cache_new = nn.scan(
            nnp.remat(TransformerBlock),
            length=self.cfg.n_layer,
            variable_axes=dict(params=0, intermediates=0),  # use axis 0 for params,sown
            variable_broadcast=False,  # no variable sharing across layers
            split_rngs=dict(params=True),  # each layer's init shall use a distinct rng
            in_axes=0,  # use n_layer first for inputted kv cache
            out_axes=0,  # use n_layer first for outputted kv cache
            metadata_params={nn.PARTITION_NAME: None},  # no pipeline parallel
        )(cfg=self.cfg, global_mesh=self.global_mesh)(x, kv_cache)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.global_mesh)
        x = nnp.remat(Unembedding)(self.cfg, self.global_mesh)(x)
        if not self.cfg.is_decoding:
            # prefill: in this case, count the eos/pad tokens in batch of prompts.
            #  the padding should always be on the right; the first token may be bos,
            #  which for some tokenizers may equal eos/pad, so we exclude it from count
            chex.assert_shape(inputs, (None, self.cfg.sequence_len))
            npad = jnp.sum(
                jnp.logical_or(
                    jnp.equal(inputs[:, 1:], self.cfg.pad_token_id),
                    jnp.equal(inputs[:, 1:], self.cfg.eos_token_id),
                ),
                axis=-1,
            )
            npad = jnp.tile(npad[None, ...], reps=[self.cfg.n_layer, 1])
            npad = sharding_constraint(npad, MESH_AXES["NX"], self.global_mesh)
            pos_ids = self.cfg.sequence_len - npad  # eg slen=5 npad=2 ==> next pos_id=3
            kv_cache_new["pos_ids"] = pos_ids
        return dict(logits=x, kv_cache=kv_cache_new)
