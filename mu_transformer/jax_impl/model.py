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

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.linen import partitioning as nnp
from jax.nn.initializers import normal as gauss
from jax.nn.initializers import zeros as zero

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
    dm: int
    dh: int
    kv_group_size: int
    g_type: str
    v_type: str
    rotary_base: int
    ff_act_name: str
    ff_act_square: bool
    ff_multiple: int
    norm_eps: float
    nl: int
    nv: int
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
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh
    suffix: str

    @nn.compact
    def __call__(self, x):
        eps = jnp.array([self.cfg.norm_eps], dtype=x.dtype)
        ms = jnp.mean(jnp.square(x), axis=-1)
        rms = jnp.sqrt(ms + eps)
        normed = x / rms[..., None]
        return normed


class RotaryEncoding(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh
    is_keys: bool

    @nn.compact
    def __call__(self, x):
        *_, length, width = x.shape

        positions = jnp.arange(length)
        positions = sharding_constraint(positions, MESH_AXES["N"], self.mesh)
        positions = positions[..., None]  # expand along width axis
        positions = sharding_constraint(positions, MESH_AXES["NN"], self.mesh)

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        dimensions = sharding_constraint(dimensions, MESH_AXES["N"], self.mesh)
        ang_freqs = jnp.power(self.cfg.rotary_base, -dimensions / (width // 2))
        ang_freqs = ang_freqs[None, ...]  # expand along length axis
        ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NN"], self.mesh)

        # expand along leading axes, such as batch and head.
        positions = positions[None, None, ...]
        ang_freqs = ang_freqs[None, None, ...]
        positions = sharding_constraint(positions, MESH_AXES["NNNN"], self.mesh)
        ang_freqs = sharding_constraint(ang_freqs, MESH_AXES["NNNN"], self.mesh)
        chex.assert_shape(positions, [1, 1, length, 1])
        chex.assert_shape(ang_freqs, [1, 1, 1, width // 2])

        radians = positions * ang_freqs
        radians = sharding_constraint(radians, MESH_AXES["NNNN"], self.mesh)
        chex.assert_shape(radians, [1, 1, length, width // 2])

        if not self.is_keys:
            radians = radians[None, ...]
            radians = sharding_constraint(radians, MESH_AXES["NNNNN"], self.mesh)

        cos = jnp.cos(radians).astype(x.dtype)
        sin = jnp.sin(radians).astype(x.dtype)

        mesh_axes = MESH_AXES["XNNN"] if self.is_keys else MESH_AXES["XYNNN"]

        even, odd = jnp.split(x, 2, axis=-1)
        even = sharding_constraint(even, mesh_axes, self.mesh)
        odd = sharding_constraint(odd, mesh_axes, self.mesh)

        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos
        r_even = sharding_constraint(r_even, mesh_axes, self.mesh)
        r_odd = sharding_constraint(r_odd, mesh_axes, self.mesh)

        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, mesh_axes, self.mesh)
        chex.assert_shape(r, x.shape)
        return r


class CausalMask(nn.Module):
    present_length: int
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.present_length)[..., None]
        j = jnp.arange(self.present_length)[None, ...]
        i = sharding_constraint(i, MESH_AXES["NN"], self.mesh)
        j = sharding_constraint(j, MESH_AXES["NN"], self.mesh)
        mask = jnp.less(i, j)  # i.e., j > i, indicator masks out non-causal connections
        mask = sharding_constraint(mask, MESH_AXES["NN"], self.mesh)
        mask = mask[None, None, None, ...]
        mask = sharding_constraint(mask, MESH_AXES["NNNNN"], self.mesh)
        x = x - jnp.array([INFTY_APPROX], dtype=x.dtype) * mask
        x = sharding_constraint(x, MESH_AXES["XYNNN"], self.mesh)
        return x


def get_dim_names(bsz, cfg):
    shapes = Dimensions(
        B=bsz,
        T=cfg.sequence_len,
        M=cfg.dm,
        D=cfg.dh,
        G=cfg.kv_group_size,
        H=cfg.dm // (cfg.dh * cfg.kv_group_size),
        P=3,
    )
    return shapes


class QueryProjection(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "w_aq",
            nn.with_partitioning(zero, MESH_AXES["XYNN"], self.mesh),
            get_dim_names(None, self.cfg)["MGHD"],
            self.cfg.param_dtype,
        )
        x = jnp.einsum("bim,mghd->bghid", x, w.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XYNNN"], self.mesh)
        return x


class KeyProjection(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "w_ak",
            nn.with_partitioning(
                gauss(self.cfg.dm**-0.5), MESH_AXES["XYN"], self.mesh
            ),
            get_dim_names(None, self.cfg)["MGD"],
            self.cfg.param_dtype,
        )
        x = jnp.einsum("bim,mgd->bgid", x, w.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XYNN"], self.mesh)
        return x


class ValueProjection(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh
    suffix: str
    move_type: str

    @nn.compact
    def __call__(self, x):
        if self.move_type == "linear":
            init = gauss(self.cfg.dm**-0.5)
            w = self.param(
                "w_a" + self.suffix,
                nn.with_partitioning(init, MESH_AXES["XYN"], self.mesh),
                get_dim_names(None, self.cfg)["MGD"],
                self.cfg.param_dtype,
            )
            x = jnp.einsum("bim,mgd->bgid", x, w.astype(self.cfg.dtype))
            x = sharding_constraint(x, MESH_AXES["XYNN"], self.mesh)
            return x
        elif self.move_type == "depsepconv":
            init = gauss(self.cfg.dm**-0.5)
            w = self.param(
                "w_a" + self.suffix,
                nn.with_partitioning(init, MESH_AXES["XYN"], self.mesh),
                get_dim_names(None, self.cfg)["MGD"],
                self.cfg.param_dtype,
            )
            x = jnp.einsum("bim,mgd->bgid", x, w.astype(self.cfg.dtype))
            x = sharding_constraint(x, MESH_AXES["XYNN"], self.mesh)
            init = gauss(3**-0.5)
            s = self.param(
                "s_a" + self.suffix,
                nn.with_partitioning(init, MESH_AXES["NYN"], self.mesh),
                get_dim_names(None, self.cfg)["PGD"],
                self.cfg.param_dtype,
            )
            x = (
                jnp.pad(x[:, :, 0:-2, :], ((0, 0), (0, 0), (2, 0), (0, 0))) * s[-3]
                + jnp.pad(x[:, :, 0:-1, :], ((0, 0), (0, 0), (1, 0), (0, 0))) * s[-2]
                + x * s[-1]
            )
            x = sharding_constraint(x, MESH_AXES["XYNN"], self.mesh)
            return x
        elif self.move_type == "conv":
            init = gauss((3 * self.cfg.dm) ** -0.5)
            c = self.param(
                "c_a" + self.suffix,
                nn.with_partitioning(init, MESH_AXES["NXYN"], self.mesh),
                get_dim_names(None, self.cfg)["PMGD"],
                self.cfg.param_dtype,
            )
            x = (
                jnp.einsum(
                    "bim,mgd->bgid",
                    jnp.pad(x[:, 0:-2, :], ((0, 0), (2, 0), (0, 0))),
                    c[-3].astype(self.cfg.dtype),
                )
                + jnp.einsum(
                    "bim,mgd->bgid",
                    jnp.pad(x[:, 0:-1, :], ((0, 0), (1, 0), (0, 0))),
                    c[-2].astype(self.cfg.dtype),
                )
                + jnp.einsum("bim,mgd->bgid", x, c[-1].astype(self.cfg.dtype))
            )
            x = sharding_constraint(x, MESH_AXES["XYNN"], self.mesh)
            return x
        else:
            raise NotImplementedError


class OutputProjection(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "w_ao",
            nn.with_partitioning(zero, MESH_AXES["YNNX"], self.mesh),
            get_dim_names(None, self.cfg)["GHDM"],
            self.cfg.param_dtype,
        )
        x = jnp.einsum("bghid,ghdm->bim", x, w.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        return x


class MultiHeadAttention(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        bsz = x.shape[0]
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        self.sow("intermediates", "ax_l1", coord_check_l1(x))

        q = QueryProjection(self.cfg, self.mesh)(x)
        k = KeyProjection(self.cfg, self.mesh)(x)
        v = ValueProjection(
            cfg=self.cfg,
            mesh=self.mesh,
            suffix="v",
            move_type=self.cfg.v_type,
        )(x)
        if self.cfg.g_type != "none":
            g = ValueProjection(
                cfg=self.cfg,
                mesh=self.mesh,
                suffix="g",
                move_type=self.cfg.g_type,
            )(x)
            g = jax.nn.silu(g)
            v = g * v
        q = sharding_constraint(q, MESH_AXES["XYNNN"], self.mesh)
        k = sharding_constraint(k, MESH_AXES["XYNN"], self.mesh)
        v = sharding_constraint(v, MESH_AXES["XYNN"], self.mesh)
        self.sow("intermediates", "aq_l1", coord_check_l1(q))
        self.sow("intermediates", "ak_l1", coord_check_l1(k))
        self.sow("intermediates", "av_l1", coord_check_l1(v))

        rope_kws = dict(cfg=self.cfg, mesh=self.mesh)
        q = RotaryEncoding(**rope_kws, is_keys=False)(q)  # todo: use v2 for decode
        k = RotaryEncoding(**rope_kws, is_keys=True)(k)
        q = sharding_constraint(q, MESH_AXES["XYNNN"], self.mesh)
        k = sharding_constraint(k, MESH_AXES["XNNN"], self.mesh)
        self.sow("intermediates", "aqr_l1", coord_check_l1(q))
        self.sow("intermediates", "akr_l1", coord_check_l1(k))

        mult = jnp.array([self.cfg.dh**-0.5], dtype=self.cfg.dtype)
        s = jnp.einsum("bghid,bgjd->bghij", q * mult, k * mult)
        s = sharding_constraint(s, MESH_AXES["XYNNN"], self.mesh)
        self.sow("intermediates", "as_l1", coord_check_l1(s))

        s = CausalMask(self.cfg.sequence_len, self.mesh)(s)
        s = sharding_constraint(s, MESH_AXES["XYNNN"], self.mesh)

        p = jax.nn.softmax(s, axis=-1)
        p = sharding_constraint(p, MESH_AXES["XYNNN"], self.mesh)
        self.sow("intermediates", "ap_l1", coord_check_l1(p))

        o = jnp.einsum("bghij,bgjd->bghid", p, v)
        o = sharding_constraint(o, MESH_AXES["XYNNN"], self.mesh)
        self.sow("intermediates", "ao_l1", coord_check_l1(o))

        r = OutputProjection(self.cfg, self.mesh)(o)
        self.sow("intermediates", "ar_l1", coord_check_l1(r))
        return r


class MultiLayerPerceptron(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        d_ff = int(self.cfg.ff_multiple * self.cfg.dm)
        if self.cfg.act_name == "swiglu":
            d_ff_pre = d_ff * 2
            d_ff_post = d_ff
        else:
            d_ff_pre = d_ff
            d_ff_post = d_ff

        shapes = Dimensions(
            B=x.shape[0],
            T=self.cfg.sequence_len,
            M=self.cfg.dm,
            E=d_ff_pre,
            F=d_ff_post,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        self.sow("intermediates", "fx_l1", coord_check_l1(x))

        i_init = gauss(self.cfg.dm**-0.5)
        o_init = gauss(d_ff_post**-0.5)

        wi = self.param(
            "w_fi",
            nn.with_partitioning(i_init, MESH_AXES["XY"], self.mesh),
            shapes["ME"],
            self.cfg.param_dtype,
        )
        wo = self.param(
            "w_fo",
            nn.with_partitioning(o_init, MESH_AXES["YX"], self.mesh),
            shapes["FM"],
            self.cfg.param_dtype,
        )

        x = jnp.einsum("btm,me->bte", x, wi.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        self.sow("intermediates", "fh_l1", coord_check_l1(x))

        if self.cfg.act_name == "swiglu":
            # a more communication-efficient implementation of swiglu would define
            # two separate projections for xg, xf with the same sharding.
            xg, xf = jnp.split(x, 2, axis=-1)
            x = jax.nn.silu(xg) * xf
        else:
            x = getattr(jax.nn, self.cfg.act_name)(x)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)

        if self.cfg.act_square:
            x = jnp.square(x)
            x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        self.sow("intermediates", "fa_l1", coord_check_l1(x))

        x = jnp.einsum("btf,fm->btm", x, wo.astype(self.cfg.dtype))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        self.sow("intermediates", "fr_l1", coord_check_l1(x))
        return x


class TransformerBlock(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        kws = dict(cfg=self.cfg, mesh=self.mesh)
        x += MultiHeadAttention(**kws)(RMSNorm(**kws, suffix="a")(x))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        x += MultiLayerPerceptron(**kws)(RMSNorm(**kws, suffix="f")(x))
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        return x, None


class Embedding(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        w_emb = self.param(
            "w_e",
            nn.with_partitioning(gauss(1.0), MESH_AXES["NY"], self.mesh),
            [self.cfg.nv, self.cfg.dm],
            self.cfg.param_dtype,
        )
        x = sharding_constraint(x, MESH_AXES["XN"], self.mesh)
        x = jnp.take_along_axis(
            w_emb.astype(self.cfg.dtype)[None, ...],  # 1VM
            x[..., None],  # BT1
            axis=1,
        )
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        return x


class Unembedding(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        x = RMSNorm(self.cfg, self.mesh, "u")(x)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        wu = self.param(
            "w_u",
            nn.with_partitioning(zero, MESH_AXES["YN"], self.mesh),
            [self.cfg.dm, self.cfg.nv],
            self.cfg.param_dtype,
        )
        out_dtype = self.cfg.dtype if self.cfg.is_train else self.cfg.param_dtype
        x = jnp.einsum("btm,mv->btv", x, wu.astype(out_dtype))
        x = sharding_constraint(x, MESH_AXES["XNN"], self.mesh)
        return x


class Transformer(nn.Module):
    cfg: TransformerConfig
    mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, inputs: jax.Array) -> Dict[str, Any]:
        x = nnp.remat(Embedding)(self.cfg, self.mesh)(inputs)
        x, _ = nn.scan(
            nnp.remat(TransformerBlock),
            length=self.cfg.nl,
            variable_axes=dict(params=0, intermediates=0),  # use axis 0 for params,sown
            variable_broadcast=False,  # no variable sharing across layers
            split_rngs=dict(params=True),  # each layer's init shall use a distinct rng
            in_axes=0,  # use n_layer first for inputted kv cache
            out_axes=0,  # use n_layer first for outputted kv cache
            metadata_params={nn.PARTITION_NAME: None},  # no pipeline parallel
        )(cfg=self.cfg, mesh=self.mesh)(x, None)
        x = sharding_constraint(x, MESH_AXES["XNY"], self.mesh)
        x = nnp.remat(Unembedding)(self.cfg, self.mesh)(x)
        return dict(logits=x)
