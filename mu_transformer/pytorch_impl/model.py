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
import dataclasses
import functools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.utils.checkpoint import checkpoint as remat

from mu_transformer.dims import Dimensions
from mu_transformer.pytorch_impl.sow import Intermediates

# from flash_attention import flash_attn_func

INF_APPROX = 1e30


@dataclasses.dataclass
class TransformerConfig:
    sow_intermediates: bool
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
    device: torch.device

    @classmethod
    def create(cls, **kwargs):
        signature = {f.name: f.type for f in dataclasses.fields(TransformerConfig)}
        flt = {k: v for k, v in kwargs.items() if k in signature}
        flt.update(
            {k: getattr(torch, v) for k, v in flt.items() if k.endswith("dtype")}
        )
        return cls(**flt)


class RMSNorm(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps

    def forward(self, x):
        ms = torch.mean(torch.pow(x, 2), dim=-1)
        rms = torch.sqrt(ms + self.hps.norm_eps)
        return x / rms[..., None]


@functools.lru_cache(maxsize=1)
def sinusoidal_embs(length, width, base, dtype, device):
    positions = torch.arange(length, device=device)
    dimensions = torch.arange(width // 2, device=device)
    ang_freqs = torch.pow(base, -dimensions / (width // 2))

    # expand to a shape broadcastable with q/k dims
    positions = torch.reshape(positions, [1, length, 1, 1])
    ang_freqs = torch.reshape(ang_freqs, [1, 1, 1, width // 2])

    radians = positions * ang_freqs
    cos = torch.cos(radians).to(dtype)
    sin = torch.sin(radians).to(dtype)
    return cos, sin


class RotaryEncoding(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps

    def forward(self, x):
        _, length, _, width = x.shape  # B, T, H, D

        cos, sin = sinusoidal_embs(
            length=length,
            width=width,
            base=self.hps.rotary_base,
            dtype=x.dtype,
            device=self.hps.device,
        )

        even, odd = torch.chunk(x, 2, dim=-1)
        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos

        r = torch.cat([r_even, r_odd], dim=-1)
        return r


class MultiheadSelfAttention(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps
        self.shapes = Dimensions(
            M=self.hps.d_model,
            H=self.hps.d_model // self.hps.d_head,
            D=self.hps.d_head,
        )

        self.w_aq = nn.parameter.Parameter(
            torch.zeros(
                size=self.shapes["MHD"],
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )
        self.w_ak = nn.parameter.Parameter(
            torch.normal(
                mean=0.0,
                std=self.hps.d_model**-0.5,
                size=self.shapes["MHD"],
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )
        self.w_av = nn.parameter.Parameter(
            torch.normal(
                mean=0.0,
                std=self.hps.d_model**-0.5,
                size=self.shapes["MHD"],
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )
        self.w_ao = nn.parameter.Parameter(  # table 3 with nh*dh=dm; var mult 1/2l
            torch.normal(  # table 3 with nh*dh=dm; var mult 1/2l
                mean=0.0,
                std=(self.hps.d_model * 2 * self.hps.n_layer) ** -0.5,
                size=self.shapes["HDM"],
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )
        self.rope = RotaryEncoding(self.hps)

    def forward(self, x, intermediates, layer_id):
        intermediates.coord_check_l1("ax_l1", x, layer_id)

        q = torch.einsum("btm,mhd->bthd", x, self.w_aq.to(self.hps.dtype))
        k = torch.einsum("btm,mhd->bthd", x, self.w_ak.to(self.hps.dtype))
        v = torch.einsum("btm,mhd->bthd", x, self.w_av.to(self.hps.dtype))
        intermediates.coord_check_l1("aq_l1", q, layer_id)
        intermediates.coord_check_l1("ak_l1", k, layer_id)
        intermediates.coord_check_l1("av_l1", v, layer_id)

        if self.hps.rotary_base > 0:
            q = self.rope(q)
            k = self.rope(k)
            intermediates.coord_check_l1("aqr_l1", q, layer_id)
            intermediates.coord_check_l1("akr_l1", k, layer_id)

        # current flash impl doesnt allow storing intermediates like the avg qk scale
        # o = flash_attn_func(q, k, v, softmax_scale=1.0 / self.hps.d_head, causal=True)

        s = torch.einsum("bihd,bjhd->bhij", q, k)
        s = s / self.hps.d_head
        intermediates.coord_check_l1("as_l1", s, layer_id)

        i = torch.arange(x.shape[1], device=self.hps.device)[..., None]
        j = torch.arange(x.shape[1], device=self.hps.device)[None, ...]
        mask = torch.less(i, j)  # i.e., j > i, indicator masks out non-causal
        mask = mask[None, None, ...]
        s = s - torch.tensor([INF_APPROX], dtype=x.dtype, device=self.hps.device) * mask

        p = F.softmax(s, dim=-1)
        intermediates.coord_check_l1("ap_l1", p, layer_id)

        o = torch.einsum("bhij,bjhd->bihd", p, v)
        intermediates.coord_check_l1("ao_l1", o, layer_id)

        r = torch.einsum("bihd,hdm->bim", o, self.w_ao.to(self.hps.dtype))
        intermediates.coord_check_l1("ar_l1", r, layer_id)
        return r, intermediates


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps
        self.shapes = Dimensions(
            M=self.hps.d_model,
            F=self.hps.d_model * self.hps.ff_multiple,
        )

        self.w_fi = nn.parameter.Parameter(
            torch.normal(
                mean=0.0,
                std=self.hps.d_model**-0.5,  # table 3
                size=self.shapes["MF"],
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )
        self.w_fo = nn.parameter.Parameter(
            torch.normal(
                mean=0.0,
                std=(self.hps.d_model * 2 * self.hps.n_layer) ** -0.5,  # table 3
                size=self.shapes["FM"],
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )

    def forward(self, x, intermediates, layer_id):
        intermediates.coord_check_l1("fx_l1", x, layer_id)

        x = torch.einsum("btm,mf->btf", x, self.w_fi.to(self.hps.dtype))
        intermediates.coord_check_l1("fp_l1", x, layer_id)

        x = getattr(F, self.hps.act_name)(x)
        if self.hps.act_square:
            x = torch.pow(x, 2)
        intermediates.coord_check_l1("fa_l1", x, layer_id)

        x = torch.einsum("btf,fm->btm", x, self.w_fo.to(self.hps.dtype))
        intermediates.coord_check_l1("fr_l1", x, layer_id)
        return x, intermediates


class TransformerBlock(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps
        self.norm1 = RMSNorm(self.hps)
        self.mha = MultiheadSelfAttention(self.hps)
        self.norm2 = RMSNorm(self.hps)
        self.mlp = MultiLayerPerceptron(self.hps)

    def _forward(self, x, intermediates, layer_id):
        r1, intermediates = self.mha(self.norm1(x), intermediates, layer_id)
        x = x + r1
        r2, intermediates = self.mlp(self.norm2(x), intermediates, layer_id)
        x = x + r2
        return x, intermediates

    def forward(self, x, intermediates, layer_id):
        return remat(self._forward, x, intermediates, layer_id, use_reentrant=True)
        # return self._forward(x, intermediates, layer_id)


class Embedding(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps
        self.w_ei = nn.parameter.Parameter(
            torch.normal(  # appendix b.1
                mean=0.0,
                std=1.0,
                size=(self.hps.n_vocab, self.hps.d_model),
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )

    def forward(self, x):
        x = F.embedding(input=x, weight=self.w_ei.to(self.hps.dtype))
        return x


class PredictionHead(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps
        self.norm = RMSNorm(self.hps)
        self.w_eo = nn.parameter.Parameter(
            torch.zeros(  # appendix d.2
                size=(self.hps.d_model, self.hps.n_vocab),
                dtype=self.hps.param_dtype,
                device=self.hps.device,
            ),
        )

    def forward(self, x):
        x = self.norm(x)
        if self.training:
            output_logits_dtype = self.hps.output_logits_dtype
        else:
            output_logits_dtype = self.hps.param_dtype
        x = torch.einsum("btm,mv->btv", x, self.w_eo.to(output_logits_dtype))
        return x


class Transformer(nn.Module):
    def __init__(self, hps: TransformerConfig) -> None:
        super().__init__()
        self.hps = hps
        self.embed = Embedding(self.hps)
        self.stack = nn.ModuleList(
            [TransformerBlock(self.hps) for _ in range(self.hps.n_layer)]
        )
        self.predict = PredictionHead(self.hps)

    def forward(self, x):
        intermediates = Intermediates(enabled=self.hps.sow_intermediates)
        x = F.pad(x[:, 0:-1], (1, 0), value=self.hps.bos_token_id)
        x = self.embed(x)
        for layer_id in range(self.hps.n_layer):
            x, intermediates = self.stack[layer_id](x, intermediates, layer_id)
        x = self.predict(x)
        return dict(
            logprobs=F.log_softmax(x, dim=-1),
            intermediates=intermediates,
        )
