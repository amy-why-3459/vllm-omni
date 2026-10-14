# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gather both CFM convolution histories directly into the Ascend CNN cache."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import (
    causal_cache_index,
    gather_causal_cache,
)


@triton.jit
def _write_pair(
    h1,
    h2,
    lengths,
    out,
    c1: tl.constexpr,
    c2: tl.constexpr,
    width: tl.constexpr,
    a0: tl.constexpr,
    a1: tl.constexpr,
    a2: tl.constexpr,
    b0: tl.constexpr,
    b1: tl.constexpr,
    b2: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    o2: tl.constexpr,
    ls: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.program_id(1) * block + tl.arange(0, block)
    c, offset = i // width, i % width
    live = i < (c1 + c2) * width
    t = tl.load(lengths + row * ls) + offset
    a = tl.load(h1 + row * a0 + t * a1 + c * a2, live & (c < c1), other=0)
    b = tl.load(h2 + row * b0 + t * b1 + (c - c1) * b2, live & (c >= c1), other=0)
    tl.store(out + row * o0 + c * o1 + offset * o2, tl.where(c < c1, a, b), live)


def write_cache(history: torch.Tensor, history2: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor) -> None:
    """Write both NTC histories into the NCW cache without intermediate tensors.

    The caller supplies valid lengths in [0, frames]. Histories include the
    W-frame prefix and must not alias the output; the DiT body materializes
    them before updating its reusable CNN cache buffer.
    """
    width = out.shape[-1]
    if width == 0:
        return
    if out.device.type != "npu" or not HAS_TRITON:
        index = causal_cache_index(lengths, width)
        split = history.shape[-1]
        out[:, :split].copy_(gather_causal_cache(history, index))
        out[:, split:].copy_(gather_causal_cache(history2, index))
        return
    rows, channels, _ = out.shape
    _write_pair[(rows, triton.cdiv(channels * width, 512))](
        history,
        history2,
        lengths,
        out,
        history.shape[-1],
        history2.shape[-1],
        width,
        *history.stride(),
        *history2.stride(),
        *out.stride(),
        lengths.stride(0),
        512,
    )
