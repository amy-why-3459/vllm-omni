# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only MiniCPM DiT projections and attention on Ascend."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import blocks_forward_chunk_fused, supports_fused_body
from vllm_omni.model_executor.models.minicpmo_4_5.ops.fused_residual_layer_norm import residual_layer_norm

from .cosyvoice2_dit_attn import _expand_attn_mask_for_npu
from .minicpmo_causal_cache import write_cache


def supports_packed_qkv(attn: nn.Module) -> bool:
    projections = [getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")]
    return (
        all(type(p) is nn.Linear for p in projections)
        and all(p.weight.shape == projections[0].weight.shape for p in projections)
        and (all(p.bias is None for p in projections) or all(p.bias is not None for p in projections))
    )


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and a.device == b.device
    )


def prepare_estimator(estimator: nn.Module, *, pack_convs: bool = False) -> None:
    """Establish weight storage before either eager-body or fused-body capture."""
    pack_convs = pack_convs and supports_fused_body(estimator)
    for block in getattr(estimator, "blocks", ()):
        attn = getattr(block, "attn", None)
        if attn is not None and not attn.training and supports_packed_qkv(attn):
            if attn.to_q.weight.device.type == "npu" and attn.to_q.weight.dtype == torch.float32:
                packed_qkv(attn)
        if pack_convs:
            for conv in (block.conv.block[1], block.conv.block[6]):
                if not conv.training and conv.weight.device.type == "npu" and conv.weight.dtype == torch.float32:
                    packed_conv_taps(conv)


def packed_qkv(attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack frozen projections once, retaining original Parameters as views.

    In-place weight loads then update the packed storage too. Replacing a
    parameter or moving the module rebuilds the pack on the next eager call.
    As with other device graphs, reloads require recapture. This also supports
    inference tensors, which have no version counter. Warm up before capture.
    """
    projections = (attn.to_q, attn.to_k, attn.to_v)
    cached = attn.__dict__.get("_npu_packed_qkv")
    rows = projections[0].out_features
    if cached is not None:
        weight, bias = cached
        if all(
            _same_view(p.weight, weight[i * rows : (i + 1) * rows])
            and (
                (p.bias is None and bias is None)
                or (p.bias is not None and bias is not None and _same_view(p.bias, bias[i * rows : (i + 1) * rows]))
            )
            for i, p in enumerate(projections)
        ):
            return cached
    if projections[0].weight.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("MiniCPM NPU QKV weights must be packed during warmup, before graph capture")
    with torch.no_grad():
        weight = torch.cat([p.weight for p in projections])
        bias = None if projections[0].bias is None else torch.cat([p.bias for p in projections])
        for i, p in enumerate(projections):
            p.weight.data = weight[i * rows : (i + 1) * rows]
            if bias is not None:
                p.bias.data = bias[i * rows : (i + 1) * rows]
    attn.__dict__["_npu_packed_qkv"] = weight, bias
    return weight, bias


def attention(attn, x, att_cache, attn_mask, kv, *, min_cache_frames: int):
    """One QKV GEMM, preserving contiguous attention for short histories."""
    batch, frames, _ = x.shape
    weight, bias = packed_qkv(attn)
    q, k, v = F.linear(x, weight, bias).chunk(3, dim=-1)
    q, k, v = attn.q_norm(attn.to_heads(q)), attn.k_norm(attn.to_heads(k)), attn.to_heads(v)
    cached = 0 if att_cache is None else int(att_cache.shape[2])
    dest = kv[:, :, : frames + cached]
    if cached >= min_cache_frames:
        behind = dest[:, :, frames:]
        if att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride():
            behind.copy_(att_cache)
        dim = k.shape[-1]
        dest[:, :, :frames, :dim].copy_(k)
        dest[:, :, :frames, dim:].copy_(v)
        k, v = dest.chunk(2, dim=-1)
    else:
        if att_cache is not None:
            old_k, old_v = att_cache.chunk(2, dim=-1)
            k, v = torch.cat((k, old_k), dim=2), torch.cat((v, old_v), dim=2)
        dest.copy_(torch.cat((k, v), dim=-1))
    mask = _expand_attn_mask_for_npu(attn_mask, frames, frames + cached)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn.proj(out.transpose(1, 2).reshape(batch, frames, -1))


def packed_conv_taps(conv: nn.Conv1d) -> torch.Tensor:
    """Pack taps without a version counter, retaining weight loads as views.

    Like packed_qkv, in-place loads update the pack and replacing/moving a
    Parameter rebuilds it. Inference tensors have no version counter. Graphs
    must be recaptured after weight reloads, as with the rest of the body.
    """
    shape = (conv.kernel_size[0], conv.out_channels, conv.in_channels)
    packed = conv.__dict__.get("_npu_packed_conv_taps")
    if packed is not None and _same_view(conv.weight, packed.view(shape).permute(1, 2, 0)):
        return packed
    if conv.weight.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("MiniCPM NPU convolution weights must be packed during warmup, before graph capture")
    with torch.no_grad():
        packed = conv.weight.detach().permute(2, 0, 1).reshape(-1, conv.in_channels).contiguous()
        conv.weight.data = packed.view(shape).permute(1, 2, 0)
    conv.__dict__["_npu_packed_conv_taps"] = packed
    return packed


def gated_residual_norm(x, gate, branch, norm):
    """Return both the residual stream and its normalization with ACLNN AddLayerNorm."""
    import torch_npu

    op = getattr(torch_npu, "npu_add_layer_norm", None)
    if op is None:
        residual = torch.addcmul(x, gate, branch)
        return residual, norm(residual)
    weight, bias = norm.weight, norm.bias
    if weight is None or bias is None:
        cached = norm.__dict__.get("_npu_norm_affine")
        if cached is None or cached[0].device != x.device or cached[0].dtype != x.dtype:
            cached = (x.new_ones(x.shape[-1]), x.new_zeros(x.shape[-1]))
            norm.__dict__["_npu_norm_affine"] = cached
        weight = cached[0] if weight is None else weight
        bias = cached[1] if bias is None else bias
    normalized, _, _, residual = op(x, gate * branch, weight, bias, epsilon=norm.eps, additional_output=True)
    return residual, normalized


def _ones_zeros(like: torch.Tensor, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _ones_zeros.__dict__.get("_cache")
    if cached is None:
        cached = {}
        _ones_zeros.__dict__["_cache"] = cached
    key = (like.device, like.dtype, channels)
    affine = cached.get(key)
    if affine is None:
        affine = (like.new_ones(channels), like.new_zeros(channels))
        cached[key] = affine
    return affine


def _tap_sum(y: torch.Tensor, taps: int, frames: int, channels: int) -> torch.Tensor:
    """Sum causal conv taps. Contiguous activations reduce in one kernel."""
    if taps == 1:
        return y[:, :frames, :channels]
    if (
        y.is_contiguous()
        and y.stride(-1) == 1
        and int(y.shape[1]) >= frames + taps - 1
        and int(y.shape[2]) >= taps * channels
    ):
        # y[n, t + k, k * C + c] has tap stride stride(time) + C.
        shifted = y.as_strided(
            (int(y.shape[0]), frames, taps, channels),
            (int(y.stride(0)), int(y.stride(1)), int(y.stride(1)) + channels, 1),
        )
        return shifted.sum(dim=2)
    value = y[:, :frames, :channels]
    for k in range(1, taps):
        value = value + y[:, k : k + frames, k * channels : (k + 1) * channels]
    return value


def _apply_gate_bias(branch: torch.Tensor, gate: torch.Tensor | None, y_bias: torch.Tensor | None) -> torch.Tensor:
    """``gate * (branch + bias)`` with one elementwise kernel on the activation."""
    if gate is not None and y_bias is not None:
        return torch.addcmul(gate * y_bias, branch, gate)
    if y_bias is not None:
        return branch + y_bias
    if gate is not None:
        return gate * branch
    return branch


def residual_layer_norm_npu(
    residual: torch.Tensor | None,
    y: torch.Tensor | None = None,
    *,
    gate: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    taps: int = 1,
    y_bias: torch.Tensor | None = None,
    activation: str | None = None,
    residual_out: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``residual_layer_norm`` with ACLNN AddLayerNorm for residual + gate * y.

    Tap reduction, conv bias and Mish stay outside the op. Anything the op
    cannot express falls back to the shared implementation.
    """
    reference = residual if residual is not None else y
    if (
        reference is not None
        and reference.device.type == "npu"
        and reference.dtype == torch.float32
        and not torch.is_grad_enabled()
        and (residual is None or y is None)
        and activation in (None, "mish")
    ):
        # ACLNN LayerNorm accepts affine parameters directly. The shared
        # native fallback launches separate scale, shift and output copies.
        if y is None:
            value = residual
        else:
            frames = int(y.shape[1]) - (taps - 1)
            channels = int(y.shape[2]) // taps
            value = _apply_gate_bias(_tap_sum(y, taps, frames, channels), gate, y_bias)
        if residual_out is not None:
            residual_out.copy_(value)
        result = F.layer_norm(value, (int(value.shape[-1]),), weight, bias, eps)
        if activation == "mish":
            result = F.mish(result)
        if out is not None:
            out.copy_(result)
            return out
        return result
    fused = None
    if (
        activation is None
        and residual is not None
        and y is not None
        and reference is not None
        and reference.device.type == "npu"
        and reference.dtype == torch.float32
        and not torch.is_grad_enabled()
    ):
        try:
            import torch_npu
        except ImportError:
            torch_npu = None
        op = getattr(torch_npu, "npu_add_layer_norm", None) if torch_npu is not None else None
        if op is not None:
            frames, channels = int(residual.shape[1]), int(residual.shape[2])
            branch = _apply_gate_bias(_tap_sum(y, taps, frames, channels), gate, y_bias)
            gamma, beta = weight, bias
            if gamma is None or beta is None:
                ones, zeros = _ones_zeros(residual, channels)
                gamma = ones if gamma is None else gamma
                beta = zeros if beta is None else beta
            normalized, _, _, added = op(
                residual[:, :frames], branch, gamma, beta, epsilon=float(eps), additional_output=True
            )
            if residual_out is not None:
                residual_out[:, :frames].copy_(added)
            fused = normalized
    if fused is None:
        return residual_layer_norm(
            residual,
            y,
            gate=gate,
            weight=weight,
            bias=bias,
            eps=eps,
            taps=taps,
            y_bias=y_bias,
            activation=activation,
            residual_out=residual_out,
            out=out,
        )
    if out is None:
        return fused
    out.copy_(fused)
    return out


def attention_npu(attn, hidden, att_cache, attn_mask, kv, slots=None):
    """Packed-QKV attention using the shared ``[K | V]`` cache layout."""
    if slots is not None:
        raise RuntimeError("MiniCPM NPU fused body does not support the CFM attention slot pool")
    batch, frames, _ = hidden.shape
    if supports_packed_qkv(attn):
        weight, bias = packed_qkv(attn)
        qkv = F.linear(hidden.reshape(batch * frames, -1), weight, bias).view(batch, frames, -1)
    else:
        qkv = torch.cat([attn.to_q(hidden), attn.to_k(hidden), attn.to_v(hidden)], dim=-1)
    cached = 0 if att_cache is None else int(att_cache.shape[2])
    cache_capacity = int(kv.shape[2]) - frames
    logical_cache = cached if attn_mask is None else int(attn_mask.shape[-1]) - frames
    if logical_cache < cached or logical_cache > cache_capacity:
        raise ValueError(
            f"NPU fused attention cache/mask exceeds output capacity: "
            f"cache={cached}, mask_cache={logical_cache}, capacity={cache_capacity}"
        )
    # Whole-Euler reserves query padding between the live queries and cache.
    # Keep this capacity in the attention sequence so the capture mask remains
    # aligned with the static cache buffer on every replay.
    kv = kv[:, :, : frames + logical_cache]
    behind = kv[:, :, frames:]
    if cached and (att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride()):
        behind[:, :, :cached].copy_(att_cache)
    packed = qkv.view(batch, frames, 3, int(attn.num_heads), int(attn.head_dim))
    query = attn.q_norm(packed[:, :, 0].transpose(1, 2))
    current_keys = attn.k_norm(packed[:, :, 1].transpose(1, 2))
    current_values = packed[:, :, 2].transpose(1, 2)
    head_dim = int(query.shape[-1])
    # The shared cache format stores K in the first half and V in the second.
    # A masked gap past the real cache must stay finite: an uninitialized key
    # poisons real rows through ``0 * NaN`` in the attention epilogue.
    if logical_cache > cached:
        behind[:, :, cached:].zero_()
    current = kv[:, :, :frames]
    if current.is_contiguous():
        torch.cat((current_keys, current_values), dim=-1, out=current)
    else:
        current[..., :head_dim].copy_(current_keys)
        current[..., head_dim:].copy_(current_values)
    keys, values = kv[..., :head_dim], kv[..., head_dim:]
    mask = _expand_attn_mask_for_npu(attn_mask, frames, int(keys.shape[-2]))
    if mask is not None and int(mask.shape[-1]) != int(keys.shape[-2]):
        raise ValueError(
            f"NPU fused attention mask/cache mismatch: mask has {mask.shape[-1]} keys, cache has {keys.shape[-2]}"
        )
    out = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
    out = out.transpose(1, 2).reshape(batch * frames, -1)
    return F.linear(out, attn.proj.weight, attn.proj.bias).view(batch, frames, -1)


def blocks_forward_chunk_fused_npu(
    estimator: nn.Module,
    estimator_input: torch.Tensor,
    time_embedding: torch.Tensor,
    attn_mask: torch.Tensor | None,
    cnn_cache,
    att_cache,
    cnn_cache_buffer: torch.Tensor,
    att_cache_buffer: torch.Tensor,
    valid_lengths,
    modulation: torch.Tensor | None = None,
    slots: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    fuse_cache_update: bool = True,
) -> torch.Tensor:
    """NPU DiT body: the CUDA fused schedule with AddLayerNorm and packed QKV."""
    if slots is not None:
        raise RuntimeError("MiniCPM NPU fused body does not support the CFM attention slot pool")
    if attn_mask is not None and int(attn_mask.shape[-1]) != int(att_cache_buffer.shape[-2]):
        # Whole-Euler's static output cache and key mask must describe the
        # same captured sequence; otherwise keep the ragged eager path.
        raise RuntimeError("NPU fused body mask does not match the captured cache layout")
    return blocks_forward_chunk_fused(
        estimator,
        estimator_input,
        time_embedding,
        attn_mask,
        cnn_cache,
        att_cache,
        cnn_cache_buffer,
        att_cache_buffer,
        valid_lengths,
        modulation=modulation,
        slots=None,
        norm=residual_layer_norm_npu,
        attend=attention_npu,
        conv_weight=packed_conv_taps,
        defer_conv_history=True,
        write_cache=write_cache if fuse_cache_update else None,
    )
