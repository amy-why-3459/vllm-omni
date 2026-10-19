# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``bsgemm-moe``: block-FP8 routed experts; every other layer stays bf16."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform

QUANTIZATION_METHOD = "bsgemm-moe"


def _resolve_experts_on_cpu(self, layer: torch.nn.Module, weight_name: str, scale_name: str) -> torch.Tensor:
    """Expand this method's expert tensor on the host, then return it on the device.

    Upstream expands in place on the device, so the FP8 expert stack and its
    bfloat16 expansion are resident together; on a 64 GB 910B that peak aborts
    the load. Moving both parameters to the host first releases the FP8 device
    storage before the expansion allocates, leaving only the expanded tensor on
    the device. The expansion itself stays with the upstream implementation,
    which keeps its block-scale shape validation and its bounded FP32 staging.
    """
    from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockFusedMoEMethod

    weight_param = getattr(layer, weight_name)
    scale_param = getattr(layer, scale_name)
    device = weight_param.device
    weight_param.data = weight_param.data.to("cpu")
    scale_param.data = scale_param.data.to("cpu")
    if device.type == "npu":
        torch.npu.empty_cache()
    resolved = AscendFp8BlockFusedMoEMethod._resolve_experts(self, layer, weight_name, scale_name)
    return resolved.to(device=device)


def _dequantize_npu_block_fp8_on_cpu(method: QuantizeMethodBase | None) -> None:
    """Scope the CPU expansion workaround to this Drama 3 quantization method.

    Ascend's adapter owns the block-FP8 scheme in ``quant_method``. Override
    only that instance; other models and the Atlas 950 MXFP8 path retain
    their upstream implementation.
    """
    from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockFusedMoEMethod

    scheme = getattr(method, "quant_method", None)
    if isinstance(scheme, AscendFp8BlockFusedMoEMethod):
        scheme._resolve_experts = MethodType(_resolve_experts_on_cpu, scheme)


@register_quantization_config(QUANTIZATION_METHOD)
class ExpertFp8Config(Fp8Config):
    """vLLM's block-FP8 MoE path for routed experts, unquantized linears."""

    def __init__(self, weight_block_size: list[int]) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=weight_block_size,
        )

    @classmethod
    def get_name(cls) -> str:
        return QUANTIZATION_METHOD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExpertFp8Config:
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError(f"{QUANTIZATION_METHOD} needs dynamic activation scales")
        block = config.get("weight_block_size")
        if not (isinstance(block, list) and len(block) == 2 and all(int(b) > 0 for b in block)):
            raise ValueError(f"{QUANTIZATION_METHOD} needs a 2-int weight_block_size, got {block!r}")
        return cls(weight_block_size=[int(b) for b in block])

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str, tid2eid: torch.Tensor | None = None
    ) -> QuantizeMethodBase | None:
        if current_platform.device_type == "npu":
            return self._npu_quant_method(layer, prefix, tid2eid)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)

    def _npu_quant_method(
        self, layer: torch.nn.Module, prefix: str, tid2eid: torch.Tensor | None
    ) -> QuantizeMethodBase | None:
        """Ascend's own methods: block-FP8 routed experts, bf16 Ascend linears.

        CUDA's ``Fp8MoEMethod`` selects no Ascend kernel (``Fp8MoeBackend.NONE``),
        and the generic ``UnquantizedLinearMethod`` would skip the FRACTAL_NZ
        weight conversion and the Ascend GEMM that ``AscendLinearBase`` installs
        for an unquantized checkpoint, so both sides need the vLLM-Ascend
        implementation.
        """
        # Delayed imports: vllm_ascend exists only on an Ascend host.
        if isinstance(layer, LinearBase):
            from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

            return AscendUnquantizedLinearMethod()
        from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config

        npu_config = AscendFp8Config(
            is_checkpoint_fp8_serialized=self.is_checkpoint_fp8_serialized,
            activation_scheme=self.activation_scheme,
            weight_block_size=self.weight_block_size,
        )
        method = npu_config.get_quant_method(layer, prefix, tid2eid=tid2eid)
        _dequantize_npu_block_fp8_on_cpu(method)
        return method
