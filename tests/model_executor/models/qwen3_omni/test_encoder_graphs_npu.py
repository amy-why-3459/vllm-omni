# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU encoder replay parity, mutable boundaries, and cached-output lifetime."""

import socket

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"})]


@pytest.fixture(scope="module")
def npu_state():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    torch.npu.set_device(0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    parallel_state.init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="hccl"
    )
    with set_current_vllm_config(VllmConfig()), pytest.MonkeyPatch.context() as patch:
        patch.setitem(op_registry_oot, "MMEncoderAttention", AscendMMEncoderAttention)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    torch.npu.synchronize()
    parallel_state.destroy_model_parallel()
    parallel_state.destroy_distributed_environment()


def _initialize(module):
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "norm" in name or "ln_post" in name:
                param.fill_(1 if name.endswith("weight") else 0)
            elif param.dim() > 1:
                param.copy_(torch.randn(param.shape, generator=generator) / param[0].numel() ** 0.5)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.02)


def _capture(manager):
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm_ascend.worker.model_runner_v1 import graph_capture

    with torch.inference_mode(), graph_capture(device=torch.device("npu:0")):
        set_cudagraph_capturing_enabled(True)
        try:
            manager.capture(torch.npu.graph_pool_handle())
        finally:
            set_cudagraph_capturing_enabled(False)


def test_audio_and_vision_replay(npu_state):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from tests.model_executor.models.qwen3_omni.test_vision_encoder_cudagraph_cuda import _engine_config, _vision_config
    from vllm_omni.model_executor.models.qwen3_omni.audio_encoder_cudagraph import audio_chunk_metadata
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3Omni_VisionTransformer,
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )
    from vllm_omni.platforms.npu.models.qwen3_omni_encoder_graphs import (
        Qwen3OmniAudioEncoderNpuGraphs,
        SingleReplayEncoderNpuGraphManager,
    )

    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=128,
        encoder_attention_heads=2,
        encoder_layers=2,
        encoder_ffn_dim=256,
        downsample_hidden_size=8,
        output_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device("npu"):
        tower = Qwen3OmniMoeAudioEncoder(config)
        visual = Qwen3Omni_VisionTransformer(vision_config=_vision_config())
    _initialize(tower)
    _initialize(visual)
    model = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.visual = visual
    model.vllm_config = _engine_config()
    model.multimodal_config = model.vllm_config.model_config.multimodal_config
    model.visual_dim, model.multiscale_dim = 64, 128
    model._enable_image_encoder_cudagraph()
    assert model.supports_encoder_cudagraph
    audio = Qwen3OmniAudioEncoderNpuGraphs(tower, budgets=(1, 2, 3, 4))
    model._audio_encoder_graphs = audio
    from types import SimpleNamespace

    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    runner = SimpleNamespace(
        get_model=lambda: model,
        supports_mm_inputs=True,
        compilation_config=model.vllm_config.compilation_config,
        vllm_config=model.vllm_config,
        device=torch.device("npu"),
        dtype=torch.bfloat16,
    )
    manager = OmniGPUModelRunner._create_encoder_cudagraph_manager(runner)
    assert isinstance(manager, SingleReplayEncoderNpuGraphManager)
    retained = []
    try:
        _capture(manager)
        with torch.inference_mode():
            # Same captured tensor shapes, different clip boundaries. Interleave
            # vision replay to catch audio/vision FIA registry collisions.
            for lengths, grids in [
                ([100, 100], [[1, 4, 4]]),
                ([200], [[1, 2, 8], [1, 4, 4]]),
                ([101], [[2, 4, 4]]),
                ([201], [[1, 16, 16]]),
            ]:
                features = torch.randn(80, sum(lengths), dtype=torch.bfloat16, device="npu")
                lens = torch.tensor(lengths, device="npu")
                output_lens = audio_chunk_metadata(lengths, 100, 800)[-1]
                expected = tower(features, lens, torch.tensor(output_lens, device="npu")).split(output_lens)
                if lengths == [100, 100]:
                    merged = tower(features, torch.tensor([200], device="npu"), torch.tensor([26], device="npu"))
                    assert not torch.allclose(torch.cat(expected), merged, rtol=0.016, atol=0.002)
                actual = audio.execute(features, lengths)
                assert actual is not None
                for got, ref in zip(actual, expected):
                    torch.testing.assert_close(got, ref, rtol=0.016, atol=0.002)
                    retained.append((got, got.clone()))
                modality = "video" if grids[0][0] > 1 else "image"
                pixels = torch.randn(sum(t * h * w for t, h, w in grids), 384, device="npu", dtype=torch.bfloat16)
                inputs = {
                    f"{modality}_grid_thw": torch.tensor(grids),
                    "pixel_values_videos" if modality == "video" else "pixel_values": pixels,
                }
                ref = model.encoder_eager_forward(inputs)
                parts = manager.execute(inputs)
                assert parts is not None
                got = torch.cat(parts)
                torch.testing.assert_close(got, ref, rtol=0.016, atol=0.002)
                retained.extend((part, part.clone()) for part in parts)
            assert audio.execute(torch.zeros(80, 99, device="npu", dtype=torch.bfloat16), [99]) is None
            for got, saved in retained:
                torch.testing.assert_close(got, saved, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        manager.clear()
    assert not audio.graphs

    # Audio-only deployments use the same capture/clear lifecycle.
    from vllm_omni.worker.encoder_cudagraph import AudioOnlyEncoderCudaGraphManager

    audio_only = AudioOnlyEncoderCudaGraphManager(model)
    try:
        _capture(audio_only)
        with torch.inference_mode():
            actual = audio.execute(features, lengths)
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, rtol=0.016, atol=0.002)
    finally:
        torch.npu.synchronize()
        audio_only.clear()
    assert not audio.graphs


@pytest.mark.parametrize(
    "enabled,explicit,expected,profiled", [(True, None, 6, True), (False, None, 10, False), (True, 10, 10, False)]
)
def test_npu_encoder_memory_reservation(monkeypatch, enabled, explicit, expected, profiled):
    from types import SimpleNamespace
    from unittest.mock import Mock

    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.worker.base import NPUWorker, OmniNPUWorkerBase

    monkeypatch.setattr(NPUWorker, "determine_available_memory", lambda self: 10 << 30)
    profile = Mock(return_value=4 << 30)
    worker = object.__new__(OmniNPUWorkerBase)
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mm_encoder=enabled))
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=explicit)
    worker.model_runner = SimpleNamespace(
        profile_encoder_cudagraph_memory=profile,
        get_model=lambda: SimpleNamespace(encoder_cudagraph_single_replay=True),
    )
    assert worker.determine_available_memory() == expected << 30
    assert profile.called == profiled
