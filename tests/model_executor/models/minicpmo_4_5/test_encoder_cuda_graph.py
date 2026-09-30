# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cuda_graph import EncoderCudaGraph
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
)

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_cpu_and_grad_paths_remain_eager():
    graph = EncoderCudaGraph(lambda x, mask: x.sin() if mask is None else x.sin() + mask)
    x = torch.randn(2, 3, requires_grad=True)
    graph(x, None).sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    assert not graph.graphs
    assert not graph._seen


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_refreshes_inputs_preserves_outputs_and_bounds_shapes():
    graph = EncoderCudaGraph(lambda x, mask: x.sin() if mask is None else x.sin() + mask, max_graphs=2)
    x = torch.randn(2, 8, device="cuda")
    mask = torch.randn_like(x)
    graph(x, mask)
    old = graph(x, mask)
    expected_old = old.clone()
    x.mul_(2)
    mask.add_(3)
    torch.testing.assert_close(graph(x, mask), x.sin() + mask)
    torch.testing.assert_close(old, expected_old)
    graph(x, None)
    torch.testing.assert_close(graph(x, None), x.sin())
    assert len(graph.graphs) == 2
    for size in range(3, 24):
        other = torch.randn(size, 8, device="cuda")
        torch.testing.assert_close(graph(other, None), other.sin())
    assert len(graph.graphs) == 2
    assert len(graph._seen) <= 8


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_vision_graph_handles_changed_mask_and_retained_embeddings():
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    )
    config._attn_implementation = "eager"
    model = SiglipVisionTransformer(config).eval().cuda()
    pixels = torch.randn(2, 3, 14, 56, device="cuda")
    sizes = torch.tensor([[2, 2], [1, 2]], dtype=torch.int32)
    mask = torch.tensor([[[1, 1, 1, 1]], [[1, 1, 0, 0]]], device="cuda", dtype=torch.bool)
    reference = model(pixels, mask, sizes).last_hidden_state
    model._encoder_graph = EncoderCudaGraph(model._encode_last_hidden_state)
    model(pixels, mask, sizes)
    output = model(pixels, mask, sizes).last_hidden_state
    torch.testing.assert_close(output, reference)
    assert len(model._encoder_graph.graphs) == 1
    # Same padded shape, different valid positions and image content.
    pixels.add_(1)
    sizes[1] = torch.tensor([1, 3])
    mask[1, 0, 2] = True
    actual = model(pixels, mask, sizes).last_hidden_state
    graph = model._encoder_graph
    model._encoder_graph = None
    expected = model(pixels, mask, sizes).last_hidden_state
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(output, reference)
    model._encoder_graph = graph
    detailed = model(pixels, mask, sizes, output_hidden_states=True)
    assert len(detailed.hidden_states) == 3
    assert len(graph.graphs) == 1


def _audio_model(device):
    from types import SimpleNamespace

    from transformers import WhisperConfig

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMO45OmniLLMForConditionalGeneration,
        MiniCPMWhisperEncoder,
        MultiModalProjector,
    )

    config = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
    )
    config._attn_implementation = "sdpa"
    model = MiniCPMO45OmniLLMForConditionalGeneration.__new__(MiniCPMO45OmniLLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(audio_chunk_length=1.0, audio_pool_step=5)
    model.apm = MiniCPMWhisperEncoder(config)
    model.audio_projection_layer = MultiModalProjector(32, 48)
    model.audio_avg_pooler = torch.nn.AvgPool1d(5, stride=5)
    model.audio_encoder_layer = -1
    model.audio_past_key_values = None
    return model.eval().to(device)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_audio_graph_refreshes_mask_and_matches_eager():
    model = _audio_model("cuda")
    data = {
        "audio_features": torch.randn(2, 80, 100, device="cuda"),
        "audio_feature_lens": torch.tensor([[100], [80]], device="cuda"),
    }
    expected = model.get_audio_hidden_states(data)
    graph = EncoderCudaGraph(model._encode_audio_features)
    model._audio_encoder_graph = graph
    model.get_audio_hidden_states(data)
    actual = model.get_audio_hidden_states(data)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    assert len(graph.graphs) == 1
    data["audio_features"].add_(1)
    data["audio_feature_lens"][1, 0] = 30
    changed = model.get_audio_hidden_states(data)
    model._audio_encoder_graph = None
    for a, b in zip(changed, model.get_audio_hidden_states(data), strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.cpu
@torch.inference_mode()
def test_streaming_audio_does_not_replay_stateless_graph():
    model = _audio_model("cpu")

    def forbidden(*args):
        raise AssertionError("streaming KV must not use the stateless graph")

    model._audio_encoder_graph = forbidden
    data = {"audio_features": torch.randn(1, 80, 100), "audio_feature_lens": torch.tensor([[100]])}
    for _ in range(2):
        output = model.get_audio_embedding_streaming(data)
        assert output[0][0].shape == (10, 48)
    assert model.audio_past_key_values.get_seq_length() == 100


@pytest.mark.cpu
@torch.inference_mode()
def test_fp16_audio_keeps_host_overflow_check_eager():
    model = _audio_model("cpu").half()

    def forbidden(*args):
        raise AssertionError("FP16 overflow guard cannot run inside capture")

    model._audio_encoder_graph = forbidden
    data = {
        "audio_features": torch.randn(1, 80, 100, dtype=torch.float16),
        "audio_feature_lens": torch.tensor([[100]]),
    }
    assert model.get_audio_hidden_states(data)[0].shape == (10, 48)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_one_off_shapes_do_not_exhaust_capture_admission():
    graph = EncoderCudaGraph(torch.sin, max_graphs=1)
    for size in range(1, 20):
        graph(torch.zeros(size, device="cuda"))
    assert len(graph._seen) == 4
    assert not graph.graphs
    x = torch.randn(32, device="cuda")
    graph(x)
    torch.testing.assert_close(graph(x), x.sin())
    assert len(graph.graphs) == 1
