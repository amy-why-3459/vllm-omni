# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.chunk_encoder_graph import ChunkEncoderGraph

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_cpu_fallback_and_capacity():
    def forward(x, **kwargs):
        return x.sin(), x + 1, x + 2

    wrapper = ChunkEncoderGraph(forward)
    x = torch.randn(4, requires_grad=True)
    wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)[0].sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    assert not wrapper.graphs
    with pytest.raises(ValueError):
        ChunkEncoderGraph(forward, max_graphs=-1)
    with pytest.raises(ValueError, match="capture_after"):
        ChunkEncoderGraph(forward, capture_after=1)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_configurable_admission_avoids_short_lived_shape_capture():
    wrapper = ChunkEncoderGraph(lambda x, **kw: (x + 1, x + 2, x + 3), capture_after=4)
    x = torch.ones(8, device="cuda")
    for _ in range(3):
        torch.testing.assert_close(wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)[0], x + 1)
    assert not wrapper.graphs
    assert wrapper.stats["admission"] == 3
    for value in (2, 3):
        x.fill_(value)
        result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
        torch.testing.assert_close(result[0], x + 1)
    assert wrapper.stats["captures"] == 1
    assert wrapper.stats["hits"] == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("amp", [False, True])
@torch.inference_mode()
def test_real_conformer_chunks_replay_and_preserve_state(amp):
    from cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav, _undecorate_dynamo

    encoder = (
        UpsampleConformerEncoderV2(
            input_size=32,
            output_size=32,
            num_blocks=1,
            num_up_blocks=1,
            attention_heads=4,
            linear_units=64,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        .eval()
        .cuda()
    )
    _undecorate_dynamo(encoder, "forward_chunk")
    backend = BatchedToken2Wav.__new__(BatchedToken2Wav)
    torch.nn.Module.__init__(backend)
    backend.flow = torch.nn.Module()
    backend.flow.input_embedding = torch.nn.Embedding(64, 32).cuda()
    backend.flow.encoder = encoder
    backend.flow.encoder_proj = torch.nn.Linear(32, 16).cuda()
    backend.flow.eval()
    graph = ChunkEncoderGraph(backend._encode_chunk_eager, max_graphs=2)
    backend._chunk_encoder_graph = graph
    cnn = att = None
    retained = []
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for last in [False, False, True]:
            tokens = torch.randint(0, 64, (2, 8), device="cuda")
            # Repeated input shape with distinct values must update graph inputs.
            for _ in range(3):
                tokens = (tokens + 1) % 64
                expected = backend._encode_chunk_eager(tokens, last_chunk=last, cnn_cache=cnn, att_cache=att)
                actual = backend._encode_chunk(tokens, last_chunk=last, cnn_cache=cnn, att_cache=att)
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, b, atol=3e-3 if amp else 1e-5, rtol=3e-3 if amp else 1e-5)
                retained.append((actual, tuple(x.clone() for x in actual)))
            if not last:
                _, cnn, att = actual
        # A replacement positional table must select a new graph entry.
        before = graph.stats["captures"]
        encoder.embed.pos_enc.pe = encoder.embed.pos_enc.pe.clone()
        for _ in range(2):
            backend._encode_chunk(tokens, last_chunk=True, cnn_cache=cnn, att_cache=att)
        assert graph.stats["captures"] == before + 1
    assert graph.stats["hits"] >= 6
    for actual, expected in retained:
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stream_isolation_cache_refresh_and_capacity():
    def forward(x, *, last_chunk, cnn_cache, att_cache):
        return x + cnn_cache + att_cache + int(last_chunk), cnn_cache + x, att_cache - x

    wrapper = ChunkEncoderGraph(forward, max_graphs=2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]
    retained = []
    for stream in streams:
        with torch.cuda.stream(stream):
            x = torch.ones(3, device="cuda")
            cnn = torch.zeros_like(x)
            att = torch.zeros_like(x)
            for i in range(3):
                cnn.fill_(i)
                att.fill_(2 * i)
                out = wrapper(x, last_chunk=False, cnn_cache=cnn, att_cache=att)
                retained.append((out[0], torch.full_like(x, 1 + 3 * i)))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert len(wrapper.graphs) == 2
    assert wrapper.stats["hits"] == 6
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_full_cache_replaces_lru_and_preserves_old_outputs():
    def forward(x, **kwargs):
        return x.sin(), x + 1, x + 2

    wrapper = ChunkEncoderGraph(forward, max_graphs=1)
    old_stream = torch.cuda.Stream()
    with torch.cuda.stream(old_stream):
        x = torch.ones(1, 32, device="cuda")
        for _ in range(3):
            old = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    # A repeated new batch replaces the old graph, retaining request outputs.
    x4 = torch.full((4, 32), 2.0, device="cuda")
    for _ in range(6):
        result = wrapper(x4, last_chunk=False, cnn_cache=None, att_cache=None)
    assert wrapper.stats["evictions"] == 1
    assert wrapper.stats["captures"] == 2
    assert wrapper.stats["hits"] == 7
    assert len(wrapper.graphs) == 1
    for actual, expected in zip(result, forward(x4), strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(old[0], torch.ones_like(old[0]).sin())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_hot_graph_survives_shape_churn_with_bounded_metadata():
    wrapper = ChunkEncoderGraph(lambda x, **kw: (x + 1, x + 2, x + 3), max_graphs=1)
    hot = torch.ones(1, device="cuda")
    for _ in range(2):
        wrapper(hot, last_chunk=False, cnn_cache=None, att_cache=None)
    for size in range(2, 30):
        wrapper(torch.ones(size, device="cuda"), last_chunk=False, cnn_cache=None, att_cache=None)
        wrapper(hot, last_chunk=False, cnn_cache=None, att_cache=None)
    assert wrapper.stats["captures"] == 1
    assert wrapper.stats["evictions"] == 0
    assert len(wrapper.seen) <= 8


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_overlapping_replays_use_independent_scratch_pools():
    def forward(x, **kwargs):
        intermediate = x.sin() + x.cos()
        for _ in range(8):
            intermediate = intermediate.sin() + 0.1 * x
        return intermediate, intermediate * 2, intermediate * 3

    wrapper = ChunkEncoderGraph(forward, max_graphs=2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    # Complete captures before testing concurrent replay (warmup synchronizes).
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            x = torch.full((512, 512), float(i), device="cuda")
            for _ in range(2):
                wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    torch.accelerator.synchronize()
    assert wrapper._slots[0][1] != wrapper._slots[1][1]
    retained = []
    for iteration in range(6):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                x = torch.full((512, 512), float(iteration + i), device="cuda")
                result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
                retained.append((result, x))
    torch.accelerator.synchronize()
    for result, x in retained:
        for a, b in zip(result, forward(x), strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_capture_failure_requires_restart(monkeypatch):
    wrapper = ChunkEncoderGraph(lambda x, **kw: (x, x, x))
    x = torch.ones(3, device="cuda")
    wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)

    def fail(*args):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(wrapper, "_capture", fail)
    with pytest.raises(RuntimeError, match="simulated capture failure"):
        wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    with pytest.raises(RuntimeError, match="restart the stage"):
        wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)


@pytest.mark.cpu
@torch.inference_mode()
def test_failed_capture_blocks_even_eager_retry(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=1))
    wrapper = ChunkEncoderGraph(lambda x, **kw: (x, x, x))
    x = SimpleNamespace(device=torch.device("cuda:0"), shape=(3,), dtype=torch.float32)
    wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)

    def fail(*args):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(wrapper, "_capture", fail)
    with pytest.raises(RuntimeError, match="simulated capture failure"):
        wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    with pytest.raises(RuntimeError, match="restart the stage"):
        wrapper(torch.ones(3), last_chunk=False, cnn_cache=None, att_cache=None)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_lru_refresh_and_repeated_gemm_eviction():
    weight = torch.randn(32, 32, device="cuda")

    def forward(x, **kwargs):
        y = (x @ weight).sin()
        return y, y + 1, y + 2

    wrapper = ChunkEncoderGraph(forward, max_graphs=2)

    def call(size, repeats=1):
        x = torch.randn(size, 32, device="cuda")
        for _ in range(repeats):
            result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
        for a, b in zip(result, forward(x), strict=True):
            torch.testing.assert_close(a, b)
        return next(reversed(wrapper.graphs))

    key_a = call(1, 2)
    key_b = call(2, 2)
    call(1)  # A becomes MRU, despite having been captured first.
    call(3, 2)
    assert key_a in wrapper.graphs
    assert key_b not in wrapper.graphs
    for size in range(4, 36):
        call(size, 2)
        assert len(wrapper.graphs) == len(wrapper._slots) == 2
    assert wrapper.stats["evictions"] == 33


@pytest.mark.cpu
def test_lru_reset_precedes_slot_reuse(monkeypatch):
    from unittest.mock import Mock

    wrapper = ChunkEncoderGraph(None, max_graphs=2)
    events = []
    graph_a, graph_b = Mock(), Mock()
    graph_a.reset.side_effect = lambda: events.append("reset-a")
    graph_b.reset.side_effect = lambda: events.append("reset-b")
    wrapper.graphs.update(a=(graph_a,), b=(graph_b,))
    wrapper._entry_slots = {"a": 0, "b": 1}
    wrapper._slots = [(torch.device("cuda:0"),), (torch.device("cuda:1"),)]
    wrapper.graphs.move_to_end("a")
    wrapper._entry_streams = {"a": Mock(), "b": Mock()}
    wrapper._entry_streams["b"].synchronize.side_effect = lambda: events.append("sync-b")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda d: pytest.fail("device-wide sync"))
    assert wrapper._evict_lru() == 1
    assert list(wrapper.graphs) == ["a"]
    assert events == ["sync-b", "reset-b"]
    graph_a.reset.assert_not_called()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_eviction_does_not_drain_unrelated_stream():
    import time

    wrapper = ChunkEncoderGraph(lambda x, **kw: (x.sin(), x + 1, x + 2), max_graphs=1)
    owner = torch.cuda.Stream()
    unrelated = torch.cuda.Stream()
    with torch.cuda.stream(owner):
        x = torch.ones(32, device="cuda")
        for _ in range(2):
            retained = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    torch.accelerator.synchronize()
    # Measure the old device-wide waiting policy under the same queued load.
    with torch.cuda.stream(unrelated):
        torch.cuda._sleep(200_000_000)
    start = time.perf_counter()
    torch.accelerator.synchronize()
    old_wait_ms = (time.perf_counter() - start) * 1000
    with torch.cuda.stream(unrelated):
        torch.cuda._sleep(200_000_000)
        done = torch.cuda.Event()
        done.record()
    start = time.perf_counter()
    wrapper._evict_lru()
    elapsed = (time.perf_counter() - start) * 1000
    assert not done.query(), "eviction waited for unrelated GPU work"
    done.synchronize()
    torch.testing.assert_close(retained[0], x.sin())
    print(f"eviction under unrelated GPU load: old device wait={old_wait_ms:.3f} ms, new eviction={elapsed:.3f} ms")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_capture_streams_are_not_reused_by_torch_pool():
    # Retaining torch.cuda.Stream objects does not reserve their underlying
    # streams. Resetting a graph clears all cuBLAS workspaces on its capture
    # stream, so a pooled capture stream can invalidate a still-live peer.
    weight = torch.randn(512, 512, device="cuda")
    wrapper = ChunkEncoderGraph(lambda x, **kw: (x @ weight, x + 1, x + 2), max_graphs=2)
    for size in (28, 32):
        x = torch.randn(size, 512, device="cuda")
        for _ in range(2):
            wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    streams = {slot[5].cuda_stream for slot in wrapper._slots}
    assert len(streams) == 2
    pooled = {torch.cuda.Stream().cuda_stream for _ in range(128)}
    assert streams.isdisjoint(pooled)
    # Eviction and recapture must reuse the slot's exclusive stream, not
    # allocate new pooled streams as the number of observed shapes grows.
    for size in (36, 40, 44):
        x = torch.randn(size, 512, device="cuda")
        for _ in range(2):
            result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
        torch.testing.assert_close(result[0], x @ weight)
    assert {slot[5].cuda_stream for slot in wrapper._slots} == streams
    assert wrapper.stats["evictions"] == 3
