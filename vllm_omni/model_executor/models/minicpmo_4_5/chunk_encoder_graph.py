# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape graphs for the state-explicit Code2Wav Conformer chunk encoder."""

import weakref
from collections import Counter, OrderedDict

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _new_capture_stream(device):
    # torch.cuda.Stream() draws from a round-robin pool: retaining the Python
    # object does not prevent another graph from capturing on the same stream.
    # PyTorch reset() clears cuBLAS workspaces for the entire capture stream,
    # including workspaces still referenced by other graphs. Own a CUDA stream
    # outside that pool so retiring one slot cannot invalidate a live peer.
    from cuda.bindings import runtime

    from vllm_omni.platforms import current_omni_platform

    previous_device = current_omni_platform.current_device()
    current_omni_platform.set_device(device)
    try:
        error, handle = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot create Code2Wav capture stream: {error}")
        stream = torch.cuda.ExternalStream(int(handle), device=device)
    finally:
        current_omni_platform.set_device(torch.device(device.type, previous_device))
    weakref.finalize(stream, runtime.cudaStreamDestroy, handle)
    return stream


class ChunkEncoderGraph:
    """Capture repeated shapes; caches are inputs/outputs, never hidden state.

    Owners must grow positional tables outside this wrapper and pass the current
    tables so graph entries keep their storage alive and cannot use stale PE.
    Capture errors propagate: continuing after a failed capture is not safe.
    Like IndexTTS2's DiT wrapper, cache hits move to the MRU end and an
    admitted miss evicts the LRU entry at capacity. One-off shapes stay eager.
    Each slot owns a private pool; eviction synchronizes the owning stream and
    resets the graph before reusing its slot. Each slot owns a non-pooled
    capture stream; the sentinel keeps its private memory pool alive.
    """

    def __init__(self, forward, *, max_graphs=8, capture_after=2):
        self.forward = forward
        self.max_graphs = int(max_graphs)
        if self.max_graphs < 0:
            raise ValueError("encoder graph capacity must be >= 0")
        self.capture_after = int(capture_after)
        if self.capture_after < 2:
            raise ValueError("encoder graph capture_after must be >= 2")
        self.graphs = OrderedDict()
        self.seen = OrderedDict()
        # Independent pools remain alive across same-device slot reuse.
        self._slots = []
        self._entry_slots = {}
        self._entry_streams = {}
        self._failed = False
        self.stats = Counter()

    def __call__(self, tokens, *, last_chunk, cnn_cache, att_cache, position_tables=()):
        if self._failed:
            raise RuntimeError("Code2Wav encoder CUDA capture failed; restart the stage before retrying")
        inputs = (tokens, cnn_cache, att_cache)
        self.stats["calls"] += 1

        def eager(reason):
            self.stats["eager"] += 1
            self.stats[reason] += 1
            self._log_stats()
            return self.forward(tokens, last_chunk=last_chunk, cnn_cache=cnn_cache, att_cache=att_cache)

        if (
            not self.max_graphs
            or tokens.device.type != "cuda"
            or torch.is_grad_enabled()
            or any(x is not None and x.device != tokens.device for x in inputs)
            or torch.cuda.is_current_stream_capturing()
        ):
            return eager("ineligible")
        amp = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
        caller_stream = torch.cuda.current_stream(tokens.device)
        key = (
            caller_stream.cuda_stream,
            last_chunk,
            amp,
            dtype,
            tuple(None if x is None else (x.shape, x.dtype, x.device) for x in inputs),
            tuple((id(x), x.data_ptr(), x.shape, x.dtype, x.device) for x in position_tables),
        )
        entry = self.graphs.get(key)
        captured = entry is None
        if entry is None:
            count = self.seen.pop(key, 0) + 1
            self.seen[key] = count
            if len(self.seen) > self.max_graphs * 8:
                self.seen.popitem(last=False)
            if count < self.capture_after:
                return eager("admission")
            try:
                if len(self.graphs) >= self.max_graphs:
                    slot = self._evict_lru()
                else:
                    slot = len(self._slots)
                    self._slots.append(None)
                entry = self._capture(inputs, last_chunk, amp, dtype, position_tables, slot)
            except Exception:
                self._failed = True
                logger.exception("Code2Wav encoder CUDA capture failed; stage restart required")
                raise
            self.graphs[key] = entry
            self._entry_slots[key] = slot
            self._entry_streams[key] = caller_stream
            self.seen.pop(key, None)
            self.stats["captures"] += 1
            logger.info("Code2Wav encoder captured CUDA graph %d/%d", len(self.graphs), self.max_graphs)
        self.graphs.move_to_end(key)
        graph, static_inputs, outputs, _tables = entry
        # Capture already cloned these exact inputs; only cache hits need copies.
        if not captured:
            for target, source in zip(static_inputs, inputs, strict=True):
                if target is not None:
                    target.copy_(source)
        graph.replay()
        self.stats["hits"] += 1
        # Subsequent replay must not overwrite another chunk/request's state.
        result = tuple(x.clone() for x in outputs)
        self._log_stats()
        return result

    def _log_stats(self):
        if self.stats["calls"] % 256 == 0:
            logger.info(
                "Code2Wav encoder graph stats: calls=%d hits=%d (%.1f%%) "
                "eager=%d admission=%d capacity=%d ineligible=%d captures=%d evictions=%d resident=%d",
                self.stats["calls"],
                self.stats["hits"],
                100 * self.stats["hits"] / self.stats["calls"],
                self.stats["eager"],
                self.stats["admission"],
                self.stats["capacity"],
                self.stats["ineligible"],
                self.stats["captures"],
                self.stats["evictions"],
                len(self.graphs),
            )

    def _evict_lru(self):
        # Match the ordered-cache policy used by IndexTTS2, but drain in-flight
        # work before destroying a graph that can replay on another stream.
        key = next(iter(self.graphs))
        slot = self._entry_slots[key]
        # The key includes the caller stream, and returned clones are queued
        # there too. No other stream can replay this entry, so waiting for the
        # whole GPU needlessly stalls unrelated stages.
        self._entry_streams[key].synchronize()
        entry = self.graphs[key]
        entry[0].reset()
        del self.graphs[key]
        del self._entry_slots[key]
        del self._entry_streams[key]
        del entry
        self.stats["evictions"] += 1
        return slot

    def _ensure_pool(self, device, slot):
        existing = self._slots[slot]
        if existing is not None and existing[0] == device:
            return existing[1]
        pool = torch.cuda.graph_pool_handle()
        stream = _new_capture_stream(device)
        static = torch.zeros(1, device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                kept = static + 0
            stream.synchronize()
            sentinel = torch.cuda.CUDAGraph()
            sentinel._capture_stream_owner = stream
            with torch.cuda.graph(sentinel, pool=pool, stream=stream):
                kept = static + 0
        torch.cuda.current_stream(device).wait_stream(stream)
        self._slots[slot] = (device, pool, sentinel, static, kept, stream)
        return pool

    def _capture(self, inputs, last_chunk, amp, dtype, position_tables, slot):
        device = inputs[0].device
        pool = self._ensure_pool(device, slot)
        stream = self._slots[slot][5]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.autocast("cuda", enabled=amp, dtype=dtype, cache_enabled=False):
            static = tuple(None if x is None else x.clone() for x in inputs)

            def compute():
                return self.forward(static[0], last_chunk=last_chunk, cnn_cache=static[1], att_cache=static[2])

            # Prime kernels on the capture stream, with autocast weight caching
            # disabled so temporary casted weights cannot escape capture.
            for _ in range(2):
                compute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph._capture_stream_owner = stream
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                outputs = compute()
        torch.cuda.current_stream(device).wait_stream(stream)
        return graph, static, outputs, tuple(position_tables)
