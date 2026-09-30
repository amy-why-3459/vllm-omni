# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, exact-shape CUDA graphs for stateless MiniCPM input encoders."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor | None, ...]
    output: torch.Tensor


class EncoderCudaGraph:
    """Capture repeated shapes without padding or changing attention semantics.

    The owner must bypass this object for training, stateful KV caches and
    data-dependent Python branches. Each entry owns a private graph pool;
    callers get a clone so a later replay cannot overwrite retained embeddings.
    Unseen shapes after the cap run eagerly instead of growing GPU memory.
    """

    def __init__(self, forward: Callable[..., torch.Tensor], *, max_graphs: int = 4):
        self.forward = forward
        self.max_graphs = max_graphs
        self.graphs: dict[tuple, _Graph] = {}
        self._seen: dict[tuple, None] = {}

    def __call__(self, *inputs: torch.Tensor | None) -> torch.Tensor:
        tensors = [value for value in inputs if value is not None]
        if (
            not tensors
            or any(value.device.type != "cuda" for value in tensors)
            or len({value.device for value in tensors}) != 1
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.forward(*inputs)
        # Do not share mutable replay buffers across independent CUDA streams.
        key = (
            torch.cuda.current_stream(tensors[0].device).cuda_stream,
            tuple(None if x is None else (x.shape, x.dtype, x.device) for x in inputs),
        )
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= self.max_graphs:
                return self.forward(*inputs)
            if key not in self._seen:
                # Avoid paying capture latency for one-off shapes. Bound even
                # the admission metadata for streams with arbitrary resolutions.
                if len(self._seen) >= self.max_graphs * 4:
                    self._seen.pop(next(iter(self._seen)))
                self._seen[key] = None
                return self.forward(*inputs)
            entry = self._capture(inputs)
            self.graphs[key] = entry
            self._seen.pop(key)
        for source, target in zip(inputs, entry.inputs, strict=True):
            if source is not None:
                assert target is not None
                target.copy_(source)
        entry.graph.replay()
        return entry.output.clone()

    def _capture(self, inputs: tuple[torch.Tensor | None, ...]) -> _Graph:
        device = next(x.device for x in inputs if x is not None)
        static = tuple(None if x is None else x.clone() for x in inputs)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward(*static)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.forward(*static)
        torch.cuda.current_stream(device).wait_stream(stream)
        return _Graph(graph, static, output)
