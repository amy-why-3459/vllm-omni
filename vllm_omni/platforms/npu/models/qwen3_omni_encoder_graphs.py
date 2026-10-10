# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni audio NPUGraphs with per-replay FIA sequence boundaries."""

import torch
import torch_npu
from vllm_ascend.worker.encoder_acl_graph import (
    EncoderAclGraphManager,
    get_encoder_graph_params,
    set_encoder_forward_context,
    set_encoder_graph_params,
    update_encoder_graph_params,
    weak_ref_workspaces,
)

from vllm_omni.model_executor.models.qwen3_omni.audio_encoder_cudagraph import Qwen3OmniAudioEncoderCudaGraphs
from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager


class SingleReplayEncoderNpuGraphManager(SingleReplayEncoderCudaGraphManager, EncoderAclGraphManager):
    def clear(self):
        super().clear()
        set_encoder_graph_params([])


class Qwen3OmniAudioEncoderNpuGraphs(Qwen3OmniAudioEncoderCudaGraphs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Vision uses positive token budgets. Audio needs separate entries for
        # every (CNN chunks, output tokens) pair, including equal token counts.
        self._budget_keys = {shape: -i for i, shape in enumerate(self.capture_shapes, start=1)}
        self.update_stream = None

    def capture(self, pool):
        # Conv2D must use ACLNN: the internal-format ACL-op path cannot be
        # captured. Restore the process setting after startup capture.
        allow_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        try:
            torch.npu.config.allow_internal_format = False
            super().capture(pool)
        finally:
            torch.npu.config.allow_internal_format = allow_internal_format != b"disable"

    def _capture_graph(self, forward, pool, key):
        if get_encoder_graph_params() is None:
            set_encoder_graph_params([])
        params = get_encoder_graph_params()
        budget = self._budget_keys[key]
        params.events[budget] = []
        params.handles[budget] = []
        params.attn_params[budget] = []
        params.workspaces[budget] = None
        graph = torch.npu.NPUGraph()
        with set_encoder_forward_context(budget, True), torch.npu.graph(graph, pool=pool):
            output = forward()
        weak_ref_workspaces()
        return graph, output

    def _replay_graph(self, captured, key, boundaries):
        if self.update_stream is None:
            self.update_stream = torch.npu.Stream()
        # Enqueue this dependency before replay, which waits for FIA updates.
        # Waiting on the replay itself here would create a stream cycle.
        self.update_stream.wait_stream(torch.npu.current_stream())
        captured.graph.replay()
        with set_encoder_forward_context(
            self._budget_keys[key], False, cu_seqlens_cpu=torch.tensor(boundaries, dtype=torch.int32)
        ):
            update_encoder_graph_params(self.update_stream, self._budget_keys[key])

    def clear(self):
        super().clear()
        params = get_encoder_graph_params()
        if params is not None:
            for budget in self._budget_keys.values():
                for mapping in (params.events, params.handles, params.attn_params, params.workspaces):
                    mapping.pop(budget, None)
        self.update_stream = None
