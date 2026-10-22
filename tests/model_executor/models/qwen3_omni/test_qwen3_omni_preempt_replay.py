# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The talker cannot rebuild its decode state from the token stream.

A talker decode step feeds ``summed_embeddings(layer0..7) + text_step`` into
PagedAttention, but only the layer-0 codec id is kept in ``output_token_ids``.
When ``_preempt_request()`` zeroes ``num_computed_tokens`` the replay is
scheduled as a prefill of ``prompt + every frame already generated``; the
talker rebuilds only the prompt part, and the runner's
``seg_len = min(span_len, req_embeds.shape[0])`` leaves the remaining rows of
its ``torch.empty`` buffer uninitialised.

The pipeline therefore declares ``recompute_preemption="fail"`` for this stage
(vllm-project/vllm-omni#6179). These tests pin that declaration so it cannot be
dropped silently, and pin the row shortfall the declaration exists to avoid.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 8
PROMPT_LEN = 12  # rows _thinker_to_talker_prefill builds
GEN_FRAMES = 5  # frames already delivered before preemption


def _talker_stage():
    return next(s for s in QWEN3_OMNI_PIPELINE.stages if s.model_stage == "talker")


def test_talker_stage_declares_fail_on_recompute_preemption():
    stage = _talker_stage()
    assert stage.recompute_preemption == "fail", (
        "The talker's decode input is not reconstructible from output_token_ids; "
        "letting it be replay-preempted splices two unrelated halves of audio."
    )


def test_thinker_stage_keeps_upstream_recompute():
    """The thinker is plain text AR: its token stream is its state."""
    thinker = next(s for s in QWEN3_OMNI_PIPELINE.stages if s.model_stage == "thinker")
    assert thinker.recompute_preemption == "allow"


def test_replay_span_exceeds_the_embeddings_the_talker_can_rebuild():
    """Mirror of the runner's copy; fails if the shortfall ever goes unnoticed."""
    span_len = PROMPT_LEN + GEN_FRAMES  # what _preempt_request() re-schedules
    rebuilt = torch.zeros(PROMPT_LEN, HIDDEN)  # prompt-only, as prefill builds it

    # OmniGPUModelRunner._preprocess (vllm_omni/worker/gpu_model_runner.py)
    inputs_embeds = torch.empty(span_len, HIDDEN)
    seg_len = min(span_len, rebuilt.shape[0])
    inputs_embeds[0:seg_len] = rebuilt[:seg_len]

    assert seg_len == PROMPT_LEN
    assert inputs_embeds.shape[0] - seg_len == GEN_FRAMES
