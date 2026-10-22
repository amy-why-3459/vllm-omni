# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Recompute preemption on a stage that cannot replay its decode state.

Upstream's only preemption mode frees KV and replays the prompt. A native-AR
stage whose decode step feeds more than the sampled token into the model --
Qwen3-Omni's talker sums all eight RVQ code embeddings and adds the thinker's
text step, but keeps only layer 0 in the token stream -- cannot rebuild that
state. Under ``recompute_preemption="fail"`` the request is finished with one
terminal error instead of being requeued, so the caller never receives audio
that silently restarts at frame 0 (vllm-project/vllm-omni#6179).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)  # isort: skip
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubUpstream:
    """Stands in for ``Scheduler._preempt_request``'s requeue path."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request, timestamp, drop_stale_output=False):
        self.calls += 1
        # What upstream does: drop the KV, zero the frontier, requeue.
        request.num_computed_tokens = 0
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1


def _scheduler(mocker, mode: str):
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(recompute_preemption=mode))
    sched.recompute_preemption_fails = mode == "fail"
    sched._pending_input_timeout_outputs = {}
    upstream = _StubUpstream()
    mocker.patch.object(type(sched).__mro__[2], "_preempt_request", staticmethod(upstream))
    return sched, upstream


def _request() -> Request:
    request = Request(
        request_id="req-preempt",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 7
    request.num_preemptions = 0
    return request


def test_allow_mode_requeues_the_request(mocker):
    sched, upstream = _scheduler(mocker, "allow")
    request = _request()

    sched._preempt_request(request, 1.0)

    assert upstream.calls == 1
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0


def test_fail_mode_finishes_with_one_terminal_error(mocker):
    sched, upstream = _scheduler(mocker, "fail")
    request = _request()

    finished: list[tuple[list[str], RequestStatus]] = []
    mocker.patch.object(
        sched,
        "finish_requests",
        side_effect=lambda ids, status: finished.append((list(ids), status)),
    )

    sched._preempt_request(request, 1.0)

    # Never requeued: no replay, no restart from frame 0.
    assert upstream.calls == 0
    assert finished == [([request.request_id], RequestStatus.FINISHED_ERROR)]

    # Exactly one terminal output, carrying the truncation reason.
    (output,) = sched._pending_input_timeout_outputs.values()
    client_index, engine_output = output
    assert engine_output.request_id == request.request_id
    assert engine_output.new_token_ids == []
    assert "cannot replay" in engine_output.stop_reason
    assert "truncated" in engine_output.stop_reason


def test_fail_mode_is_idempotent_for_an_already_finished_request(mocker):
    sched, upstream = _scheduler(mocker, "fail")
    request = _request()
    request.status = RequestStatus.FINISHED_ABORTED
    finish = mocker.patch.object(sched, "finish_requests")

    sched._preempt_request(request, 1.0)

    assert upstream.calls == 0
    finish.assert_not_called()
    assert sched._pending_input_timeout_outputs == {}


def test_prefix_cache_reset_still_resumes_under_fail_mode(mocker):
    """reset_prefix_cache's own reset intends to resume; it must not fail requests."""
    sched, upstream = _scheduler(mocker, "fail")
    request = _request()
    finish = mocker.patch.object(sched, "finish_requests")

    sched._preempt_request(request, 1.0, drop_stale_output=True)

    assert upstream.calls == 1
    finish.assert_not_called()
    assert sched._pending_input_timeout_outputs == {}
