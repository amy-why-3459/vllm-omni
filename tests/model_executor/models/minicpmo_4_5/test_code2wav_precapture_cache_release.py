# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test for ``_release_precapture_cache`` on Ascend NPU.

``cfm_precapture_empty_cache`` is enabled in the shipped deploy config, but the
release path used to gate on ``torch.cuda.is_available()`` and call
``torch.accelerator.empty_cache()``. On an NPU host the gate is False, so the
whole block was dead; routing it through the CUDA branch instead crashes with
"Allocator for npu is not a DeviceAllocator". These tests pin the NPU branch to
``torch.npu.empty_cache`` so the flag keeps meaning what it says.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5 import (
    minicpmo_4_5_code2wav as code2wav_module,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage() -> MiniCPMO45Code2Wav:
    stage = MiniCPMO45Code2Wav.__new__(MiniCPMO45Code2Wav)
    return stage


def _fake_npu(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, current_device: int = 0) -> None:
    npu = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        current_device=lambda: current_device,
        memory_allocated=lambda device=None: 1024,
        memory_reserved=lambda device=None: 2048,
    )
    monkeypatch.setattr(code2wav_module.torch, "npu", npu, raising=False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)


def test_release_precapture_cache_uses_npu_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """On NPU the release must call torch.npu.empty_cache, not the CUDA path."""
    calls: list[str] = []
    _fake_npu(monkeypatch, calls)

    def _forbidden(*args, **kwargs):
        raise AssertionError("torch.accelerator.empty_cache crashes on NPU and must not be reached")

    monkeypatch.setattr(code2wav_module.torch.accelerator, "empty_cache", _forbidden)
    monkeypatch.setattr(code2wav_module.torch.accelerator, "synchronize", _forbidden)

    _stage()._release_precapture_cache()

    assert calls == ["synchronize", "empty_cache"]


def test_release_precapture_cache_is_reachable_on_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NPU branch must not be gated on torch.cuda, which is always unavailable there."""
    calls: list[str] = []
    _fake_npu(monkeypatch, calls)
    # A host that never initialized CUDA: the old guard skipped the whole body.
    monkeypatch.setattr(code2wav_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(code2wav_module.torch.cuda, "is_initialized", lambda: False)

    _stage()._release_precapture_cache()

    assert "empty_cache" in calls
