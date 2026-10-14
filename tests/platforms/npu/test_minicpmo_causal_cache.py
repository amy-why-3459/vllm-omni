# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ragged cache writes with changing graph inputs and non-contiguous storage."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import prepare_code2wav_graph_runtime
from vllm_omni.platforms.npu.models.minicpmo_causal_cache import write_cache

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.mark.parametrize("rows,frames,channels,width", [(2, 2, 512, 2), (8, 50, 512, 2), (4, 1, 17, 3), (2, 3, 16, 0)])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_history_cache_graph(rows, frames, channels, width, strided):
    prepare_code2wav_graph_runtime()
    histories = [torch.randn(rows, channels, frames + width, device="npu").transpose(1, 2) for _ in range(2)]
    if not strided:
        histories = [h.contiguous() for h in histories]
    lengths = torch.zeros(rows * 2, dtype=torch.long, device="npu")[::2]
    storage = torch.full((rows, 2 * channels + 3, width), 123.0, device="npu")
    out = storage[:, : 2 * channels]
    for _ in range(3):
        write_cache(*histories, lengths, out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write_cache(*histories, lengths, out)
    for step in range(3):
        for h in histories:
            h.normal_()
        lens = (torch.arange(rows) + step) % (frames + 1)
        lengths.copy_(lens)
        graph.replay()
        torch.npu.synchronize()
        expected = []
        for history in histories:
            cpu = history.cpu()
            expected.append(torch.stack([cpu[r, int(lens[r]) : int(lens[r]) + width].T for r in range(rows)]))
        torch.testing.assert_close(out.cpu(), torch.cat(expected, dim=1), rtol=0, atol=0)
        assert (storage[:, 2 * channels :].cpu() == 123).all()
