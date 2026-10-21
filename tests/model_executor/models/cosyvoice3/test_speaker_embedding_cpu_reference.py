# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve the CPU speaker graph's pooling semantics before TRT conversion."""

import os

import numpy as np
import pytest

from vllm_omni.model_executor.models.cosyvoice3 import speaker_embedding_trt
from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import _cpu_reference_onnx

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_model(path, scale=1.0):
    h = onnx.helper
    graph = h.make_graph(
        [
            h.make_node("Pad", ["x", "pads"], ["padded"], mode="constant"),
            h.make_node("AveragePool", ["padded"], ["pooled"], kernel_shape=[100], strides=[100], ceil_mode=1),
            h.make_node("Mul", ["pooled", "scale"], ["y"]),
        ],
        "speaker_pool",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, "T"])],
        [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, "O"])],
        [
            h.make_tensor("pads", onnx.TensorProto.INT64, [6], [0] * 6),
            h.make_tensor("scale", onnx.TensorProto.FLOAT, [1], [scale]),
        ],
    )
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10), path)


def _run(path, features, optimization):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = optimization
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": features})[0]


@pytest.mark.parametrize("frames", [97, 100, 200, 598])
def test_export_retains_reference_ceil_pool_semantics(tmp_path, monkeypatch, frames):
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "speaker.onnx"
    _write_model(source)
    features = np.arange(frames, dtype=np.float32).reshape(1, 1, frames)
    reference = _run(source, features, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    exported = _cpu_reference_onnx(str(source))
    # No optimizer here: another backend must receive the effective graph,
    # including the reference runtime's treatment of a partial final window.
    actual = _run(exported, features, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    np.testing.assert_array_equal(actual, reference)


def test_cache_identity_includes_model_content(tmp_path, monkeypatch):
    monkeypatch.setenv("COSYVOICE3_TRT_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "speaker.onnx"
    _write_model(source)
    before = source.stat()
    first = _cpu_reference_onnx(str(source))
    assert _cpu_reference_onnx(str(source)) == first
    _write_model(source, scale=2.0)
    assert source.stat().st_size == before.st_size
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = _cpu_reference_onnx(str(source))
    assert second != first
    features = np.ones((1, 1, 97), dtype=np.float32)
    np.testing.assert_array_equal(
        _run(second, features, ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        _run(source, features, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    )


def test_reference_mode_does_not_reuse_the_default_engine(monkeypatch):
    monkeypatch.setattr(speaker_embedding_trt, "_CAMPPLUS_CACHE", {})
    monkeypatch.setattr(speaker_embedding_trt, "_cpu_reference_onnx", lambda path: path + ".cpu.onnx")
    monkeypatch.setattr(
        speaker_embedding_trt,
        "CampplusTRT",
        lambda path, device, *, allow_tf32: {"path": path, "device": device, "allow_tf32": allow_tf32},
    )
    original = speaker_embedding_trt.get_campplus_trt("speaker.onnx", "cuda:0")
    reference = speaker_embedding_trt.get_campplus_trt("speaker.onnx", "cuda:0", cpu_reference=True)
    assert reference is speaker_embedding_trt.get_campplus_trt("speaker.onnx", "cuda:0", cpu_reference=True)
    assert original is not reference
    assert original["allow_tf32"] is True
    assert reference["allow_tf32"] is False
    assert original["path"] != reference["path"]
