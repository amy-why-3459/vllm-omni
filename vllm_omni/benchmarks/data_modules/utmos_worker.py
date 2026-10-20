# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""UTMOS scorer process.

``balacoon/utmos`` traces ``torch.device("cuda:0")`` into the graph. Launch this
file with ``CUDA_VISIBLE_DEVICES`` already set to one physical GPU so that card
is ``cuda:0`` before torch initializes. Do not import torch at module scope.
"""

from __future__ import annotations

import struct
import sys


def _emit_ready() -> None:
    sys.stdout.buffer.write(b"R")
    sys.stdout.buffer.flush()


def _emit_error(message: str) -> None:
    encoded = message.encode("utf-8", errors="replace")
    sys.stdout.buffer.write(b"E" + struct.pack("<I", len(encoded)) + encoded)
    sys.stdout.buffer.flush()


def _emit_score(value: float | None) -> None:
    if value is None:
        sys.stdout.buffer.write(b"Z")
    else:
        sys.stdout.buffer.write(b"S" + struct.pack("<d", value))
    sys.stdout.buffer.flush()


def _read_exact(count: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < count:
        chunk = sys.stdin.buffer.read(count - len(buffer))
        if not chunk:
            raise EOFError
        buffer.extend(chunk)
    return bytes(buffer)


def main() -> None:
    """Load the JIT on ``cuda:0`` and score waveforms from stdin."""
    jit_path = sys.argv[1]
    try:
        import math

        import numpy as np
        import torch

        model = torch.jit.load(jit_path, map_location="cuda:0")
        model.eval()
        warmup = torch.zeros(1, 16000, dtype=torch.float32, device="cuda:0")
        with torch.no_grad():
            model(warmup)
        del warmup
    except Exception as exc:
        _emit_error(f"{type(exc).__name__}: {exc}")
        return
    _emit_ready()

    while True:
        try:
            (count,) = struct.unpack("<I", _read_exact(4))
        except EOFError:
            return
        if count == 0xFFFFFFFF:
            return
        try:
            raw = _read_exact(count * 4)
            wav = np.frombuffer(raw, dtype=np.float32).copy()
            audio = torch.from_numpy(wav).unsqueeze(0).to("cuda:0")
            with torch.no_grad():
                out = model(audio)
            value = float(out.reshape(-1)[0].item())
            _emit_score(value if math.isfinite(value) else None)
        except Exception as exc:
            _emit_error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
