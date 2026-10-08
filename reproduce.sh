#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${MODEL:?Set MODEL to your MiniCPM-o-4_5 checkpoint}"
: "${REF_AUDIO:?Set REF_AUDIO to a reference voice WAV}"
python realtime_duplex_demo.py \
  --url "${URL:-ws://localhost:28889/v1/realtime?duplex=1}" \
  --model "$MODEL" --ref-audio "$REF_AUDIO" \
  --input-video ./duplex_sliding_window_180s.mp4 \
  --video-fps 1 --frame-max-side 0 --stack-frames 1 \
  --timeout-s 60 --output-dir "${OUTPUT_DIR:-./repro-output}"
