MiniCPM-o 4.5 native duplex: incomplete final response and post-commit timeout

Artifacts from the 2026-10-08 19:00-19:04 Asia/Shanghai run.
input.wav: synthetic Mandarin questions, mono PCM16 16 kHz, 180 seconds.
duplex_sliding_window_180s.mp4: exact audiovisual fixture used, including audio.
output.wav: concatenated model output (not a wall-clock playback recording).
timeline.json: question timings and expected scene answers.
generate.py: original fixture generator (Pillow, numpy, ffmpeg, espeak-ng).
realtime_duplex_demo.py: client snapshot; no sliding-window override.
deploy.yaml: current deployment configuration snapshot.
events.sanitized.jsonl: original events with embedded audio/reference data omitted.
server-excerpt.log: startup/version information and matching-session timing entries.
result.json: original client result.

Run against a compatible vLLM-Omni installation:
MODEL=/path/to/MiniCPM-o-4_5 REF_AUDIO=/path/to/reference.wav bash reproduce.sh
The original reference was MiniCPM-o-5_0/assets/HT_ref_audio.wav; it is not bundled.
Do not substitute audio-only input: this reproducer depends on synchronized video.

Server command (adapt checkpoint/path):
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniCPM-o-4_5 --omni --port 28889 --trust-remote-code --deploy-config ./deploy.yaml --allowed-local-media-path / --interleave-mm-strings --media-io-kwargs '{"video":{"fps":1,"num_frames":128}}' --init-timeout 1800 --stage-init-timeout 1800 --log-stats

Recorded package versions: vllm 0.31.0; vllm-omni 0.1.dev38+g7f7e747fe.
Current environment: torch 2.13.0+cu130; transformers 5.14.1; NVIDIA L20X; driver 570.133.20.
Current source HEAD at report preparation: b843a28e36ae89a951a348fc0504302de07d24eb.
Editable-source revision loaded by the already-running server was not independently captured.
No new reproduction run was performed while preparing these artifacts.
