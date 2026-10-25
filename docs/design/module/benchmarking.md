---
title: Benchmarking
kind: module
status: draft
owners:
  - "@alex-jw-brooks"
  - "@Bounty-hunter"
primary_code_paths:
  - vllm_omni/benchmarks/**
  - vllm_omni/entrypoints/cli/benchmark/**
related_code_paths:
  - benchmarks/**
depends_on:
  - entrypoints.md
  - profiling.md
validation_paths:
  - tests/benchmarks/**
  - tests/dfx/perf/**
upstream_refs:
  - vllm.benchmarks
last_reviewed: 2026-07-16
---

# Benchmarking

Benchmarking defines reproducible workloads, metrics, warm-up, measurement,
and result metadata for comparing vLLM-Omni performance.

## Candidate invariants

### BENCH-INV-001: Results are reproducible

**Rule:** Results MUST identify the model, revision, commit, hardware,
configuration, workload, concurrency, warm-up, and measurement window.

### BENCH-INV-002: Correctness precedes performance

**Rule:** A benchmark MUST validate successful and semantically acceptable
outputs before using timings as performance evidence.

### BENCH-INV-003: Metric definitions are explicit

**Rule:** Metrics MUST state their unit, population, and aggregation.

## Safe-change guide

Test workload parsing and metric calculations with deterministic fixtures. Do
not compare materially different workloads or runtime settings.

## Current streaming-audio result fields

The `openai-chat-omni` benchmark starts audio TTFA only when a nonempty audio
chunk is accepted into its audio buffer. Empty or invalid base64 chunks,
unparsable or empty WAV chunks, and WAV chunks with inconsistent audio
parameters do not start TTFA or extend the audio generation interval. RTF
uses the interval from request start through the last accepted audio chunk
divided by decoded audio duration. Text timing is independent of audio timing.

When audio metrics are selected or any audio measurement is observed, results
include `audio_request_metrics`. Use `--save-result --save-detailed` to retain
these per-request rows in the benchmark JSON, including failed and audio-less
replies in original request order:

| Field | Meaning |
| --- | --- |
| `request_index`, `request_id` | Input position and request identifier. |
| `success`, `error` | Request outcome; failures remain in the result. |
| `start_time` | Request start from `perf_counter`, in seconds; not Unix wall time. |
| `latency` | End-to-end request latency, in seconds. |
| `audio_ttfp` | Time from request start to first accepted audio chunk, in seconds. |
| `audio_duration` | Decoded audio duration, in seconds. |
| `audio_frames` | Decoded PCM frame count, not codec-token count. |
| `audio_rtf` | Dimensionless audio generation interval divided by audio duration. |

Missing, nonfinite or nonpositive measurements are serialized as `null`, not
zero-valued observations. Downstream analysis must retain failure and missing
audio counts instead of treating missing measurements as fast requests. These
are client benchmark measurements, distinct from server Prometheus metrics.
