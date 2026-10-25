# Qwen3-Omni MRv2 performance

This model adaptation builds on #8184's existing stage runtime and output
contracts. It adds Talker CPU metadata parsing, sampled-token embedding
handoff, an optional first-frame decoder, length-grouped Code2Wav graph
buckets, and opt-in incremental short-KV code prediction. The H200 profile
also enables conservative fused sampling and bounded cuDNN algorithm search.
The shared predictor keeps the upstream re-prefill helper and its default
sampling behavior; short KV is explicitly selected by the deployment profile.

Use `vllm_omni/deploy/qwen3_omni_moe_mrv2_h200.yaml` for the experimental H200
profile. Preserve a control with `code_predictor_fused_sampling=false` and
`codec_cudnn_benchmark=false` when C64 P99 is the primary requirement.
cuDNN search restores process settings on exit and has startup cost. Kernel
arithmetic is not bitwise-equivalent; the first-frame decoder uses additional
model memory. `vllm_omni/deploy/qwen3_omni_duplex_mrv2.yaml` runs these MRv2
stages for duplex. Qwen duplex submits each committed utterance as a new
request, so it uses the turn-mode data plane. Turn-mode measurements do not
establish duplex latency or interruption behavior.

First-frame delivery requires `talker_first_audio: true` in connector extra,
a one-frame initial codec chunk, CUDA, a local single-rank Talker and prefix caching disabled.
Other deployments retain the regular Code2Wav path.
`codec_fused_snake` explicitly opts CUDA profiles into fused decoder blocks;
the base V1 profile retains its original blocks and indexed text cache.

## Opt-in codec convolution autotuning

`vllm_omni/deploy/qwen3_omni_moe_mrv2_cudnn.yaml` exposes the existing
`codec_cudnn_benchmark` capability from
[#8223](https://github.com/vllm-project/vllm-omni/pull/8223). It inherits the
MRv2 profile, keeps MPS disabled, and enables only this additional codec flag:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe_mrv2_cudnn.yaml
```

The inherited deployment places the Thinker on GPU 0 and the Talker and
Code2Wav on GPU 1. Its existing first-audio, chunk-ramp and fused-Snake settings
remain part of the shared MRv2 baseline. For an isolated control, use the same
configuration with `codec_cudnn_benchmark: false`; keep MPS disabled in both.

On CUDA, cuDNN searches convolution algorithms during eager warmup before
graph capture, with `benchmark_limit=10`; process settings are restored after
the decode. Search adds warmup work, and selected algorithms can change
floating-point rounding. This profile does not promise bitwise audio parity.
A single paired pilot on two L20X GPUs at revision
`a83eb6ed38fde8c6964a0315420f52e69706d31c` found a throughput/latency tradeoff:
at concurrency 64, PCM throughput increased 7.79% and mean RTF decreased 6.06%,
but TTFA p50/p95 increased 7.12%/2.02%. Concurrency 1 showed no material
throughput benefit, and concurrency 2 regressed. The separate same-device
256-item English Seed-TTS quality pair passed its frozen mean-WER criterion
(0.05030 to 0.03797); this single pair does not establish a causal quality gain.
The evidence and limitations are tracked in
[InferRSI #112](https://github.com/InferMatrixLab/InferRSI/issues/112).
This is an opt-in throughput experiment, not an all-metric improvement or
qualification against the campaign's endpoint RTF targets.
