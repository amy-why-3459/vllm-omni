# PR 10184: default optimization validation

Engine revision: `31364f79890ea559b4580ef4bc46eeabacf7af53`.
The NPU H3 image pipeline now defaults to graph-backed batch32 decoding,
FP16 batch8 encoding, scoped deterministic execution and eligible AdaLN host
staging. This changes the default numerical path. Explicit graph-off, FP32,
single-tile and resident-weight overrides remain available.

This check compares omitted optimization arguments with an explicit configuration
on the same revision, hardware and workloads. It is not a new speedup claim.
See [validation.json](validation.json) for per-request image hashes, descriptive
timings, configuration, software, source/harness hashes and exit codes.

## Reproduction

Use a matching vLLM/torch-npu environment and the source revision above. Select
four allocated Ascend 910B1 devices and source the installed CANN environment.
With `MODEL_PATH` pointing to the H3 image checkpoint, start the default server:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HCCL_DETERMINISTIC=true TASK_QUEUE_ENABLE=1 \
vllm serve "${MODEL_PATH}" --omni \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --num-gpus 4 --tensor-parallel-size 1 --usp 4 --text-encoder-tp-size 4 \
  --host 127.0.0.1 --port 18125
```

After running the workloads, stop that server and start a fresh explicit server
with the same command plus:

```bash
--cache-config '{"minimax_h3_adaln_offload":true}' \
--additional-config '{"text_encoder_cudnn_sdp":false,"npu_deterministic":true,"npu_disable_internal_format":true,"npu_disable_jit_compile":true,"npu_vae_tile_graph":true,"npu_vae_tile_batch":32,"npu_vae_encoder_precision":"fp16","npu_vae_encoder_batch":8}'
```

For each server, set `MODE` to `defaults` or `explicit` and run the
[frozen campaign harness](https://github.com/InferMatrixLab/InferRSI/tree/e4773063f1c16fdbd5897b772be12262fe5035d5/evidence/h3-image-ascend/validator):

```bash
python benchmark.py --base-url http://127.0.0.1:18125 --plan plan/workload.json \
  --examples "${MODEL_PATH}/examples" --output "results/${MODE}" --timeout 180
python benchmark.py --base-url http://127.0.0.1:18125 --plan plan/workload.json \
  --examples "${MODEL_PATH}/examples" --output "results/${MODE}-quality" --timeout 180 --quality
```

Each server handles 12 author-workload requests (one warmup plus three measured
requests for each W1/W2/W3) and six extra-seed requests (43/44). Compare the decoded
RGB hashes for matching workload/seed/index keys, including all 18 requests.
Check server and worker exit codes after shutdown. Raw images and host logs stay
local; this does not complete supplemental-prompt or final visual/ABBA acceptance.

## CPU regression commands

Prerequisites: the project's matching Python, PyTorch, vLLM and test dependencies;
no accelerator or model weights are required for these CPU tests. From the repo root:

```bash
python -m pytest tests/platforms/npu/test_minimax_h3_image_graphs.py -m 'core_model and cpu' -q
python -m pytest tests/platforms/npu/test_minimax_h3_image_graphs.py tests/platforms/npu/test_minimax_h3_encoder_batch.py tests/diffusion/models/minimax_h3/test_adaln_cache.py -m 'core_model and cpu' -q
```

The selected suite passed 119 tests. Two cases were rerun after fixture cleanup
and passed. Applicable format/repository hooks and the selected mypy wrapper
passed; the wrapper excludes the model-port tree by repository policy.
The existing Buildkite CPU shards already collect these test modules.

AI-assisted implementation, validation orchestration and documentation: Codex.
