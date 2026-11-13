# PR 10184: matched-runtime validation

Engine revision: `5e579b70dec7c65f3a8560a539f48a75445e83b7`.
Four allocated Ascend 910B1 devices; concurrency 1; original six-step workloads.
The JSON report contains every configuration, source/harness hash, raw timing sample,
request RGB/PNG hash, numerical image check and process exit result.

All three configurations explicitly set `npu_disable_internal_format=true` and
`npu_disable_jit_compile=true`. OMP/MKL threads are 4,
`HCCL_DETERMINISTIC=true`, and `TASK_QUEUE_ENABLE=1`.

Use the environment and dependencies recorded in [results.json](results.json).
Check out the engine revision above and select four allocated NPU devices.
Set `MODE` to `control`, `graph4`, then `batched`, starting and stopping a fresh
server for each mode. Supply the matching `configurations` entry from the JSON
report as `ADDITIONAL_CONFIG`:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HCCL_DETERMINISTIC=true TASK_QUEUE_ENABLE=1 \
vllm serve "${MODEL_PATH}" --omni \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --num-gpus 4 --tensor-parallel-size 1 --usp 4 --text-encoder-tp-size 4 \
  --host 127.0.0.1 --port 18124 \
  --cache-config '{"minimax_h3_adaln_offload":true}' \
  --additional-config "${ADDITIONAL_CONFIG}"
```

For each healthy server, run the frozen harness from its validator directory:

```bash
python benchmark.py --base-url http://127.0.0.1:18124 --plan plan/workload.json \
  --examples "${MODEL_PATH}/examples" --output "results/${MODE}" --timeout 180
python benchmark.py --base-url http://127.0.0.1:18124 --plan plan/workload.json \
  --examples "${MODEL_PATH}/examples" --output "results/${MODE}-quality" --timeout 180 --quality
```

[The frozen workload and validator scripts](https://github.com/InferMatrixLab/InferRSI/tree/e4773063f1c16fdbd5897b772be12262fe5035d5/evidence/h3-image-ascend/validator)
are the same as the existing campaign. Quality scoring ran after all servers stopped.
Each mode completed 12 author-workload requests (one warmup and three timed requests
per workload) and six heldout-seed requests. Timing statistics exclude warmups and
heldout requests. The graph4 arm changes only single-tile graph/stream execution;
the batched arm also changes encoder precision and tile batching.

| Workload | Graph-off control | Graph4 | Graph4 reduction | Batched | Batched reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 17.2961 s | 11.5666 s | 33.1% | 11.1270 s | 35.7% |
| W2 | 4.4602 s | 3.0831 s | 30.9% | 2.7563 s | 38.2% |
| W3 | 6.1854 s | 4.9972 s | 19.2% | 4.7389 s | 23.4% |

After all servers stop, compare each candidate's `requests.json` and images
against the corresponding control directory:

```bash
for mode in graph4 batched; do
  for suffix in '' -quality; do
    python validate_approximate_images_cached.py \
      --reference "results/control${suffix}" \
      --candidate "results/${mode}${suffix}" \
      --output "results/${mode}${suffix}-screen.json"
  done
done
```

The graph4 outputs matched all 18 corresponding control RGB images exactly.
All 18 batched images passed the frozen numerical screen, with nonzero differences.
The batched minimum PSNR was 37.5745 dB and minimum SSIM was 0.990270.
All three servers and all four workers per server exited with code 0.
Raw images and host logs remain local. No supplemental prompt coverage, full visual
acceptance or final ABBA qualification is claimed. All original latency targets
remain unmet; campaign verdict is **INCONCLUSIVE**.

AI-assisted implementation, measurement orchestration and evidence preparation: Codex.
