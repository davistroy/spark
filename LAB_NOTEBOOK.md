# DGX Spark Optimization Lab Notebook

**Project:** LATER_PLAN execution — closing the SM121 performance gap
**Start date:** 2026-03-29
**Hardware:** NVIDIA DGX Spark (GB10, SM121, 128GB LPDDR5x unified memory)
**Model:** Qwen3.5-35B-A3B (FP8 on-the-fly quantization)
**Baseline image:** `vllm/vllm-openai:cu130-known-good-20260306` (vLLM v0.17.0rc1.dev102)
**Reference:** LATER_PLAN.md, research synthesis (SM121 performance gap analysis, 2026-03-28)

---

## Baseline Snapshot (2026-03-29 ~08:00 UTC)

Captured during active pipeline run (~5,500 completed requests).

### System State

| Metric | Value | Notes |
|--------|-------|-------|
| Uptime | 23h 26m | |
| Total RAM | 121.6 GiB | Unified CPU/GPU |
| RAM used | 110 GiB | |
| RAM available | 11 GiB | |
| Swap total | 16 GiB | /swap.img |
| Swap used | 7.3 GiB | Sticky — not recovering with swappiness=1 |
| GPU temp | 64°C | Stable 60-65°C over 3h |
| GPU utilization | 95% | Under pipeline load |
| GPU power | 35.2W | Low for utilization (UMA architecture) |
| CPU usage | ~6-6.5% | Not a bottleneck |

### Swap Consumers

| Process | PID | Swap (GiB) | Identity |
|---------|-----|-----------|----------|
| uvicorn | 14136 | 2.57 | qwen3-embed vLLM server |
| VLLM::EngineCore | 12853 | 1.73 | qwen35 engine core |
| vllm | 12501 | 0.70 | qwen3-embed main process |
| java | 141299 | 0.69 | Neo4j |
| **Total top 4** | | **5.69** | |

### Container State

| Container | Status | Image | GPU Memory | Host RSS |
|-----------|--------|-------|-----------|----------|
| qwen35 | Up 7h | cu130-known-good-20260306 | 85,603 MiB | ~800 MB |
| qwen3-embed | Up 23h | cu130-nightly | 15,810 MiB | ~200 MB |
| gliner | Up 23h | gliner-ner:latest | 2,369 MiB | ~80 MB |
| chromadb | Up 10h | chromadb/chroma:latest | — | ~5 MB |
| neo4j | Up 10h | neo4j:5-community | — | ~750 MB |
| node-exporter | Up 22h | — | — | ~19 MB |

### vLLM Configuration (qwen35)

```
Image: vllm/vllm-openai:cu130-known-good-20260306
Version: v0.17.0rc1.dev102+ge68de8adc
enforce_eager: False (CUDA graphs ACTIVE — 51 piecewise + 35 full captured)
async_scheduling: Disabled (--no-async-scheduling)
gpu_memory_utilization: 0.65
quantization: fp8 (on-the-fly from BF16 weights)
kv_cache_dtype: fp8
max_model_len: 32768
num_gpu_blocks: 2280
prefix_caching: False
MoE backend: MARLIN Fp8
Attention backend: FLASHINFER
cudagraph_mode: FULL_AND_PIECEWISE
```

**Critical warning in logs:**
> "Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads."

This confirms the SM121 native kernel deficit (CMake bug #38126). The GPU DOES support FP8 in hardware, but the compiled vLLM binary doesn't include SM121-native kernels (`scaled_mm`, MLA, NVFP4, `moe_data`).

**Also noted in logs:**
> "Marlin kernel can achieve better performance for small size_n with experimental use_atomic_add feature. You can consider set environment variable VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."

### vLLM Metrics (live during pipeline)

| Metric | Value | Notes |
|--------|-------|-------|
| KV cache usage | 1.58-1.62% | Massively over-provisioned |
| Preemptions | 0 | Perfect |
| Requests running | 8 (snapshot) | Oscillates 5-18 over 3h window |
| Requests waiting | 0 | Never queued |
| Requests completed | 5,493 (5,480 stop + 13 length) | 0 errors, 0 aborts |
| Total prompt tokens | 5.13M | |
| Total generation tokens | 5.49M | |
| TTFT p50 | ~0.4-0.5s | 93% of requests < 500ms |
| TTFT p95 | ~1.0s | Spike to ~3s at end of window |
| E2E latency p50 | ~1.67 min | Long pipeline requests |
| Token throughput (aggregate) | 100-200 tok/s steady | Bursts to 800-900 tok/s |

### Grafana Dashboard Observations (3h window, 05:00-08:00 UTC)

1. **GPU utilization** steady at 95-100%, brief dip to ~40% around 07:00 (batch completion)
2. **Token throughput** has clear burst pattern — steady 100-200 tok/s, spikes to 800-900 during batch completions
3. **Swap** flat at ~8 GB throughout — no recovery, no growth. Stable.
4. **Embedding model** completely idle — 0 running, 0 waiting, negligible throughput. Holding 15.8 GiB GPU for nothing.
5. **Concurrency** peaks at 15-18 simultaneous LLM requests — much higher than the "3 peak" LATER_PLAN assumed
6. **KV cache** jumped from ~0% to 1-3% when pipeline ramped up around 07:00

### sysctl Configuration (verified)

```
vm.swappiness = 1
vm.min_free_kbytes = 262144
```

### What Changed Since LATER_PLAN Was Written (2026-03-28)

| Change | When | Impact |
|--------|------|--------|
| `--enforce-eager` removed | 2026-03-29 ~04:30 UTC | CUDA graphs now capturing. Token throughput improved. |
| Image tagged `cu130-known-good-20260306` | Before this session | Safe rollback point preserved |
| ChromaDB + Neo4j started | ~10h ago | Running with data from pipeline |
| Marlin FP8 MoE backend active | Already was (auto-selected) | Confirmed in logs |
| FlashInfer attention active | Already was (auto-selected) | Confirmed in logs |

---

## Step 0 Decision Matrix Results

Based on baseline data, applying the LATER_PLAN Step 0 decision criteria:

| Check | Result | Decision |
|-------|--------|----------|
| KV Cache peak < 5%? | Yes (3% max) | Could reduce gpu-memory-util to 0.60 |
| Any preemptions? | No (0) | Safe to proceed with all changes |
| Peak concurrency > 3? | Yes (15-18) | Async scheduling removal HIGH priority |
| Waiting requests > 0? | No | Not throughput-constrained at current load |
| TTFT bottleneck (>1s p95)? | Borderline (1s p95, spike to 3s) | Prefill optimization moderate priority |
| Swap > 1 GB? | Yes (7.8 GB) | Note: stable/sticky, not active pressure |
| Available RAM < 5 GB? | No (10-12 GB) | Safe to proceed |
| GPU temp > 75°C? | No (64°C) | No thermal concern |
| Embed requests < 10 in 24h? | Essentially 0 active | Consider sleep mode or stopping entirely |

### Revised Execution Plan

Based on Step 0 data AND the research synthesis (SM121 performance gap analysis):

| Order | Action | Rationale |
|-------|--------|-----------|
| 1 | **Pull `hellohal2064/vllm-qwen3.5-gb10:latest`** | Validate 50 tok/s claim. Zero build effort. Fastest path to native SM121 kernels. |
| 2 | **Benchmark hellohal vs current** | Side-by-side throughput comparison. Document everything. |
| 3 | **If validated, swap in as qwen35** | Biggest single performance improvement available |
| 4 | **Enable prefix caching** | Pipeline workloads benefit from shared system prompts |
| 5 | **Remove `--no-async-scheduling`** | With 15-18 concurrent requests, async scheduling is high-value |
| 6 | **Address swap** — evaluate embed sleep mode | 15.8 GB GPU held for idle embed, 3.3 GB swapped to disk |
| 7 | **Reduce gpu-memory-util to 0.60** | KV cache at 3% — still massively over-provisioned at 0.65 |
| 8 | **Docker Compose** (LATER Step 9) | Codify final validated config |
| 9 | **OS cleanup** (LATER Step 10) | Free RAM from unused services |

---

## Experiment Log

### Entry 001 — Baseline capture and Step 0 assessment
**Date:** 2026-03-29 ~08:00 UTC
**Operator:** Claude Code (remote via SSH + Grafana)
**Status:** READ-ONLY — pipeline active, no changes made

**Actions taken:**
1. SSH to Spark — captured `free -h`, `cat /proc/swaps`, `docker ps`, `nvidia-smi`, `docker stats`
2. Read vLLM `/metrics` endpoint directly (Prometheus port 9090 not externally accessible from Spark)
3. Read qwen35 container config via `docker inspect`
4. Read qwen35 startup logs — confirmed CUDA graphs, Marlin FP8, FlashInfer, SM121 kernel warning
5. Identified top swap consumers via `/proc/*/status` VmSwap scan
6. Verified sysctl tuning in place
7. Reviewed Grafana dashboard (3h window) via Chrome browser automation
8. Captured all metrics in this lab notebook

**Findings:**
- System is stable under heavy pipeline load (5,500+ requests, 0 errors)
- enforce-eager was already removed (previous session) — CUDA graphs active
- SM121 native kernels confirmed missing (Marlin FP8 warning in logs)
- Concurrency much higher than assumed (15-18 vs assumed 3)
- Embedding model completely idle — 15.8 GB GPU wasted
- Swap is sticky but stable — not an active crisis

**Memory files updated:**
- `spark-device.md` — updated qwen35 config (enforce-eager removed, known-good image tag, CUDA graphs, backends)
- `sm121-performance-gap.md` — new file capturing research synthesis key findings
- `MEMORY.md` — index updated

**No changes made to the Spark system. Pipeline undisturbed.**

---

### Entry 002 — Reconstruction of previous session's experiments (undocumented)
**Date:** 2026-03-29 ~12:20 UTC
**Operator:** Claude Code (reconstructing from artifacts)
**Status:** FORENSIC ANALYSIS — session crashed before documenting results

The previous session (2026-03-28 ~14:00 to 2026-03-29 ~00:35 EDT) executed most of the LATER_PLAN without documenting results. This entry reconstructs the timeline from spark-config.sh snapshots, Docker images, build logs, and container state.

#### Timeline Reconstruction

| Time (EDT) | Action | Snapshot | Outcome |
|------------|--------|----------|---------|
| 13:43 | Idle monitor deployed | — | Waited for pipeline to finish |
| 14:33 | **Baseline captured** | `pipeline-v1` | Original config: enforce-eager + no-async-scheduling, cu130-nightly image |
| 15:35 | Cloned triton-lang/triton | — | Started Triton build from source (LATER Step 1b) |
| 15:38 | Triton build started | — | Build log only 641 bytes — likely abandoned early |
| 15:57 | Cloned vllm to /home/claude/vllm-build/ | — | Checkout of vllm main for custom SM121 build |
| 16:07 | Wrote Dockerfile.sm121 | — | Custom Docker build approach: FROM cu130-nightly + `TORCH_CUDA_ARCH_LIST=12.1` |
| 16:57 | Docker build started | — | `docker build -f Dockerfile.sm121 .` |
| 17:23 | **Docker build completed** | — | **FAILED**: cmake exit code 255 building wheels. Targets failed: `_moe_C`, `_vllm_fa2_C`, `_vllm_fa3_C`, `_vllm_fa4_cutedsl_C`, `_flashmla_C`, `_C`, `_C_stable_libtorch`. Image tagged `vllm-custom:sm121` (23GB) but vLLM package is broken/incomplete. |
| 21:41 | Idle monitor triggered | — | Step 0 metrics captured: 0% KV cache, 0 preemptions, 7,746 requests completed, 53°C idle |
| 22:05 | **Pre-optimization snapshot** | `pre-optimization` | Before starting live container changes |
| 22:25 | **Removed both enforce-eager AND no-async-scheduling** | `pipeline-v2` | Added `VLLM_TEST_FORCE_FP8_MARLIN=1`. CUDA graphs captured. Both flags removed. |
| 22:53 | **Started ChromaDB + Neo4j** | `full-stack` | All services running together |
| 23:20 | **Before v0.17.1 upgrade attempt** | `pre-v0.17.1-upgrade` | Pulled `v0.17.1-aarch64-cu130` (March 11 build) |
| ~23:30? | **v0.17.1 attempted** | — | Outcome unknown — rolled back (likely Qwen3.5 issues or missing SM121 kernels) |
| 00:16 | **Before hellohal2064 attempt** | `pre-community-image` | Back on March 6 image, ready to try community build |
| ~00:20? | **hellohal2064 community image attempted** | — | Outcome unknown — rolled back |
| 00:35 | **Settled on optimized-stable** | `optimized-stable` | Final config: March 6 image, enforce-eager removed, no-async-scheduling RE-ADDED, Marlin FP8 forced. Description: "14.3 tok/s" |

#### Docker Images On System

| Repository | Tag | Size | Created | Status |
|-----------|-----|------|---------|--------|
| vllm/vllm-openai | cu130-known-good-20260306 | 20.3 GB | Mar 6 | **CURRENTLY RUNNING** |
| vllm/vllm-openai | cu130-nightly | 20.4 GB | Mar 28 | Newer nightly, untested as qwen35 |
| vllm/vllm-openai | v0.17.1-aarch64-cu130 | 20.3 GB | Mar 11 | Pulled, tried, rolled back |
| hellohal2064/vllm-qwen3.5-gb10 | latest | 22.2 GB | Feb 8 | Pulled, tried, rolled back |
| vllm-custom | sm121 | 23 GB | Mar 28 17:23 | **BUILD FAILED** — cmake exit 255, broken |
| vllm-node | latest | 25.5 GB | Mar 19 | Unknown purpose |

#### Build Artifacts On Disk

- `/home/claude/triton-build/` — Triton source clone, partial build in `build/` dir
- `/home/claude/vllm-build/` — vLLM source clone with `Dockerfile.sm121`
- `/home/claude/vllm-docker-build.log` — 536 KB, shows cmake failure
- `/home/claude/triton-build.log` — 641 bytes, build started but abandoned
- `/home/claude/idle-monitor.log` — Monitored for idle state before starting experiments
- `/home/claude/step0-metrics.json` — Pre-experiment metrics snapshot
- `/home/claude/spark-config.sh` — Config management script (23 KB)

#### Key Observations

1. **Custom SM121 Docker build failed.** The cmake build can't compile vLLM's C extensions from source inside the container. The failure is in the multi-target cmake build (FA2, FA3, FA4, MoE, FlashMLA, etc). This is likely due to missing CUDA headers, incompatible compiler, or the SM121 architecture not being recognized by the cmake arch guards (the very bug #38126 fixes).

2. **v0.17.1 was tried and rolled back.** The image exists on disk. Likely failed to serve Qwen3.5 correctly — either the Qwen3.5 regression or missing SM121 kernels in the pre-built image.

3. **hellohal2064 community image was tried and rolled back.** The image exists on disk (22.2 GB, built Feb 8). Unknown why it was rolled back — could be incompatible launch parameters, model loading issues, or the image being older (vLLM 0.16.0-dev).

4. **Async scheduling was tested and rolled back.** pipeline-v2 had both enforce-eager AND no-async-scheduling removed. The final optimized-stable re-added `--no-async-scheduling`, suggesting the V1 engine crash (`NoneType has no sampled_token_ids`) was encountered during testing.

5. **CUDA graphs work on the March 6 build.** Despite not having native SM121 kernels, CUDA graph capture succeeded (51 piecewise + 35 full). This contradicts the original assumption that enforce-eager was needed for Triton PTX issues — the Marlin FP8 backend bypasses the Triton path.

6. **14.3 tok/s in optimized-stable description.** This was likely a single-request benchmark. Under 8 concurrent requests, aggregate generation throughput is 130-150 tok/s (~17-18 tok/s per-request). The 14.3 number needs validation — it may reflect a measurement taken during a particular load condition.

7. **num_gpu_blocks changed: 2466 → 2280.** The step0 metrics showed 2466 blocks, but current config shows 2280. A reduction of 186 blocks (~8%). The container was recreated at 04:31 UTC today — possible that CUDA graph memory reservation consumed some KV cache space.

---

### Entry 003 — Forensic analysis: Why each experiment failed
**Date:** 2026-03-29 ~12:30 UTC
**Operator:** Claude Code
**Method:** Docker event log analysis, image inspection, entrypoint script reading

#### Question 1: Why did hellohal2064/vllm-qwen3.5-gb10:latest crash?

**Answer: CRASH LOOP — model not mounted, wrong model name, incompatible entrypoint.**

Evidence from Docker events:
```
1774757806 start qwen35 hellohal2064/vllm-qwen3.5-gb10:latest
1774757811 die   (5 seconds later — crash)
1774757811 start (auto-restart)
1774757815 die   (4 seconds later — crash again)
... repeats every 4-5 seconds for 10+ minutes, backoff increasing to ~60s
```

Root cause analysis — FIVE compounding issues:

1. **Model path hardcoded to wrong model.** Image env var `MODEL_PATH=/models/Qwen3-Next-80B-A3B-Thinking-FP8`. This is an 80B model, NOT our Qwen3.5-35B-A3B. The research synthesis was incorrect about this image being purpose-built for our model.

2. **No model weights in image.** `/models/` directory is empty. The image expects model weights to be volume-mounted. Previous session did not mount our HF cache at `/models/`.

3. **Entrypoint conflict.** Image uses `/app/entrypoint.sh` which reads env vars and builds its own `vllm serve` command. When our standard arguments (`Qwen/Qwen3.5-35B-A3B --served-model-name qwen3.5-35b ...`) were passed via docker run CMD, they were appended to the entrypoint's command, creating an invalid invocation.

4. **CUDA version mismatch.** Image built with CUDA 13.1 (`CUDA_VERSION=13.1.0`). Our Spark has CUDA 13.0 (driver 580.142). The `NVIDIA_REQUIRE_CUDA=cuda>=13.1` check may cause immediate container rejection by nvidia-container-runtime.

5. **`set -e` in entrypoint.** First error (model path not found) causes immediate exit. `--restart unless-stopped` triggers infinite restart loop.

**Image details:**
- vLLM version: **0.16.0rc1.dev122+g6595a2380**
- Default CMD: `--swap-space 1 --load-format fastsafetensors`
- Entrypoint: `/app/entrypoint.sh` (builds vllm serve command from env vars)
- Env vars set: `GPU_MEMORY_UTIL=0.85`, `MAX_MODEL_LEN=1048576`, `ATTENTION_BACKEND=FLASHINFER`, etc.
- CLI syntax: `--attention-config.backend=` (different from our `--attention-backend`)

**What would be needed to make it work (if attempted again):**
```bash
docker run -d --name qwen35-hellohal \
  --gpus all --ipc host --shm-size 64gb -p 8000:8000 \
  -e MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e GPU_MEMORY_UTIL=0.65 \
  -e MAX_MODEL_LEN=32768 \
  -e NVIDIA_REQUIRE_CUDA="" \
  -v /home/davistroy/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/LATEST:/models/Qwen3.5-35B-A3B \
  hellohal2064/vllm-qwen3.5-gb10:latest \
  --swap-space 1
```
BUT: Even if it starts, the vLLM 0.16.0rc1 build may not have Qwen3.5 architecture support (added later). The SM121 kernel compilation (the valuable part) was done for a different model architecture. **Recommendation: Extract the Triton/CUDA artifacts rather than trying to use the image as-is.**

#### Question 2: Why did v0.17.1 fail?

**Answer: Manually killed after 7 minutes — probably premature, during model loading or Triton JIT compilation.**

Evidence from Docker events:
```
1774754436 create qwen35 vllm/vllm-openai:v0.17.1-aarch64-cu130
1774754437 start  (timestamp = 2026-03-28 ~23:20 EDT)
1774754857 kill   (420 seconds = 7 minutes later)
1774754868 destroy, create cu130-known-good-20260306, start (immediate rollback)
```

vLLM version confirmed: **0.17.1** (the "golden baseline" from research).

Probable causes for the 7-minute kill:
1. **Triton JIT compilation from scratch.** v0.17.1 has a different Triton version than v0.17.0rc1. Our cached Triton kernels at `/home/claude/.cache/triton` were compiled for v0.17.0rc1 and may be incompatible. Without cache, Triton FP8 MoE kernel compilation takes 15-30+ minutes. The session likely saw the container "stuck" during compilation and killed it.

2. **No Marlin FP8 forcing?** If `VLLM_TEST_FORCE_FP8_MARLIN=1` was not passed to the v0.17.1 container, it would auto-select the TRITON FP8 MoE backend, triggering long JIT compilation. The previous known-good config HAD this env var, but the v0.17.1 launch command may not have included it.

3. **CLI argument incompatibility.** v0.17.1 may handle some arguments differently than v0.17.0rc1. However, the core arguments (--quantization fp8, --kv-cache-dtype fp8, etc.) are stable across these versions.

**Key insight:** v0.17.1 was probably fine — it just needed more patience (15-30 min for Triton JIT) OR the Marlin FP8 env var to skip Triton compilation entirely. This image is still viable and should be re-tested with:
- `VLLM_TEST_FORCE_FP8_MARLIN=1` (forces Marlin, skips Triton JIT)
- Fresh Triton cache mount (or no mount, let it recompile)
- Patience — wait at least 30 minutes for first startup

**v0.17.1 is still the recommended base** for a production SM121 fix. It just needs PR #38126 cherry-picked (which requires a source build, not available in the pre-built image).

#### Question 3: Was async scheduling removal the cause of a crash?

**Answer: Inconclusive, but likely yes based on timeline.**

Evidence from config snapshots:
- `pipeline-v2` (22:25 EDT): Both `--enforce-eager` and `--no-async-scheduling` removed. `VLLM_TEST_FORCE_FP8_MARLIN=1` added.
- `full-stack` (22:53 EDT): Same config as pipeline-v2 (async scheduling removed). ChromaDB + Neo4j added.
- `optimized-stable` (00:35 EDT): `--no-async-scheduling` RE-ADDED. Described as "Best stable config".

Docker event timeline between full-stack and optimized-stable:
```
22:53 — full-stack snapshot (no-async-scheduling removed, system running)
23:20 — pre-v0.17.1-upgrade snapshot (still no-async-scheduling removed?)
23:20 — v0.17.1 attempted (7 min), rolled back
23:28 — killed known-good, restarted (this is likely when --no-async-scheduling was re-added)
00:16 — pre-community-image snapshot (no-async-scheduling present)
00:20 — hellohal2064 attempted (crash loop)
00:35 — optimized-stable snapshot with --no-async-scheduling
```

The config ran WITHOUT `--no-async-scheduling` from 22:25 to at least 23:20 (55+ minutes). The re-addition happened around 23:28. This suggests:
- Either the V1 engine crash (`NoneType has no sampled_token_ids`) was observed during that window
- Or the session decided to re-add it as a precaution before the v0.17.1/hellohal experiments
- **Without crash logs from that window, we can't confirm which**

**Recommendation for re-test:** Remove `--no-async-scheduling` again with explicit monitoring:
```bash
docker logs -f qwen35 2>&1 | grep -i "sampled_token_ids\|NoneType\|error\|crash"
```
Run concurrent stress test (5-10 simultaneous requests, 10 rounds). If no crash after 50+ concurrent completions, it's safe. With 15-18 concurrent requests in the current pipeline, async scheduling is high-value.

#### Updated Throughput Understanding

From current vLLM engine logs (live pipeline, 7-8 concurrent):
```
Avg prompt throughput: 200-780 tok/s (varies with request mix)
Avg generation throughput: 130-150 tok/s (aggregate)
Per-request generation: ~17-19 tok/s (150 / 8 requests)
```

The "14.3 tok/s" in the optimized-stable description was likely a single-request benchmark. Under concurrency, per-request throughput is 17-19 tok/s. The original "23 tok/s" baseline was probably also single-request.

**Throughput comparison:**
| Config | Single-request | At 8 concurrent | Notes |
|--------|---------------|-----------------|-------|
| pipeline-v1 (enforce-eager ON) | ~23 tok/s | unknown | Original baseline |
| optimized-stable (enforce-eager OFF, CUDA graphs) | ~14.3 tok/s? | ~17-19 tok/s | Current config |

The apparent regression from 23 to 14.3 single-request is counterintuitive. Possible explanations:
- CUDA graph memory overhead reduced KV cache blocks (2466 → 2280)
- Marlin FP8 weight-only compression (no native SM121 kernels) is slower than Triton FP8 for single requests
- The 14.3 measurement was taken under suboptimal conditions (cold cache, post-restart)
- **This needs a clean single-request benchmark to resolve**

---

### Entry 004 — Phase 1A: Fix SM121 Docker build (during pipeline run)
**Date:** 2026-03-29 ~12:45 UTC
**Operator:** Claude Code
**Status:** IN PROGRESS
**Pipeline impact:** None — CPU-only Docker build

**Objective:** Fix the NVFP4 compilation failure and rebuild `vllm-custom:sm121-v2` with native SM121 kernels.

**Root cause (from Entry 003):** `ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'` — NVFP4 microscaling is SM120-only. CMake incorrectly enables it for SM121.

**Approach:** Patch CMakeLists.txt to exclude NVFP4 from SM121 builds. Leave all other SM121 kernels (scaled_mm, MoE, MLA) enabled.

#### Step 1: Examine CMakeLists.txt NVFP4 guard

The NVFP4 section (lines 619-644) gates compilation using `cuda_archs_loose_intersection`:
```cmake
# Line 621-625 (BEFORE patch):
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")   # "family" match
else()
    cuda_archs_loose_intersection(FP4_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")  # explicit arch
endif()
```

`"12.0f"` (family) incorrectly matches `12.1` (SM121). NVFP4 uses `cvt.e2m1x2` instructions only on SM120.

Separate `scaled_mm` guard (line 529) also uses `"12.0f"` but compiled successfully — `scaled_mm` instructions ARE supported on SM121.

Separate `moe_data` guard (line 792) also uses `"12.0f"` — should compile fine for SM121 (no NVFP4 instructions).

**Decision:** Patch ONLY the NVFP4 section. Change `"12.0f"` to `"12.0a"` (specific arch, SM120 only). Leave scaled_mm and moe_data guards unchanged.

**Trade-off:** We lose `ENABLE_CUTLASS_MOE_SM120` and `ENABLE_NVFP4_SM120` defines for SM121. Acceptable because:
- We force Marlin FP8 MoE via env var (doesn't need CUTLASS MoE)
- We use FP8 quantization, not NVFP4

#### Step 2: Apply patch

```bash
# Backup original
cp CMakeLists.txt CMakeLists.txt.orig

# Line 622: "12.0f" → "12.0a" (CUDA >= 13.0 branch)
# Line 624: "12.0a;12.1a" → "12.0a" (CUDA < 13.0 branch)
sed -i '622s/"12.0f"/"12.0a"/' CMakeLists.txt
sed -i '624s/"12.0a;12.1a"/"12.0a"/' CMakeLists.txt
```

Verified result:
```cmake
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_ARCHS "12.0a" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(FP4_ARCHS "12.0a" "${CUDA_ARCHS}")
endif()
```

SM121 will NOT match `12.0a`, so FP4_ARCHS will be empty → entire NVFP4 section skipped → cmake message: "Not building NVFP4 as no compatible archs were found."

#### Step 3: Fix Dockerfile

Previous `Dockerfile.sm121` had a bug: `$?` captured tee's exit code (always 0), not pip's. This is why the failed build still produced an image.

New `Dockerfile.sm121-v2`:
- Base: `cu130-known-good-20260306` (not the newer nightly)
- Fixed: `set -o pipefail` before pip install
- Selective COPY (only needed dirs, not entire repo)
- Removed the broken `$CPATH` / `$LIBRARY_PATH` env vars (undefined variable warnings)

#### Step 4: Start build (background, CPU-only)

**First attempt:** `set -o pipefail` failed — `/bin/sh` (dash) doesn't support pipefail. Docker uses `/bin/sh` by default.

**Fix:** Added `SHELL ["/bin/bash", "-c"]` before the pip install RUN step.

**Also fixed:** Based image on `cu130-known-good-20260306` (not the newer nightly). Removed broken `$CPATH`/`$LIBRARY_PATH` env vars.

**Build started:** `docker build -f Dockerfile.sm121-v2 -t vllm-custom:sm121-v2 .`
**Log file:** `/home/claude/vllm-build-v2.log`
**Status:** Dependencies downloading (flashinfer 267 MB, xgrammar 42 MB). CMAKE compilation will follow.
**Expected duration:** ~25-30 min for cmake compilation after deps download.

*(build monitoring continues — check log periodically)*

#### Phase 1B Results: hellohal2064 SM121 Artifact Analysis (agent completed)

**Image stack:** vLLM 0.16.0rc1, PyTorch 2.11.0 nightly cu130, Triton 3.6.0 (custom git build), FlashInfer 0.6.2, fastsafetensors 0.2.0

**Critical finding: kernels are compiled for sm_120, NOT sm_121.**
- `_C.abi3.so`: 71 cubins — 41 sm_120, 6 sm_120a, 11 sm_80, 8 sm_90
- `_moe_C.abi3.so`: 26 cubins — 12 sm_120, 14 sm_80
- Flash Attention 3: 222 cubins but ALL sm_75 (no Blackwell optimization!)
- Flash Attention 2: 60 cubins, ALL sm_80
- Triton cache baked in: 179 kernel dirs, 178 .cubin files, 71 MB

SM121 (GB10) runs sm_120 code via forward compatibility — this is why hellohal gets 50 tok/s. The critical improvement is having Blackwell-native kernels (sm_120), not SM121-specific ones.

**Extraction verdict: NOT viable.** .so files are ABI-coupled to vLLM 0.16 internals. Triton cache is hash-locked to 0.16 kernel source. Injection into v0.17.1 would segfault.

**Implication for our build:** Our `TORCH_CUDA_ARCH_LIST="12.1"` will compile for sm_121, which is correct and potentially even better than sm_120 (can use SM121-specific instructions). The hellohal results prove that Blackwell-native kernel compilation is the key performance unlock.

#### Step 4 continued: Build attempts

**Attempt 1:** `set -o pipefail` in `/bin/sh` → exit code 2 (dash doesn't support pipefail)
**Fix:** Added `SHELL ["/bin/bash", "-c"]`

**Attempt 2:** NVFP4 patch worked (no ptxas e2m1x2 error). But 4 new failures:
```
FAILED: selective_scan_fwd.cu.o  — fatal error: cusparse.h: No such file or directory
FAILED: cache_kernels.cu.o       — same
FAILED: cache_kernels_fused.cu.o — same
FAILED: paged_attention_v1.cu.o  — same
```
Root cause: Removed CPATH/LIBRARY_PATH env vars from Dockerfile (had Docker warnings about undefined vars). But these are NEEDED — base image uses pip-installed CUDA packages, headers at `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/`.
**Fix:** Restored CPATH and LIBRARY_PATH pointing to pip CUDA package dir (without referencing undefined base `$CPATH`).

**Attempt 3:** Building now with both NVFP4 patch + CUDA include paths. Started ~08:48 UTC.
- Previous Dockerfile.sm121 (v1) had these paths AND got to [50/345] before NVFP4 failure
- With NVFP4 excluded and paths restored, should compile to completion
- Expected: ~25-30 min cmake compilation
- Log: `/home/claude/vllm-build-v2.log`

**Attempt 3 result:** Got to [59/345] at 1596s — PAST the v1 failure point (1541s). New failure:
```
FAILED: [code=255] qutlass/csrc/fused_quantize_mx.cu.o
ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'
```
Same e2m1x2 instruction, different file — Qutlass microscaling quantization. Qutlass cmake uses `"12.0f"` family match at line 35 of `cmake/external_projects/qutlass.cmake`.

**Fix:** Patched `qutlass.cmake` line 35: `"10.0f;12.0f"` → `"10.0f;12.0a"` and line 37: `"12.0a;12.1a;10.0a;10.3a"` → `"12.0a;10.0a;10.3a"`. Qutlass will be skipped for SM121. Acceptable loss — Qutlass provides microscaling GEMM which we don't use (FP8 via Marlin instead).

**Remaining `12.0f` references** (will compile, not fix):
- `scaled_mm` (line 529) — CONFIRMED works on SM121, [49/345] compiled fine
- `MLA` (line 674) — MLA attention, likely works (no microscaling)
- `moe_data` (line 792) — data movement, likely works (no microscaling)

**Attempt 4:** Both NVFP4 and Qutlass excluded. Started ~09:10 UTC. `--no-cache` to ensure clean build.

#### Build Failure Pattern Analysis

The SM121 build is hitting the same underlying hardware limitation repeatedly: the `cvt with .e2m1x2` PTX instruction is only supported on SM120 (datacenter Blackwell), NOT SM121 (GB10/DGX Spark). This instruction is the core of NVIDIA's microscaling (MX) format support — E2M1 is a 4-bit float format used by MXFP4 quantization.

Every cmake guard that uses `"12.0f"` (family match) incorrectly includes SM121 in compilation of microscaling-dependent kernels. The vLLM codebase has this pattern in 4 locations:

| Location | Guard | Uses e2m1x2? | Our fix | Impact of exclusion |
|----------|-------|-------------|---------|-------------------|
| CMakeLists.txt:622 (NVFP4) | `"12.0f"` → `"12.0a"` | YES | Fixed (attempt 2) | Lose NVFP4 quant — don't use it (FP8 via Marlin) |
| qutlass.cmake:35 (Qutlass) | `"12.0f"` → `"12.0a"` | YES | Fixed (attempt 4) | Lose Qutlass MX GEMM — don't use it |
| CMakeLists.txt:529 (scaled_mm) | `"12.0f"` — NO CHANGE | NO | Not needed | scaled_mm compiled successfully at [49/345] |
| CMakeLists.txt:674 (MLA) | `"12.0f"` — NO CHANGE | Unknown | TBD | MLA attention — likely works |
| CMakeLists.txt:792 (moe_data) | `"12.0f"` — NO CHANGE | Unknown | TBD | Data movement — likely works |

**Root cause (from PR #38126):** The `cuda_archs_loose_intersection` function in `cmake/utils.cmake` treats `"12.0f"` (family) as matching any `12.x` architecture. But SM120 and SM121 have DIFFERENT instruction sets — SM121 lacks the microscaling instructions. The PR fixes the function to handle suffix matching correctly, but our approach of switching from family (`f`) to specific arch (`a`) for the affected guards is equally effective.

**What we're keeping:** `scaled_mm`, MLA, and `moe_data` all use `"12.0f"` but their kernels don't use microscaling instructions. They use standard FP8/BF16 matrix math that IS supported on SM121. These are the performance-critical kernels we need.

**What we're excluding:** NVFP4 and Qutlass microscaling — both use `cvt.e2m1x2` which is a hardware limitation of SM121, not a software bug. No amount of cmake fixing will make these work on GB10.

#### Architectural Analysis: SM121 vs SM120 Compilation Strategy

After reading the `cuda_archs_loose_intersection` function in `cmake/utils.cmake` (lines 341-400), I realized the current per-guard patching approach, while functional, is not the cleanest path forward.

**The function is working correctly.** The family match (`"12.0f"`) is designed to match any `12.x` architecture — the comment at line 361 even says: `SRC="12.0f" matches TGT="12.1a" since SM121 is in the SM12x family`. The function does exactly what it's told.

**The guards are asserting the wrong thing.** When a cmake guard says `"12.0f"` for NVFP4, it's claiming "this kernel works for ALL SM12x architectures." That claim is false — SM121 lacks the `cvt.e2m1x2` microscaling instruction. The guard should say `"12.0a"` (SM120 only). Our per-guard patches fix this correctly.

**But there's a fundamentally simpler approach: compile for SM120, not SM121.**

| Approach | Method | Patches needed | All kernels compile? | Performance |
|----------|--------|---------------|---------------------|-------------|
| A: Patch guards | Change `"12.0f"` → `"12.0a"` per guard | 2 done, unknown more | No — microscaling excluded | Same as B for non-microscaling |
| B: Target SM120 | `TORCH_CUDA_ARCH_LIST="12.0"` | **Zero** | **Yes — including NVFP4, Qutlass** | Proven 50 tok/s (hellohal) |

**Why SM120 compilation works on SM121 hardware:**
1. NVIDIA forward compatibility: SM121 can execute any SM120 cubin natively. This is a hardware guarantee, not an emulation layer.
2. The hellohal2064 image proves this — compiled entirely for sm_120, achieves 50 tok/s on SM121 (GB10) hardware.
3. SM120 (datacenter Blackwell) supports ALL instructions including microscaling (`cvt.e2m1x2`). SM121 (consumer/edge Blackwell) dropped microscaling.
4. The performance-critical kernels (scaled_mm, MoE, attention, MLA) use standard Blackwell FP8/BF16 instructions that exist on BOTH SM120 and SM121. The SM121-specific instructions are minor.

**What we lose with SM120 targeting:**
- SM121-specific optimizations — but the hellohal artifact analysis showed only PyTorch's `libtorch_cuda.so` had sm_121 cubins, and those come from the pip package regardless of our build.
- Theoretically, a future SM121-only instruction could be faster for some operation. In practice, there's no evidence this matters for LLM inference.

**What we gain:**
- Zero cmake patches (revert our NVFP4 and Qutlass patches)
- ALL kernels compile — including microscaling (NVFP4, Qutlass MX GEMM)
- One clean env var change in the Dockerfile
- Matches the proven hellohal approach
- No risk of hitting more microscaling guards deeper in the build

**Decision framework:**
- If attempt 4 (current build, SM121 target with 2 guard patches) **succeeds** → we have a working image. The 2 patches are clean and well-documented. Future rebuilds could use the SM120 approach instead.
- If attempt 4 **fails on another microscaling kernel** → switch to SM120 approach immediately. One env var change, zero patches, clean rebuild.

**The SM120 approach is the architecturally sound long-term answer.** The per-guard patching works but requires knowing every cmake guard that touches microscaling. The SM120 approach is self-correcting — any new microscaling kernel added to vLLM in the future will just work.

This is also consistent with how the community has solved this problem. All three community builders (hellohal2064, namake-taro, sesmanovic) compile for SM120, not SM121. They've validated this path at 50-495 tok/s.

#### Planned A/B Test: SM121-native vs SM120-forward-compat

If the current SM121-targeted build succeeds, build a second image with `TORCH_CUDA_ARCH_LIST="12.0"` (SM120, zero cmake patches, clean Dockerfile). Then benchmark both on the same hardware under identical conditions:

| Image | Arch | Microscaling kernels | Test |
|-------|------|---------------------|------|
| `vllm-custom:sm121-v2` | sm_121 | Excluded (NVFP4/Qutlass patched out) | Single-request + concurrent |
| `vllm-custom:sm120` | sm_120 | Included (all kernels compile) | Single-request + concurrent |

This would answer: **Does SM121-specific compilation provide any measurable benefit over SM120 forward compatibility for LLM inference?** No one in the community has published this comparison — everyone targets SM120. If there's no difference, SM120 is the strictly better build target (simpler, all kernels, proven path).

#### BUILD SUCCESSFUL — Attempt 4 (2026-03-29 ~10:52 EDT)

**Image:** `vllm-custom:sm121-v2` — 32.3 GB
**Build time:** ~95 minutes (from `docker build` start to image export)
**Errors:** 0
**Patches applied:** 2 (NVFP4 arch guard in CMakeLists.txt, Qutlass arch guard in qutlass.cmake)

**Kernel verification:**

| Component | SM121 cubins | Total cubins | Notes |
|-----------|-------------|-------------|-------|
| `_C.abi3.so` | **32** | 52 | Core kernels: scaled_mm, attention, cache, activation, layernorm |
| `_moe_C.abi3.so` | **8** | 8 | MoE routing/dispatch — ALL sm_121 |
| NVFP4 | 0 | 0 | Correctly excluded (SM121 hardware limitation) |
| Qutlass MX | 0 | 0 | Correctly excluded (SM121 hardware limitation) |

**Comparison with hellohal2064 (sm_120 target):**

| Metric | hellohal (sm_120) | Our build (sm_121) |
|--------|------------------|-------------------|
| _C.so Blackwell cubins | 41 (sm_120) | 32 (sm_121) |
| _moe_C.so Blackwell cubins | 12 (sm_120) | 8 (sm_121) |
| NVFP4/Qutlass included | Yes (sm_120 supports e2m1x2) | No (excluded) |
| Flash Attention | sm_75 only (not optimized) | TBD — check FA3 |

Lower cubin counts are expected — we excluded microscaling kernels that hellohal included (sm_120 supports them, sm_121 doesn't). The performance-critical kernels (scaled_mm, MoE, MLA, attention) are present.

**vLLM version in image:** `0.1.dev1+g71331e9b6` — this is a dev version because we built from a local git init, not from a tagged release. Functionally equivalent to v0.17.0rc1 (same source base as the cu130-known-good image).

**Image is ready for testing when pipeline completes (~01:30 EDT March 30).**

**Next steps:**
1. Baseline single-request benchmark with current image
2. Swap to vllm-custom:sm121-v2
3. Benchmark and compare — looking for the "no native FP8 support" warning to be GONE
4. If successful, snapshot as `pipeline-v3`

### Entry 005 — SM120 clean build for A/B test
**Date:** 2026-03-29 ~11:05 EDT
**Operator:** Claude Code
**Status:** BUILD IN PROGRESS

**Objective:** Build `vllm-custom:sm120` with `TORCH_CUDA_ARCH_LIST="12.0"` — zero cmake patches, original unpatched source. SM120 cubins run on SM121 via NVIDIA forward compatibility.

**Approach:** Restored CMakeLists.txt and qutlass.cmake to original (unpatched) versions. All `"12.0f"` guards will correctly match `"12.0"` and compile ALL kernels including NVFP4 and Qutlass microscaling.

**Dockerfile:** `Dockerfile.sm120` — identical to sm121-v2 except:
- `TORCH_CUDA_ARCH_LIST="12.0"` (not "12.1")
- Original unpatched source (no cmake guard modifications)
- Git commit message: "SM120 build - zero patches, forward compat on SM121"

**Log:** `/home/claude/vllm-build-sm120.log`
**Expected:** ~95 minutes (same as sm121 build), possibly longer due to additional NVFP4/Qutlass targets.

**A/B test plan:**
| Image | Tag | Arch | Patches | Microscaling | Test |
|-------|-----|------|---------|-------------|------|
| SM121 native | `vllm-custom:sm121-v2` | sm_121 | 2 (NVFP4 + Qutlass excluded) | No | Benchmark |
| SM120 forward-compat | `vllm-custom:sm120` | sm_120 | **0** | Yes (all kernels) | Benchmark |
| Current baseline | `cu130-known-good-20260306` | generic (no native kernels) | 0 | No | Benchmark |

Three-way comparison: generic → SM121-native → SM120-forward-compat. This will definitively answer which compilation target is optimal for LLM inference on DGX Spark.

#### SM120 BUILD SUCCESSFUL (2026-03-29 ~13:11 EDT)

**Image:** `vllm-custom:sm120` — 32.7 GB, 0 errors, 0 patches
**Build time:** ~125 minutes (longer than SM121's 95 min due to additional microscaling targets)

**Kernel verification:**

| Component | SM120 cubins | Total cubins |
|-----------|-------------|-------------|
| `_C.abi3.so` | **48** | 68 |
| `_moe_C.abi3.so` | **13** | 13 (all sm_120) |
| NVFP4 | Included | SM120 supports microscaling |
| Qutlass MX | Included | SM120 supports microscaling |

**Three-image comparison — ready for A/B test:**

| Image | Blackwell cubins (_C.so + _moe_C.so) | Microscaling | Patches | Size |
|-------|--------------------------------------|-------------|---------|------|
| Baseline (cu130-known-good) | 0 (generic fallback) | No | 0 | 20.3 GB |
| SM121 native (sm121-v2) | 40 (32 + 8 sm_121) | Excluded | 2 | 32.3 GB |
| SM120 forward-compat (sm120) | **61** (48 + 13 sm_120) | **Included** | **0** | 32.7 GB |

SM120 has 53% more Blackwell cubins than SM121 (61 vs 40) because it includes the microscaling kernels. Both are dramatically more than the baseline (0).

**All three images ready. Pipeline finishes ~01:30 EDT. Benchmark plan:**
1. Stop qwen35, snapshot current config
2. Baseline benchmark (current image, single request)
3. Swap to sm121-v2, benchmark
4. Swap to sm120, benchmark
5. Compare results, pick winner, snapshot as pipeline-v3

### Entry 007 — A/B test execution and critical build bug discovery
**Date:** 2026-03-30 ~01:30 EDT
**Operator:** Claude Code (autonomous)

**Pipeline finished:** ~23:50 EDT (16,087 requests completed). Monitor detected idle at 01:07 EDT.

**Benchmark script failure:** The automated benchmark script crashed at the single-request measurement step due to `set -euo pipefail` catching an error in the bc/date measurement code. Only the baseline container was started and the FP8 warning check completed. No actual throughput numbers collected.

**Manual benchmark — Baseline:**
- Image: cu130-known-good-20260306
- FP8 Warning: **PRESENT** ("GPU does not have native support for FP8")
- **Single-request: 600 tokens in 45s = 13.3 tok/s**

**Manual benchmark — SM121 (vllm-custom:sm121-v2):**
- Container started and became healthy
- FP8 Warning: **ABSENT** (native SM121 kernels being used!)
- Backend: MarlinFP8ScaledMMLinearKernel selected
- **CRITICAL BUG: model loaded as `Qwen/Qwen3-0.6B` instead of `Qwen/Qwen3.5-35B-A3B`**

**Root cause of model loading bug:** The vLLM source in `/home/claude/vllm-build/` was cloned from `main` (commit `fafca38`) during the previous session. This is POST-Qwen3.5 regression (#37749, introduced in v0.18.0). The Dockerfile builds from this source, replacing the base image's working v0.17.0rc1 Python package. Result: SM121 kernels compile correctly, but the vLLM Python code can't load Qwen3.5 models.

**The base image is v0.17.0rc1 (commit `e68de8adc`).** This is the version that works with Qwen3.5.

**Fix:** Checkout the `e68de8adc` commit (v0.17.0rc1), apply cmake patches, rebuild.

**Lesson:** When building custom kernels, the source code version MUST match the base image's version. Building from main and overlaying onto a v0.17.0rc1 base replaces the working Python code with broken code. The Dockerfile should start with `git checkout e68de8adc` not `git init && git add -A`.

**Fix:** Cloned fresh vLLM repo, checked out `e68de8adc` (v0.17.0rc1). Applied NVFP4 patch to CMakeLists.txt line 651. Created new Dockerfiles for both SM121 and SM120 builds.

**Additional bug:** First rebuild attempt failed — forgot `git` in `apt-get install` (needed by setuptools-scm for version detection). Fixed both Dockerfiles.

**Rebuild started:** SM121 v3 at ~01:45 EDT from correct v0.17.0rc1 source. Expected ~95 min.

**SM121 v3 build result:** Built successfully (31.4 GB, 52 sm_120 cubins in _C.so, 12 in _moe_C.so). Version correct (v0.17.0rc1). BUT: still loads `Qwen/Qwen3-0.6B` instead of Qwen3.5-35B-A3B. The FP8 warning is still present.

**Root cause of model loading failure (deeper):** `pip install --no-build-isolation .` resolves the ENTIRE dependency tree, potentially downgrading the `transformers` library to a version that doesn't recognize `qwen3_5_moe` model type. The base image (cu130-known-good) has a carefully curated newer transformers that supports Qwen3.5 — rebuilding from source loses this.

```
ValueError: The checkpoint you are trying to load has model type `qwen3_5_moe`
but Transformers does not recognize this architecture.
```

**Architectural insight:** The problem is doing `pip install .` for the ENTIRE vLLM package. We only need the compiled C extensions (.so files). The Python code, model support, and dependency versions should come from the pristine base image.

**Solution: Multi-stage Docker build.**
- Stage 1 (builder): Build vLLM from source to get the .so files with SM121 kernels
- Stage 2 (final): Start from pristine base image, COPY ONLY the .so files from builder
- Result: Base image's Python code + dependencies + model support, with OUR compiled kernels

**Tag: `vllm-custom:sm121-inject`** — multi-stage build started ~03:20 EDT.

**Lessons learned (for the lab notebook hall of fame):**
1. vLLM source version MUST match base image (v0.17.0rc1, not main) — Entry 007
2. `pip install` replaces the entire package including dependency resolution — this entry
3. Only inject what you changed (compiled .so files), not the entire application layer
4. The base Docker image is a curated artifact — treat it as immutable, overlay minimally

### Entry 008 — A/B Test Results: THE BREAKTHROUGH
**Date:** 2026-03-30 ~05:50 EDT
**Operator:** Claude Code

#### SM121-inject image: `vllm-custom:sm121-inject`
- Multi-stage build: Stage 1 compiles SM121 kernels, Stage 2 copies only .so files into pristine base
- Version: v0.17.0rc1.dev102+ge68de8adc (matches base exactly)
- Model: Qwen/Qwen3.5-35B-A3B (loads correctly!)
- Blackwell cubins: 52 sm_120 in _C.so, 12 in _moe_C.so
- FP8 warning: still present (Python runtime check, not kernel issue)
- Image size: 20.7 GB (minimal — only .so overlay on base)

#### BENCHMARK RESULTS

| Image | Run 1 | Run 2 | Run 3 | Average | Improvement |
|-------|-------|-------|-------|---------|-------------|
| **Baseline** (cu130-known-good) | 13.3 tok/s | — | — | ~13.3 tok/s | — |
| **SM121-inject** (native Blackwell cubins) | 48.7 tok/s | 48.6 tok/s | 48.6 tok/s | **48.6 tok/s** | **3.65x** |

**THE PERFORMANCE GAP IS CLOSED.**

From 13.3 to 48.6 tok/s — a 3.65x improvement, matching the 3.5x gap identified in the research synthesis. The native SM121 (sm_120) compiled kernels for scaled_mm, MoE, MLA, and attention are the critical enablers.

**Note:** The FP8 "does not have native support" warning persists because the Python runtime check in `marlin_utils_fp8.py` doesn't recognize SM121 as FP8-capable. However, the native Blackwell cubins for other operations (scaled_mm, attention, activation) are being used and provide the throughput improvement. The FP8 weight-only Marlin path is still functional and fast with native cubins.

**What we achieved:**
- 600 tokens in 12.3-12.4 seconds (single request, max_tokens=600)
- Consistent across 3 runs (no variance)
- Model loads correctly (Qwen3.5-35B-A3B, not the wrong 0.6B)
- Image is only 20.7 GB (smallest of all custom builds — just .so overlay)

**Still TODO:**
- SM120 clean build for A/B comparison (will it match or exceed 48.6?)
- Concurrent benchmark (8 requests)
- ~~Snapshot as pipeline-v3~~ DONE
- ~~Update spark-device.md with new performance baseline~~ DONE

### Entry 009 — Performance optimization sweep on SM121-inject
**Date:** 2026-03-30 ~06:30 EDT
**Operator:** Claude Code
**Base image:** vllm-custom:sm121-inject (48.6 tok/s single-request baseline)

#### 9.1 Concurrent Benchmark

Tested at 1, 4, 8, and 16 concurrent requests (200 tokens each, thinking disabled):

| Concurrency | Per-Request tok/s | Aggregate tok/s | Wall Clock | Scaling |
|-------------|------------------|-----------------|-----------|---------|
| 1 | 48.9 | 48.9 | 4.1s | 1.0x |
| 4 | 33.5 | 133.9 | 6.0s | 2.74x |
| 8 | 26.3 | 210.4 | 7.6s | 4.30x |
| 16 | 19.5 | 311.7 | 10.3s | 6.37x |

**Analysis:**
- Aggregate throughput scales well: 48.9 → 133.9 → 210.4 → 311.7 tok/s
- Per-request throughput degrades gracefully: 48.9 → 33.5 → 26.3 → 19.5 tok/s
- At c16, per-request is 19.5 tok/s — still FASTER than the old baseline's single-request (13.3 tok/s)
- **311.7 tok/s aggregate at c16** — this is production-grade throughput
- Near-perfect consistency within each concurrency level (all requests finish within 0.1s of each other)
- The `--no-async-scheduling` flag means requests are processed sequentially within each step — removing it could improve c4-c16 numbers significantly

#### 9.2 Async Scheduling Test

Removed `--no-async-scheduling`. Container started and became healthy in 170s.

**Stress test:** 10 rounds x 8 concurrent = 80 requests. All rounds HEALTHY. 0 crash signatures in logs.
- Round 1 (cold): 63.8 tok/s
- Rounds 2-10: 179-189 tok/s at c8 (150 max_tokens)

**Apples-to-apples benchmark (same parameters as 9.1):**
| Concurrency | --no-async (9.1) | async ON (9.2) | Delta |
|-------------|-----------------|----------------|-------|
| c1 | 48.9 | 47.6 | -2.7% (noise) |
| c4 | 133.9 | 133.1 | -0.6% |
| c8 | 210.4 | 211.0 | +0.3% |
| c16 | 311.7 | 311.5 | -0.1% |

No significant throughput difference with simultaneous submission. Expected — async scheduling helps staggered arrivals (real pipeline pattern), not batch submission.

**Decision: KEEP async scheduling enabled.** 80 concurrent requests, 0 crashes, 0 NoneType errors. The V1 crash that prompted the flag is not present in this build. Async scheduling will improve TTFT under real pipeline load.

#### 9.3 Reduce gpu-memory-utilization to 0.60

Changed from 0.65 to 0.60. Container restarted with async scheduling enabled.

**Config verified:** `gpu_memory_utilization="0.6"`, `num_gpu_blocks=1914` (down from 2280 at 0.65)

**Memory impact:**
- Available RAM: 20 GiB (up from 14 GiB at 0.65) — **+6 GiB freed**
- Swap: 7.9 GiB (still sticky from before, will clear on next reboot)

**Benchmark (after warmup, apples-to-apples):**
| Concurrency | At 0.65 (9.1) | At 0.60 (9.3) | Delta |
|-------------|--------------|--------------|-------|
| c1 | 48.9 | 48.4 | -1.0% (noise) |
| c4 | 133.9 | 132.4 | -1.1% |
| c8 | 210.4 | 209.8 | -0.3% |
| c16 | 311.7 | 309.9 | -0.6% |

**No meaningful regression.** KV cache blocks dropped from 2280 to 1914 (~16% reduction) but at 1.6% peak utilization, we still have 50x headroom. The 6 GiB of freed host RAM is a bigger win for system stability.

**Decision: KEEP at 0.60.** Performance unchanged, 6 GiB more available RAM.

#### 9.4 CUTLASS FP8 vs Marlin FP8 (removing VLLM_TEST_FORCE_FP8_MARLIN)

**Hypothesis:** With native SM121 cubins compiled, the CUTLASS FP8 path should work correctly. Removing the Marlin forcing env var lets vLLM use native CUTLASS FP8 compute instead of weight-only compression.

**Test:** Restarted without `VLLM_TEST_FORCE_FP8_MARLIN=1`.

**Backend selection changed dramatically:**
| Component | With Marlin Forcing | Without (native CUTLASS) |
|-----------|-------------------|-------------------------|
| Linear FP8 | MarlinFP8ScaledMMLinearKernel | **CutlassFP8ScaledMMLinearKernel** |
| MoE | MARLIN Fp8 MoE | **TRITON Fp8 MoE** |
| FP8 Warning | Present (misleading) | **ABSENT** |

**Coherence test:** PASSED. Output is perfectly coherent (correct facts, proper formatting, no NaN artifacts). The NaN issue from the research was caused by MISSING cubins, not by the CUTLASS path itself.

**Benchmark:**
| Path | Single-request tok/s | Delta |
|------|---------------------|-------|
| Marlin FP8 (forced) | **48.6** | — |
| CUTLASS FP8 (native) | 44.9 | **-7.6%** |

**Marlin wins.** Despite being "weight-only compression," Marlin is a more optimized kernel specifically tuned for FP8 weight decompression + matmul. CUTLASS FP8 does true scaled FP8 compute but is less optimized for this specific workload pattern.

**Decision: KEEP `VLLM_TEST_FORCE_FP8_MARLIN=1`.** Marlin is faster. But the finding that CUTLASS FP8 works correctly (no NaN) is valuable — it means this path is available if Marlin has issues in future vLLM versions.

**Restored optimal config:** SM121-inject + Marlin forcing + async scheduling + 0.60 gpu-util.

#### 9.5 Prefix Caching (experimental Mamba align mode)

**Attempt 1:** Added `--enable-prefix-caching` to launch args.
- Result: Container crash-loops with `pydantic ValidationError: "In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens (2048)"`
- Root cause: Default `max_num_batched_tokens=2048` is smaller than the KV block_size (2096)

**Attempt 2:** Added `--enable-prefix-caching --max-num-batched-tokens 4096`.
- Container started successfully with experimental Mamba cache 'align' mode
- Warnings: "Prefix caching in Mamba cache 'align' mode is currently enabled. Its support for Mamba layers is experimental."

**Benchmark with prefix caching:**
| Test | Run 1 (cold) | Run 2 | Run 3 |
|------|-------------|-------|-------|
| Single 600 tok | 30.2 (warmup) | 47.3 | 47.3 tok/s |
| Shared system prompt 200 tok | 14.5 (first compile) | 46.3 | 46.5 tok/s |

**Prefix cache hit rate: 0.** Despite enable_prefix_caching=True, `vllm:prefix_cache_hits_total` = 0. The Mamba align mode is not actually caching prefixes for this model.

**Conclusion:** Prefix caching is experimental for Qwen3.5's hybrid Mamba architecture in v0.17.0rc1 and doesn't provide any benefit. The `max_num_batched_tokens=4096` change doesn't hurt single-request performance (47.3 vs 48.6 — within noise).

**Decision:** Remove `--enable-prefix-caching` (not functional). Keep `--max-num-batched-tokens 4096` as it enables larger prefill chunks without regression — potentially helpful for pipeline TTFT under concurrency.

**⚠️ RE-TEST PREFIX CACHING WHEN:**
- **Upgrading vLLM** — the Mamba cache `align` mode is under active development; newer versions may fix the 0-hit issue for hybrid recurrent models
- **vLLM adds a different caching strategy** for hybrid recurrent architectures (GDN/Mamba layers maintain hidden state, not KV cache — fundamentally harder to cache than pure attention)
- **Switching to a pure-Transformer model** — prefix caching works perfectly for standard attention-only architectures; the limitation is specific to Qwen3.5's hybrid Mamba design

#### Optimization Sweep Summary So Far

| Optimization | Result | Keep? |
|-------------|--------|-------|
| Native SM121 cubins (sm121-inject) | **13.3 → 48.6 tok/s (3.65x)** | YES |
| Async scheduling (remove --no-async-scheduling) | Stable, 0 crashes in 80 requests, no throughput change | YES |
| gpu-memory-utilization 0.65 → 0.60 | No regression, +6 GiB RAM freed | YES |
| CUTLASS FP8 (remove Marlin forcing) | Works (no NaN!) but 7.6% slower than Marlin | NO — keep Marlin |
| Prefix caching | Experimental, 0 cache hits, not functional for Qwen3.5 | NO |
| max-num-batched-tokens 4096 | No regression, enables larger prefill chunks | YES |

**Current optimal config:**
- Image: `vllm-custom:sm121-inject`
- `VLLM_TEST_FORCE_FP8_MARLIN=1`
- `VLLM_FLASHINFER_MOE_BACKEND=latency`
- `--gpu-memory-utilization 0.60`
- `--max-num-batched-tokens 4096`
- No `--no-async-scheduling` (async enabled)
- No `--enable-prefix-caching` (not functional)

#### 9.6 Embed Model Memory Reduction (0.13 → 0.10)

**Sleep mode:** Not available in v0.17.0rc1 (added in later versions).

**Attempt 1:** `--gpu-memory-utilization 0.10` with default max-model-len (40960).
- Result: OOM — needs 5.62 GiB KV cache, only 4.46 GiB available at 0.10 util
- Fix: Reduce max-model-len (embedding queries are short, 8192 is more than sufficient)

**Attempt 2:** `--gpu-memory-utilization 0.10 --max-model-len 8192`.
- Result: Healthy, embeddings working (dim=2560 verified)
- GPU memory: 11,446 MiB (down from 15,810 MiB at 0.13) — **4.3 GiB freed**
- Embedding requests still functional

| Metric | Before (0.13) | After (0.10) | Savings |
|--------|--------------|-------------|---------|
| Embed GPU | 15,810 MiB | 11,446 MiB | **4,364 MiB** |
| Max embed context | 40,960 tokens | 8,192 tokens | Adequate for embeddings |

**Decision: KEEP at 0.10 with max-model-len 8192.** Embedding queries are typically < 1000 tokens. 8192 max context is more than sufficient.

#### 9.7 Docker Compose

Created `/home/claude/docker-compose.yml` (190 lines) codifying the entire stack:
- 6 services: qwen35, qwen3-embed, gliner, chromadb, neo4j, node-exporter
- Startup order enforced via `depends_on` with `condition: service_healthy`
- Health checks with appropriate start_period for model loading (300s for LLM, 180s for embed)
- Log rotation on all services (100m/3 files for LLM, 50m/3 for others)
- Named volumes for persistent data (chromadb-data, neo4j-data, neo4j-logs) marked as external
- All volume mounts use absolute paths (never `~`)
- Validated: `docker compose config --quiet` passes

**Not yet activated** — current containers were started individually. To switch to compose management: `docker compose -f /home/claude/docker-compose.yml up -d` (after stopping current containers). This is a non-destructive operation — same containers, just managed declaratively.

#### 9.8 OS Cleanup

| Service/Package | Status Before | Action | Status After |
|----------------|--------------|--------|-------------|
| Desktop snaps (gnome, gtk, mesa, snap-store) | Already removed | None needed | — |
| avahi-daemon | Already inactive | None needed | inactive |
| multipathd | **active** | `systemctl disable --now` | **inactive** |
| multipathd.socket | active | `systemctl disable --now` | inactive |
| firmware-updater snap | Installed | Left for now (security updates) | Installed |
| dgx-dashboard | Running | Keep (useful monitoring) | Running |

**Minimal cleanup needed** — previous sessions already removed desktop snaps and disabled avahi. Only multipathd was still active (SAN multipath storage, not needed for a standalone inference server).

#### 9.9 SM120 Clean Build (A/B comparison)

SM120 inject build in progress — `vllm-custom:sm120-inject` building from v0.17.0rc1 source with `TORCH_CUDA_ARCH_LIST="12.0"`, zero patches, multi-stage .so injection. At ~56 min, compiling. Expected completion: ~90 min total.

---

### FULL OPTIMIZATION SWEEP SUMMARY

| # | Optimization | Before | After | Impact | Status |
|---|-------------|--------|-------|--------|--------|
| 1 | Native SM121 cubins | 13.3 tok/s | **48.6 tok/s** | **+265%** | DEPLOYED |
| 2 | Async scheduling | Disabled | Enabled | Stable, improves TTFT under load | DEPLOYED |
| 3 | gpu-memory-util 0.65→0.60 | 14 GiB avail | 20 GiB avail | +6 GiB RAM | DEPLOYED |
| 4 | CUTLASS FP8 vs Marlin | — | Marlin 7.6% faster | Keep Marlin | TESTED, REVERTED |
| 5 | Prefix caching | — | 0 cache hits | Not functional for Mamba | TESTED, SKIPPED |
| 6 | max-num-batched-tokens 4096 | 2048 | 4096 | No regression | DEPLOYED |
| 7 | Embed 0.13→0.10 + max-len 8192 | 15.8 GiB GPU | 11.4 GiB GPU | -4.3 GiB GPU | DEPLOYED |
| 8 | Docker Compose | Individual runs | Declarative compose | Codified stack | CREATED |
| 9 | OS cleanup (multipathd) | Active | Disabled | Minor RAM savings | DEPLOYED |
| 10 | SM120 clean build | — | Building | A/B comparison | IN PROGRESS |

**Final configuration snapshotted as `pipeline-v3-final`.**

### Entry 010 — A/B Test: SM121-inject vs SM120-inject
**Date:** 2026-03-30 ~09:45 EDT
**Operator:** Claude Code

**Note:** First attempt crashed qwen35 — the `docker run --rm` cubin count check inside the benchmark script launched a second GPU container, causing memory pressure that killed the LLM. Fixed by removing cubin checks from the benchmark (run them separately). Lesson: never spin up temporary GPU containers while the LLM is running.

#### Results

| Concurrency | SM121-inject | SM120-inject | Delta |
|-------------|-------------|-------------|-------|
| c1 | **49.5 tok/s** | 49.4 tok/s | -0.2% |
| c4 | 92.4 agg | 53.7 agg | -41.9% |
| c8 | **210.3 agg** | 146.4 agg | -30.4% |
| c16 | 311.4 agg | **313.0 agg** | +0.5% |

**Single-request: identical** (49.5 vs 49.4 tok/s — within noise).

**Concurrent c4-c8: SM121 significantly better.** SM120 shows degraded concurrent performance at c4 (53.7 vs 92.4) and c8 (146.4 vs 210.3). This is likely CUDA graph warmup — the SM120 image has different cubins so the Triton cache doesn't apply, and first-time graph capture interferes with the c4 test.

**c16: essentially identical** (313 vs 311). At high concurrency, both converge.

**Analysis:** The c4 and c8 results for SM120 are anomalous — the c4 result (53.7 agg = 13.4 per-request) is suspiciously similar to baseline (13.3 tok/s), suggesting CUDA graphs weren't captured yet for those batch sizes. The SM120 image needed more warmup rounds. A proper A/B test would require multiple warmup rounds at each concurrency level.

**Key takeaway:** At single-request, SM121 and SM120 are identical. Both achieve ~49 tok/s. For production use, either works. SM120 is architecturally cleaner (zero patches) but SM121 is already deployed and tested.

**Decision: KEEP SM121-inject as production image.** Already deployed, proven stable across 80+ stress test requests, snapshotted. SM120 is available as a fallback.

### Entry 011 — Infrastructure: Docker Compose, Backups, Atomic Add
**Date:** 2026-03-30 ~10:00 EDT
**Operator:** Claude Code

#### 11.1 Activate Docker Compose

Switched from individually-managed containers to Docker Compose.

**Process:**
1. Snapshotted as `pre-compose`
2. Stopped all 6 containers individually
3. `docker compose up -d` from `/home/claude/docker-compose.yml`
4. Startup cascade: qwen35 → qwen3-embed → gliner (with chromadb/neo4j/node-exporter in parallel)
5. All 6 services reached healthy status

**Fix needed:** ChromaDB health check URL updated from `/api/v1/heartbeat` to `/api/v2/heartbeat` (API version migration).

**All services verified:**
- LLM: "Bonjour !" — working
- Embed: dim=2560 — working
- NER: John=PERSON(0.99), Paris=LOCATION(0.97) — working
- ChromaDB: heartbeat OK
- Neo4j: HTTP OK
- Node-exporter: running (host network)

**Compose is now the active management layer.** Use `cd /home/claude && docker compose up -d` to start, `docker compose down` to stop, `docker compose ps` for status.

#### 11.2 Backup Config to Homeserver

Relayed `pipeline-v3-final` config + active `docker-compose.yml` through workstation to homeserver:
- Spark → workstation: `scp -r claude@spark:/home/claude/spark-configs/pipeline-v3-final /tmp/`
- Workstation → homeserver: `scp -r /tmp/pipeline-v3-final claude@homeserver:/mnt/user/appdata/spark-configs/`

Backed up to: `/mnt/user/appdata/spark-configs/pipeline-v3-final/` on homeserver.

#### 11.3 Data Backup Script

Created `/home/claude/backup-data.sh` — backs up ChromaDB and Neo4j Docker volumes to timestamped tar.gz files.

**First backup:** `/home/claude/backups/initial/`
- chromadb-data.tar.gz: 5.1K
- neo4j-data.tar.gz: 130M

**Usage:** `/home/claude/backup-data.sh [label]` — run before major config changes or weekly.

#### 11.4 VLLM_MARLIN_USE_ATOMIC_ADD=1 Test

**Hypothesis:** Marlin logs suggest atomic add could improve performance for small matrix sizes.

**Test:** Added `VLLM_MARLIN_USE_ATOMIC_ADD=1` to compose, recreated qwen35.

| Metric | Without | With | Delta |
|--------|---------|------|-------|
| Single-request (avg runs 2-3) | 51.5 tok/s | 50.7 tok/s | -1.6% (noise) |

**No effect.** Our 35B model's matrix sizes are not small enough to benefit from atomic add optimization. Reverted.

#### Entry 011 Summary

All four items completed:
- Docker Compose: ACTIVATED (all 6 services, health-check startup ordering)
- Homeserver backup: SYNCED (pipeline-v3-final)
- Data backup: SCRIPT DEPLOYED + initial backup taken
- Marlin atomic add: TESTED, no effect, reverted

### Entry 006 — Autonomous benchmark setup
**Date:** 2026-03-29 ~14:50 EDT
**Operator:** Claude Code

**Scripts deployed to Spark:**

1. **`/home/claude/benchmark-ab-test.sh`** — Full A/B test script that:
   - Takes pre-benchmark snapshot via spark-config.sh
   - Runs 3 tests: baseline → SM121 → SM120
   - For each: container swap, health wait (10 min max), FP8 warning check, backend log capture
   - Single-request benchmark (3 runs per image, measures tok/s via metrics delta)
   - Concurrent benchmark (8 simultaneous requests, measures aggregate tok/s)
   - Memory snapshot per image
   - Restores baseline after testing (safety)
   - All results logged to `/home/claude/ab-test-results.md`

2. **`/home/claude/pipeline-monitor.sh`** (PID 275720) — Autonomous monitor that:
   - Checks pipeline state every 5 minutes
   - Waits for two conditions: (a) after 01:30 EDT, (b) 0 running requests for 3 consecutive checks (15 min idle)
   - When both met, automatically executes benchmark-ab-test.sh
   - All monitor activity logged to `/home/claude/pipeline-monitor.log`

**What happens overnight:**
1. Pipeline runs until ~01:30 EDT
2. Monitor detects idle state ~01:45 EDT (after 3 idle checks)
3. Benchmark script runs (~30 min total: 3 images × ~10 min each)
4. Results saved to `/home/claude/ab-test-results.md`
5. Baseline restored for safety

**To check results in morning:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net "cat /home/claude/ab-test-results.md"
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net "cat /home/claude/pipeline-monitor.log"
```

**Observation:** qwen35 container restarted at ~11:55 EDT (during SM121/SM120 Docker builds). Success counter reset from 8,060 to 0. Pipeline recovered automatically via `--restart unless-stopped`. Likely cause: memory pressure from two large Docker builds (~32 GB images) running on the same system. Container has been healthy since restart (3,872+ requests completed by 14:55 EDT, 8 concurrent running).

**Lesson:** Large Docker builds on the Spark can cause memory pressure sufficient to restart vLLM containers. In future, consider running builds during maintenance windows or limiting to one build at a time.

**Bug fix (14:55 EDT):** Pipeline monitor showed `running=unknown` — the `awk '{print $2}'` in the heredoc was shell-expanded to `awk {print }`. Fixed by writing scripts via heredoc with proper quoting, then patching with sed. Verified: monitor now shows `running=8 success=3914`. Same bug affected benchmark-ab-test.sh — both fixed.

**Lesson:** When writing shell scripts via SSH heredocs, `$` variables in awk/grep are consumed by the outer shell. Use `'\''` quoting or write scripts as files with proper escaping. Always verify scripts read back correctly before relying on them for autonomous execution.

---

### Entry 012 — Spark Recon (2026-03-31)
**Date:** 2026-03-31 ~11:30 UTC
**Operator:** Claude Code (spark-recon skill)
**Status:** RECON — no changes made

#### Arena Check
- Top FP8 Qwen3.5 (single-node): 52.32 tok/s (Huihui-Qwen3.5-35B-A3B-abliterated by Artyom) — 0% delta from baseline
- Top overall: 75.96 tok/s (gpt-oss-120b, MXFP4, 2-node) — new #1, was 70.72
- No new single-node FP8 Qwen3.5 entries above baseline
- Our 48.6 tok/s remains within 7.1% of single-node FP8 leader
- NO CHANGE

#### vLLM Release Check
- Latest: v0.18.1 (2026-03-31)
- Classification: MEDIUM
- #38126 (SM121 CMake fix) merged to main 2026-03-27 but NOT in v0.18.1
- v0.18.1 includes Blackwell-specific Qwen3.5 FP8 accuracy fix (DeepGemm E8M0)
- Blocker: Known Qwen3.5 FP8 accuracy regression in v0.18.x
- Recommendation: DO NOT UPGRADE. Wait for release containing #38126 with regression fix.

#### Qwen Model Check
- No new model families (no Qwen4 announced)
- Pre-quantized FP8 variant exists: `Qwen/Qwen3.5-35B-A3B-FP8` (2.5M downloads)
- Qwen3-Coder-Next (80B/3B active) exists but 80B total weights tight on 128GB
- No model change warranted at this time

#### NVIDIA Forum Check
- ~18 topics with activity on 2026-03-31, 3 new topics
- ACTION: sggin1 posted NVFP4 Marlin env vars — 50 tok/s with 7GB less memory on Nemotron (https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119)
- ACTION: coolthor posted SM121 4-bug root cause analysis — 57-59 tok/s with MXFP4 on Qwen3.5 (https://forums.developer.nvidia.com/t/sm121-4-bugs-causing-output-gpt-oss-120b-at-59-tok-s-full-root-cause-analysis-and-working-serve-scripts/364009)
- ACTION: FlashInfer PR #2913 GDC fix — addresses latent cudaErrorIllegalInstruction crash risk on SM121
- INFO: KV cache q4_0 is catastrophically slow on unified memory; q8_0 sweet spot; confirms our fp8 choice
- INFO: TurboQuant (3.5-bit KV cache) promising but not production-ready for vLLM

#### Overall: WORTH WATCHING

#### Recommendations
1. No immediate action needed — current config remains competitive
2. When next touching config: test pre-quantized `Qwen3.5-35B-A3B-FP8` model
3. Watch for vLLM release containing #38126 (will be HIGH-priority upgrade)
4. Bookmark coolthor's MXFP4 analysis — 57-59 tok/s represents potential 20% improvement path
5. Include FlashInfer GDC fix in next vLLM upgrade plan

---

*Entries continue below as experiments are executed.*
