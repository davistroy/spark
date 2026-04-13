# Implementation Plan: DGX Spark Performance Improvements

**Created:** 2026-04-13
**Updated:** 2026-04-13 (Phase 1 completed — all flag optimizations failed on cu130)
**Source:** Spark Recon Entry 025 (2026-04-13) + Ultra-Plan investigation
**Baseline:** 48.5 tok/s single-request (corrected from 53.5 — see LAB_NOTEBOOK Entry 027), vLLM v0.19.0, cu130, gpu-memory-utilization 0.65
**Target:** ~75 tok/s single-request, improved concurrent throughput
**System:** DGX Spark (GB10, SM121, 128 GB unified memory)
**Status:** Phase 1 FAILED — MTP degrades performance on cu130, FLASHINFER/prefix caching have zero effect. **Recommend skipping to Phase 3 (cu132 build).**

---

## Overview

Community benchmarks (joshua.dale.warner optimization thread, Spark Arena entries) show 70-81 tok/s on identical hardware with FP8+MTP=2 configuration. Our 53.5 tok/s baseline uses none of the three key optimizations: MTP speculative decoding, explicit FLASHINFER backend, or prefix caching. This plan addresses the gap in three phases, ordered by impact and risk.

## Risk Assessment

| Phase | Risk Level | Rollback Strategy |
|-------|-----------|-------------------|
| Phase 1 (Flag Optimizations) | Low | Remove flags, restart with original docker run command |
| Phase 2 (Memory Tuning) | Medium | Revert gpu-memory-utilization to 0.65, restart |
| Phase 3 (Image Upgrade) | Medium | Test on separate port first. Keep current image on disk. |

## Constraints

- 3-model config (qwen35 + qwen3-embed + gliner) constrains GPU memory budget
- Container restart = ~90-150s LLM downtime (warm Triton cache)
- No passwordless sudo on davistroy account; use `claude` user via SSH
- All changes follow CLAUDE.md pre-flight checks: read documented config, diff before execute, verify startup

---

## Phase 1: Flag Optimizations (Items 1-3)

**Goal:** Enable MTP=2 speculative decoding, explicit FLASHINFER attention, and prefix caching in a single container restart.
**Expected impact:** 53.5 -> ~75 tok/s single-request (+40%)
**Downtime:** ~90-150s (one restart, warm Triton cache)
**Dependency:** None

### Pre-flight

- [ ] Confirm no active pipeline or inference workload on qwen35
- [ ] Record current container config: `docker inspect qwen35 --format '{{json .Config.Cmd}}'`
- [ ] Verify Triton cache is intact: `ls /home/claude/.cache/triton/`

### Work Item 1.1: Stop current qwen35 container

```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net \
  "docker stop qwen35 && docker rm qwen35"
```

**Acceptance criteria:** Container removed cleanly. qwen3-embed and gliner remain running.

### Work Item 1.2: Start qwen35 with optimized flags

```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net \
  "docker run -d \
    --name qwen35 \
    --restart unless-stopped \
    --gpus all \
    --ipc host \
    --shm-size 64gb \
    -p 8000:8000 \
    -e VLLM_FLASHINFER_MOE_BACKEND=latency \
    -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
    -v /home/claude/.cache/triton:/root/.triton \
    vllm/vllm-openai:v0.19.0-aarch64-cu130 \
    Qwen/Qwen3.5-35B-A3B \
      --served-model-name qwen3.5-35b \
      --port 8000 \
      --host 0.0.0.0 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.65 \
      --quantization fp8 \
      --kv-cache-dtype fp8 \
      --reasoning-parser qwen3 \
      --language-model-only \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --attention-backend FLASHINFER \
      --enable-prefix-caching \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"
```

**Changes from current config (diff):**
```diff
+ --attention-backend FLASHINFER
+ --enable-prefix-caching
+ --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

All other flags, volume mounts, and env vars are identical to the documented known-working command.

**Acceptance criteria:**
- Container starts without errors
- `docker logs qwen35` shows: MTP/speculative decode enabled, FLASHINFER attention backend, prefix caching enabled
- GPU memory allocation grows beyond 3 GB within 60s
- No CUDA OOM errors

### Work Item 1.3: Verify container health

```bash
# Wait for health endpoint (poll every 10s, max 180s)
# Check GPU memory is allocating
nvidia-smi
# Health check
curl -s http://localhost:8000/health
# Verify model loaded
curl -s http://localhost:8000/v1/models
```

**Acceptance criteria:**
- `/health` returns 200
- `/v1/models` lists `qwen3.5-35b`
- GPU memory for qwen35 shows expected allocation (~79-82 GiB)

### Work Item 1.4: Verify startup logs for new features

```bash
docker logs qwen35 2>&1 | grep -iE "(speculative|mtp|prefix.cach|attention.*backend|flashinfer)"
```

**Acceptance criteria:**
- Logs confirm MTP speculative decoding with 2 tokens
- Logs confirm FLASHINFER attention backend
- Logs confirm prefix caching enabled
- No "unsupported" or "falling back" warnings for any of these features

### Work Item 1.5: Benchmark single-request throughput

Run the standard benchmark from SPARK_BASELINE.md methodology:

```bash
# Single request, ~128 output tokens
curl -s -w '\n%{time_total}' http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Write a detailed paragraph about the history of sailing, covering at least 5 major developments from ancient times to modern day."}],"max_tokens":256,"temperature":0.7}'
```

Run 3 times, take the median. Calculate tok/s from response token count / generation time.

**Acceptance criteria:**
- Single-request tok/s >= 65 (minimum 20% improvement from 53.5 baseline)
- Target: 70-81 tok/s based on community results
- TTFT (time to first token) not significantly regressed (< 2x current)

### Work Item 1.6: Verify tool calling still works

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
    "max_tokens": 256
  }'
```

**Acceptance criteria:**
- Response contains a tool_call with function name `get_weather` and location argument
- No errors or malformed tool call JSON

### Work Item 1.7: Run concurrent benchmark

```bash
# c4 aggregate throughput test
# Run 4 simultaneous requests and measure aggregate tok/s
```

**Acceptance criteria:**
- c4 aggregate >= 140 tok/s (current baseline: 140.4, should not regress)
- No request failures or timeouts

### Work Item 1.8: Update documentation

- [ ] Update `SPARK_CONFIG.md` Section 6.1 with new docker run command
- [ ] Update `SPARK_BASELINE.md` Current Config with new measured tok/s
- [ ] Append LAB_NOTEBOOK.md entry with full benchmark results
- [ ] Update CLAUDE.md if any new operational rules discovered

**Acceptance criteria:** All docs reflect the new running config and measured performance.

### Phase 1 Gate

**Proceed to Phase 2 only if ALL of the following are true:**
- Single-request tok/s >= 65
- c4 aggregate not regressed (>= 130 tok/s)
- Tool calling functional
- Swap usage < 100 MB
- No CUDA errors in logs
- System stable for >= 30 minutes under idle + occasional queries

**If MTP fails or degrades quality:** Remove `--speculative-config` only, keep FLASHINFER and prefix caching. Re-benchmark. These two flags alone may still provide modest gains.

---

## Phase 2: Memory Budget Tuning (Item 4)

**Goal:** Increase gpu-memory-utilization from 0.65 to 0.75 to expand KV cache budget
**Expected impact:** More concurrent request capacity, larger effective batch sizes
**Downtime:** ~90-150s (one restart)
**Dependency:** Phase 1 complete and stable

### Pre-flight

- [ ] Phase 1 gate criteria all met
- [ ] Record Phase 1 GPU memory usage: `nvidia-smi` (note qwen35 allocation)
- [ ] Record Phase 1 swap usage: `free -h`
- [ ] Calculate memory budget:
  - qwen35 at 0.75: 0.75 x 121.6 GiB = 91.2 GiB
  - qwen3-embed at 0.10: ~12.2 GiB
  - gliner: ~2 GiB
  - Total: ~105.4 GiB
  - Free: ~16.2 GiB (safe margin)

### Work Item 2.1: Stop and restart qwen35 with 0.75 utilization

Same docker run command as Phase 1, but change:
```diff
- --gpu-memory-utilization 0.65
+ --gpu-memory-utilization 0.75
```

Follow the same stop/start/verify sequence as Work Items 1.1-1.3.

**Acceptance criteria:**
- Container starts without OOM
- GPU memory for qwen35 shows ~91 GiB allocation
- Total system GPU < 106 GiB
- Free system memory > 14 GiB

### Work Item 2.2: Memory pressure test

```bash
# Monitor swap and memory for 10 minutes under load
watch -n 10 'free -h && echo "---" && swapon --show'

# Concurrent load test: 8 simultaneous requests
# Verify no swap growth during sustained load
```

**Acceptance criteria:**
- Swap usage stays < 200 MB during sustained c8 load
- Available system memory stays > 10 GiB
- No OOM kills in `dmesg`
- Temperature stays < 70C

### Work Item 2.3: Benchmark at new utilization

Run full benchmark suite:
- c1 single-request tok/s
- c4 aggregate tok/s
- c8 aggregate tok/s
- c16 aggregate tok/s (if memory allows)

**Acceptance criteria:**
- c1 >= Phase 1 result (MTP + larger KV cache should not regress)
- c4 >= 140 tok/s
- c8 >= 210 tok/s
- No KV cache eviction warnings in logs

### Work Item 2.4: Update documentation

Same as Work Item 1.8 — update all config docs with new gpu-memory-utilization value and benchmark results.

### Phase 2 Gate

**Proceed to Phase 3 only if:**
- All benchmarks meet or exceed Phase 1 results
- Swap < 200 MB under c8 sustained load
- System stable for >= 1 hour
- No temperature warnings

**If 0.75 causes memory pressure:** Try 0.70 as intermediate step. If even 0.70 is problematic, keep 0.65 and document the ceiling.

---

## Phase 3: Image Upgrade Evaluation (Item 5)

**Goal:** Test eugr's cu132 build (vLLM 0.19.1rc1.dev219 + FlashInfer 0.6.7) for additional SM121 kernel improvements
**Expected impact:** Unknown — potentially 5-15% additional throughput from better codegen
**Downtime:** None during testing (separate port). ~90-150s for production swap if validated.
**Dependency:** Phase 2 complete and stable (establishes optimized cu130 baseline)

### Pre-flight

- [ ] Phase 2 gate criteria all met
- [ ] Record Phase 2 benchmark numbers as cu130 optimized baseline
- [ ] Verify disk space for new image: `df -h /` (need ~25 GB)
- [ ] Check eugr/spark-vllm-docker for latest release tag

### Work Item 3.1: Build or pull cu132 image

Option A — use eugr's prebuilt wheels:
```bash
# Download wheels from eugr/spark-vllm-docker releases
# Build custom image incorporating cu132 vLLM + FlashInfer 0.6.7
```

Option B — build from spark-vllm-docker Dockerfile:
```bash
cd /home/davistroy/spark-vllm-docker
git pull
# Review Dockerfile changes
# Build with cu132 target
```

**Acceptance criteria:** Image built/pulled successfully. Size reasonable (< 30 GB).

### Work Item 3.2: Test on separate port (non-destructive)

```bash
# Start test container on port 8010 with SAME optimized flags from Phase 2
docker run -d \
  --name qwen35-cu132-test \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8010:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton-cu132:/root/.triton \
  <cu132-image-tag> \
  Qwen/Qwen3.5-35B-A3B \
    --served-model-name qwen3.5-35b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.75 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --attention-backend FLASHINFER \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**IMPORTANT:** Cannot run simultaneously with production qwen35 (GPU memory conflict). Must stop production container first, then start test container.

**Acceptance criteria:**
- Model loads successfully on cu132
- SM121 kernels selected (check startup logs for sm_121 or compute_121)
- No "unsupported architecture" or "falling back" warnings
- Health endpoint returns 200

### Work Item 3.3: Benchmark cu132 vs cu130

Run identical benchmark suite as Phase 2:
- c1, c4, c8 tok/s
- Compare directly against Phase 2 numbers (cu130 optimized)

**Acceptance criteria:**
- cu132 >= cu130 results (no regression)
- If cu132 > cu130 by >= 5%, recommend production swap
- If cu132 == cu130 (+/- 3%), recommend staying on cu130 (stable release)
- If cu132 < cu130, do NOT adopt

### Work Item 3.4: Production swap (conditional)

Only if Work Item 3.3 shows >= 5% improvement:
- Stop test container
- Stop production qwen35
- Start production qwen35 with cu132 image
- Verify full benchmark suite
- Update documentation

### Work Item 3.5: Cleanup

- Remove test container: `docker rm qwen35-cu132-test`
- Keep cu132 image on disk as rollback/future option if not adopted
- Document results in LAB_NOTEBOOK.md regardless of outcome

---

## Phase 4: Quality Baseline (Item 6)

**Goal:** Bookmark DanTup/spark-evals for future quality validation
**Expected impact:** Quality assurance framework for quant format decisions
**Dependency:** None (can happen anytime)

### Work Item 4.1: Review and bookmark spark-evals

- [ ] Review https://github.com/DanTup/spark-evals for methodology
- [ ] Add to SPARK_BASELINE.md Watch Items
- [ ] Consider running evals after Phase 1-3 changes stabilize

**Acceptance criteria:** Reference documented, methodology understood, decision made on whether to run evals.

---

## Implementation Sequence

```
Phase 1 (Flag Optimizations) ──── gate ──── Phase 2 (Memory Tuning) ──── gate ──── Phase 3 (Image Upgrade)
                                                                                          │
Phase 4 (Quality Baseline) ─────────────────────────────────── independent ────────────────┘
```

**Estimated timeline:**
- Phase 1: ~30 minutes (restart + benchmarks)
- Phase 2: ~30 minutes (restart + memory stress test + benchmarks)
- Phase 3: ~2-3 hours (image build/pull + testing)
- Phase 4: ~15 minutes (review only)

## Scope Boundaries

**In scope:**
- qwen35 container optimization (flags, memory, image)
- Benchmarking and verification
- Documentation updates

**Out of scope (and why):**
- qwen3-embed optimization — already efficient at 0.10 utilization, no community signals for improvement
- gliner optimization — PyTorch-based, minimal GPU footprint, not a performance bottleneck
- Hybrid INT4+FP8 checkpoint — requires custom checkpoint build, significantly more complex, should be a separate plan if MTP+FP8 is insufficient
- DFlash speculative decoding — experimental, real-world performance lower than MTP, should be a separate investigation
- Driver update (580.142 -> newer) — previous driver update (590.48.01) caused UMA memory leak, high risk
- Multi-node configuration — we have one Spark

**Recommended follow-up:**
- If Phase 1-3 achieve ~75+ tok/s, investigate hybrid INT4+FP8 for 100+ tok/s tier
- Run DanTup/spark-evals to validate FP8 quality hasn't degraded with MTP
- Monitor vLLM v0.20.0 release for additional SM121 improvements
