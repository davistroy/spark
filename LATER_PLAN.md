# LATER_PLAN: Future DGX Spark Optimizations

**Date:** 2026-03-28
**When to revisit:** Monthly, or when vLLM releases a new stable/nightly build
**Risk level:** Medium. These changes affect inference behavior and require testing under load.

---

## Overview

These optimizations address performance ceilings that are currently blocked by vLLM/GB10 compatibility issues. Each one should be tested in isolation, ideally during a low-traffic window, with a clear rollback path.

### Execution Order

| Order | Step | Rationale |
|-------|------|-----------|
| 1st | Step 0: Review 24h usage data | Gate — determines if parameters need adjusting before anything else |
| 2nd | Step 3: Update vLLM nightly | Do this BEFORE Steps 1-2 — a newer build may fix enforce-eager and async-scheduling issues, making those steps succeed instead of needing rollback |
| 3rd | Step 1: Remove `--enforce-eager` | Test on the new nightly (if updated) |
| 4th | Step 2: Remove `--no-async-scheduling` | Test on the new nightly (if updated) |
| 5th | Step 4: Embedding memory reduction | Quick test, independent of LLM |
| 6th | Step 5: DCGM exporter | Additive, no risk |
| 7th | Step 8: ChromaDB + Neo4j | Additive, RAM-only |
| 8th | Step 6: Ethernet static IP | Already configured — just plug in cable |
| 9th | Step 7: Ethernet optimization | Last — avoids network disruption during container work |

---

## Step 0: Review 24-Hour Usage Data Before Making Changes

### Purpose

Before touching anything, pull Prometheus metrics for the last 24 hours to understand actual usage patterns. The NOW_PLAN recommendations were based on a 3-hour snapshot — a full day of data may reveal different patterns that change priorities or parameters.

### What to Check

#### 1. KV Cache Utilization — Is 0.65 the Right Setting?

```promql
# Peak KV cache usage over 24h
max_over_time(vllm:kv_cache_usage_perc{model_name="qwen3.5-35b"}[24h])

# 95th percentile KV cache usage
quantile_over_time(0.95, vllm:kv_cache_usage_perc{model_name="qwen3.5-35b"}[24h])
```

**Decision matrix:**
| Peak KV Cache Usage | Action |
|---------------------|--------|
| < 5% | Could reduce to 0.60 for more host RAM headroom |
| 5-30% | 0.65 is right, leave it |
| 30-70% | 0.65 is right, monitor for preemptions |
| > 70% or any preemptions | Increase back to 0.72 |

#### 2. Preemptions — Any KV Cache Pressure?

```promql
# Total preemptions (should be 0)
vllm:num_preemptions_total{model_name="qwen3.5-35b"}
```

If preemptions > 0, the KV cache is too small. Increase `--gpu-memory-utilization` back toward 0.72 before attempting any other changes.

#### 3. Concurrency — How Many Simultaneous Requests?

```promql
# Peak concurrent requests
max_over_time(vllm:num_requests_running{model_name="qwen3.5-35b"}[24h])

# Average concurrent requests
avg_over_time(vllm:num_requests_running{model_name="qwen3.5-35b"}[24h])

# Were there ever queued (waiting) requests?
max_over_time(vllm:num_requests_waiting{model_name="qwen3.5-35b"}[24h])
```

**Why it matters for LATER_PLAN:**
- If peak concurrency is consistently < 3, removing `--no-async-scheduling` (Step 2) has low value — async scheduling mainly helps under concurrency.
- If waiting requests > 0, throughput improvements (Steps 1-2) are more urgent.

#### 4. Latency Distribution — Where's the Pain?

```promql
# TTFT p50 and p95 over 24h
histogram_quantile(0.5, rate(vllm:time_to_first_token_seconds_bucket{model_name="qwen3.5-35b"}[24h]))
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket{model_name="qwen3.5-35b"}[24h]))

# E2E latency p50 and p95
histogram_quantile(0.5, rate(vllm:e2e_request_latency_seconds_bucket{model_name="qwen3.5-35b"}[24h]))
histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket{model_name="qwen3.5-35b"}[24h]))
```

**Why it matters:** If TTFT is the bottleneck (>1s p95), prefill optimization matters more than decode speed. If decode is the bottleneck, removing `--enforce-eager` (Step 1) is the priority.

#### 5. Memory — Is the System Stable?

```promql
# Swap usage trend
(node_memory_SwapTotal_bytes{job="spark-node"} - node_memory_SwapFree_bytes{job="spark-node"}) / 1073741824

# Available RAM trend
node_memory_MemAvailable_bytes{job="spark-node"} / 1073741824
```

**Decision:** If swap is consistently > 1 GB or available RAM < 5 GB, do NOT proceed with Steps 1-2 (which may increase memory usage). Address memory first.

#### 6. GPU Thermal — Any Throttling Risk?

```promql
# Peak GPU temp
max_over_time(gpu_temperature_celsius[24h])

# Average GPU temp
avg_over_time(gpu_temperature_celsius[24h])
```

**Decision:** If peak temp > 75°C, removing `--enforce-eager` (which increases GPU utilization) may push into thermal throttling. Address cooling first.

#### 7. Embedding Model Usage — Worth Keeping Loaded?

```promql
# Total embedding requests in 24h
increase(vllm:request_success_total{model_name="qwen3-embedding-4b"}[24h])

# Peak concurrent embedding requests
max_over_time(vllm:num_requests_running{model_name="qwen3-embedding-4b"}[24h])
```

**Decision:** If < 10 requests in 24h, consider whether the 15.8 GB GPU allocation is justified.

### Procedure

```bash
# Query Prometheus directly for all metrics above
# (Run from any machine that can reach homeserver:9090)

# KV cache peak
curl -s 'http://homeserver.k4jda.net:9090/api/v1/query?query=max_over_time(vllm:kv_cache_usage_perc{model_name="qwen3.5-35b"}[24h])'

# Preemptions
curl -s 'http://homeserver.k4jda.net:9090/api/v1/query?query=vllm:num_preemptions_total{model_name="qwen3.5-35b"}'

# Peak concurrency
curl -s 'http://homeserver.k4jda.net:9090/api/v1/query?query=max_over_time(vllm:num_requests_running{model_name="qwen3.5-35b"}[24h])'

# Waiting requests
curl -s 'http://homeserver.k4jda.net:9090/api/v1/query?query=max_over_time(vllm:num_requests_waiting{model_name="qwen3.5-35b"}[24h])'

# Swap
curl -s 'http://homeserver.k4jda.net:9090/api/v1/query?query=(node_memory_SwapTotal_bytes{job="spark-node"}-node_memory_SwapFree_bytes{job="spark-node"})/1073741824'

# GPU peak temp
curl -s 'http://homeserver.k4jda.net:9090/api/v1/query?query=max_over_time(gpu_temperature_celsius[24h])'
```

### Output

Document results and any adjustments to Steps 1-7 before proceeding. If any metric shows a red flag (preemptions > 0, swap > 1 GB, GPU > 75°C), address that FIRST before optimization experiments.

---

## Step 1: Remove `--enforce-eager` (Potential 10-30% Throughput Gain)

### Background

`--enforce-eager` disables CUDA graph capture in vLLM. CUDA graphs pre-record GPU kernel launch sequences and replay them, eliminating CPU-side overhead per decode step. For small batch sizes like yours (avg ~9 tokens/iteration), CUDA graph capture is particularly impactful because the CPU overhead is a larger fraction of total step time.

This flag was added because Triton PTX compilation fails on ARM64 GB10 (SM 12.1). Triton generates PTX code that the CUDA driver can't JIT-compile for this architecture.

### Expected Impact

- Decode throughput: ~23 tok/s -> ~28-33 tok/s (estimated)
- TTFT: unchanged or slightly worse (graph capture adds startup cost)
- Memory: slightly higher (CUDA graphs consume GPU memory for recorded sequences)

### Test Procedure

```bash
# 1. Stop current LLM container
docker stop qwen35 && docker rm qwen35

# 2. Start WITHOUT --enforce-eager
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
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
    --no-async-scheduling

# 3. Watch startup logs for Triton errors
docker logs -f qwen35 2>&1 | head -200
# LOOK FOR: "Triton", "PTX", "compilation error", "CUDA error"
# If you see these -> rollback immediately

# 4. Wait for health
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done

# 5. Run benchmark comparison
# Simple throughput test — measure time for a known generation:
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role":"user","content":"Write a detailed 500-word essay about the history of sailing."}],
    "max_tokens": 600,
    "stream": false
  }' > /tmp/benchmark_result.json

# 6. Check metrics for tok/s improvement
curl -s http://localhost:8000/metrics | grep "vllm:generation_tokens_total"
# Compare rate over 60 seconds vs baseline
```

### What to Watch For

| Signal | Meaning | Action |
|--------|---------|--------|
| Triton PTX errors in logs | SM 12.1 still unsupported | Rollback — add `--enforce-eager` back |
| Container starts but inference hangs | CUDA graph capture failed silently | Rollback |
| Container starts, inference works, tok/s improved | Success | Keep the change |
| Container starts, inference works, tok/s unchanged | CUDA graphs not being used (model doesn't support them) | Keep or rollback — no harm either way |
| OOM during graph capture | CUDA graphs need more memory than available | Try reducing `--gpu-memory-utilization` to 0.60 to make room, or rollback |

### Rollback

```bash
docker stop qwen35 && docker rm qwen35
# Re-run with --enforce-eager added back (use NOW_PLAN Step 1 command)
```

---

## Step 2: Remove `--no-async-scheduling` (Better Prefill/Decode Overlap)

### Background

`--no-async-scheduling` was added to work around a V1 engine crash: `NoneType has no sampled_token_ids`. The async scheduler overlaps prefill computation for new requests with ongoing decode for existing requests. Without it, the engine processes strictly sequentially — each step must fully complete before the next begins.

For your workload (3 concurrent requests at peak), async scheduling would allow new request prefill to happen while existing requests decode, reducing TTFT for queued requests.

### Expected Impact

- TTFT under concurrency: 20-40% improvement (prefill starts sooner)
- Decode throughput: slight improvement (better GPU utilization during mixed workloads)
- Stability: the risk — the crash that prompted this flag

### Test Procedure

**Only attempt this AFTER Step 1 succeeds or is determined to be independent.**

```bash
# 1. Stop current LLM container
docker stop qwen35 && docker rm qwen35

# 2. Start WITHOUT --no-async-scheduling
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
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
    --enforce-eager

# 3. Wait for health
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done

# 4. Stress test with concurrent requests (this is what triggers the crash)
# Run 3-5 simultaneous requests:
for i in $(seq 1 5); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3.5-35b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write paragraph $i of a story about a sailor.\"}],\"max_tokens\":200}" \
    > /tmp/concurrent_test_$i.json &
done
wait
echo "All requests completed"

# 5. Check container is still alive
curl -sf http://localhost:8000/health && echo "Still healthy"

# 6. Check logs for the crash signature
docker logs qwen35 2>&1 | tail -50 | grep -i "sampled_token_ids\|NoneType\|error\|crash"

# 7. Run the concurrent test 10 more times to build confidence
for round in $(seq 1 10); do
  echo "Round $round..."
  for i in $(seq 1 5); do
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"qwen3.5-35b\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me fact number $i about quantum physics in round $round.\"}],\"max_tokens\":150}" \
      > /dev/null &
  done
  wait
  curl -sf http://localhost:8000/health || { echo "CRASH DETECTED at round $round"; break; }
done
echo "Stress test complete"
```

### What to Watch For

| Signal | Meaning | Action |
|--------|---------|--------|
| `NoneType has no sampled_token_ids` in logs | Known V1 crash still present | Rollback — add `--no-async-scheduling` back |
| Container crashes/restarts during concurrent requests | Async scheduler instability | Rollback |
| All 50+ concurrent requests complete successfully | Bug appears to be fixed | Keep change, monitor for 24h |

### Rollback

```bash
docker stop qwen35 && docker rm qwen35
# Re-run with --no-async-scheduling added back
```

---

## Step 3: Update vLLM Build (Critical — SM121 Kernel Fix)

### Background — This Is Not a Routine Update

**Research finding (2026-03-28):** PR vllm-project/vllm#38126 (merged 2026-03-28) fixes CMake arch guards that **silently skipped ALL SM121-family native kernels** (NVFP4, scaled_mm, MLA, moe_data). Every vLLM build before this merge — including our v0.17.0rc1 from March 6 — has been running entirely on fallback codepaths.

Community benchmarks show **50 tok/s** on the same hardware/model where we get 23 tok/s. The gap is:
1. Missing native SM121 kernels (this CMake bug)
2. `--enforce-eager` disabling CUDA graphs
3. Suboptimal FP8 MoE backend selection

**DO NOT upgrade to v0.18.0** — confirmed Qwen3.5 regression (#37749, containers silently crash) and FP8 KV cache accuracy issues on Blackwell (#37618). **Target: v0.17.1 or build from main post-#38126.**

### Additional Optimizations to Test With the New Build

| Change | Env Var / Flag | What It Does |
|--------|---------------|--------------|
| Force Marlin FP8 backend | `VLLM_TEST_FORCE_FP8_MARLIN=1` | Confirmed working on SM121, avoids TRITON JIT hang and CUTLASS crash |
| Enable prefix caching | `--enable-prefix-caching` | Cache shared system prompts across requests — critical for pipeline workloads |
| Sleep mode for embed | `--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1` on qwen3-embed | Frees ~16 GB GPU when embed is idle, wake time 0.1-6s |

### Procedure

```bash
# 1. Tag current working image as known-good BEFORE pulling anything
docker tag vllm/vllm-openai:cu130-nightly vllm/vllm-openai:cu130-known-good-20260306

# 2. Pull latest nightly (should now include #38126 fix)
docker pull vllm/vllm-openai:cu130-nightly

# 3. Check if it actually changed
docker images vllm/vllm-openai --format "{{.Tag}} {{.ID}} {{.CreatedAt}}" | head -5
# If the nightly ID matches the known-good, no update — stop here

# 4. IMPORTANT: Check the vLLM version in the new image BEFORE deploying
docker run --rm vllm/vllm-openai:cu130-nightly python3 -c "import vllm; print(vllm.__version__)"
# If it says v0.18.x — DO NOT USE (Qwen3.5 regression). Roll back:
# docker tag vllm/vllm-openai:cu130-known-good-20260306 vllm/vllm-openai:cu130-nightly

# 5. If v0.17.1 or v0.17.x — proceed. Stop containers (maintain startup order on restart)
docker stop qwen35 qwen3-embed
docker rm qwen35 qwen3-embed

# 6. Start qwen35 with new image + Marlin FP8 + prefix caching
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton:/root/.triton \
  vllm/vllm-openai:cu130-nightly \
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
    --enforce-eager \
    --no-async-scheduling \
    --enable-prefix-caching

# 7. Monitor startup (watch for Marlin backend selection and model loading)
for i in $(seq 1 40); do
  HEALTH=$(curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "HEALTHY" || echo "loading")
  GPU_MEM=$(nvidia-smi | grep "VLLM::EngineCore" | tail -1 | grep -o "[0-9]*MiB" | head -1)
  LOG_LINES=$(docker logs qwen35 2>&1 | wc -l)
  LAST_LOG=$(docker logs qwen35 2>&1 | tail -1 | cut -c1-120)
  echo "[$i] $HEALTH | GPU: $GPU_MEM | logs: $LOG_LINES | $LAST_LOG"
  if [ "$HEALTH" = "HEALTHY" ]; then echo "=== LOADED ==="; break; fi
  sleep 5
done

# 8. Verify Marlin backend was selected
docker logs qwen35 2>&1 | grep -i "MoE backend"
# Expected: "Using MARLIN Fp8 MoE backend"

# 9. Verify prefix caching enabled
docker logs qwen35 2>&1 | grep -i "prefix_caching"

# 10. Benchmark — compare against 23 tok/s baseline
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Write a detailed 500-word essay about the history of sailing."}],"max_tokens":600,"stream":false}' > /tmp/benchmark.json

# 11. Start qwen3-embed with sleep mode
docker run -d \
  --name qwen3-embed \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8001:8001 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3-Embedding-4B \
    --served-model-name qwen3-embedding-4b \
    --runner pooling \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.13 \
    --enforce-eager \
    --enable-sleep-mode

# 12. Wait for embed health
until curl -sf http://localhost:8001/health > /dev/null 2>&1; do sleep 5; done
echo "Embed healthy"
```

### Rollback

```bash
docker stop qwen35 qwen3-embed && docker rm qwen35 qwen3-embed
# Restore from known-good image:
docker tag vllm/vllm-openai:cu130-known-good-20260306 vllm/vllm-openai:cu130-nightly
# Then re-run with original pipeline-v1 config (use spark-config.sh apply pipeline-v1)
```

### What Changed vs Current Config

| Parameter | Current (pipeline-v1) | New (Step 3) |
|-----------|----------------------|--------------|
| Image | cu130-nightly (March 6, v0.17.0rc1) | cu130-nightly (latest, v0.17.1+) |
| FP8 MoE backend | TRITON (auto-selected, 15-30min JIT) | Marlin (forced via env var, no JIT) |
| Prefix caching | Disabled | Enabled |
| Embed sleep mode | Not available | Enabled (frees 16 GB when idle) |
| SM121 kernels | Silently skipped (CMake bug) | Properly compiled (#38126 fix) |

---

## Step 1b: Build Custom Triton for ARM64 → Remove --enforce-eager (Biggest Performance Lever)

### Background — Why This Matters Most

`--enforce-eager` disables CUDA graph capture, which is the **single largest performance bottleneck**. Community benchmarks on SM120 (RTX 5090) show 140 tok/s with CUDA graphs vs 17 tok/s without — an 8x difference. Even conservatively, removing enforce-eager should take us from 23 tok/s to 50+ tok/s.

The reason we need `--enforce-eager` is that **Triton's JIT compiler on ARM64 can't generate PTX code**. PTX is the intermediate GPU instruction format that NVIDIA's driver compiles to final GPU machine code. On x86_64, Triton's LLVM backend includes the NVPTX target, but the ARM64 LLVM builds shipped in the vLLM Docker image don't include it.

The fix: build Triton from source on the Spark with LLVM's NVPTX backend enabled, then inject it into the vLLM container.

### What Triton and Marlin Do (for reference)

- **Marlin**: Pre-compiled CUDA library for FP8 matrix math (MoE expert layers only). Works immediately, no JIT. The `VLLM_TEST_FORCE_FP8_MARLIN=1` env var forces its use. Gives 10-20% gain.
- **Triton**: JIT compiler that generates GPU kernels at runtime from Python. Used by vLLM for CUDA graph capture, attention kernels, and many other operations. When Triton can't compile (our ARM64 issue), vLLM falls back to eager mode — running each operation individually instead of capturing and replaying optimized sequences. This is the 2-8x performance gap.

### Procedure

This is a multi-hour build process. Run it on the Spark during a maintenance window.

```bash
# 1. Install build dependencies
sudo apt update && sudo apt install -y cmake ninja-build python3-dev

# 2. Clone Triton with NVPTX support
cd /home/claude
git clone https://github.com/triton-lang/triton.git
cd triton

# 3. Build LLVM with NVPTX backend (this is the key step)
# The default ARM64 build excludes NVPTX — we force-include it
export TRITON_BUILD_WITH_NVPTX=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# 4. Build Triton (this takes 1-2 hours on ARM64)
pip install -e python/ 2>&1 | tee /tmp/triton-build.log
# Watch for: "NVPTX" in the LLVM targets list during build

# 5. Verify the build includes PTX support
python3 -c "import triton; print(triton.__version__)"
python3 -c "
import triton
from triton.compiler import compile
print('Triton PTX backend available')
"

# 6. Package the built Triton for injection into Docker
# Create a volume with the built Triton
mkdir -p /home/claude/triton-custom
cp -r /home/claude/triton/python/triton /home/claude/triton-custom/
```

### Test: Remove --enforce-eager

After Triton is built and verified:

```bash
# 1. Snapshot current config
/home/claude/spark-config.sh snapshot pre-triton "Before custom Triton / enforce-eager removal"

# 2. Stop qwen35
docker stop qwen35 && docker rm qwen35

# 3. Start WITHOUT --enforce-eager, mounting custom Triton
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton:/root/.triton \
  -v /home/claude/triton-custom/triton:/usr/local/lib/python3.12/dist-packages/triton \
  vllm/vllm-openai:cu130-nightly \
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
    --no-async-scheduling \
    --enable-prefix-caching

# NOTE: --enforce-eager is REMOVED

# 4. Monitor — CUDA graph capture will log during warmup
for i in $(seq 1 60); do
  HEALTH=$(curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "HEALTHY" || echo "loading")
  GPU_MEM=$(nvidia-smi | grep "VLLM::EngineCore" | tail -1 | grep -o "[0-9]*MiB" | head -1)
  LAST_LOG=$(docker logs qwen35 2>&1 | tail -1 | cut -c1-120)
  echo "[$i] $HEALTH | GPU: $GPU_MEM | $LAST_LOG"
  if [ "$HEALTH" = "HEALTHY" ]; then echo "=== LOADED ==="; break; fi
  if echo "$LAST_LOG" | grep -qi "error\|PTX\|illegal"; then
    echo "=== TRITON/PTX ERROR — ROLLING BACK ==="
    docker logs qwen35 2>&1 | grep -i "error\|PTX\|triton" | tail -10
    break
  fi
  sleep 10
done

# 5. Benchmark — THE MOMENT OF TRUTH
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Write a detailed 500-word essay about the history of sailing."}],"max_tokens":600,"stream":false}' > /tmp/benchmark_no_eager.json

# Compare: if 600 tokens took ~26s with enforce-eager (23 tok/s),
# without it we'd expect ~12s (50 tok/s) or better
```

### Expected Outcome

| Scenario | tok/s | Action |
|----------|-------|--------|
| CUDA graphs work, 40-60 tok/s | Success — keep config, snapshot as `pipeline-v2-fast` |
| CUDA graphs work, 25-35 tok/s | Partial — graphs captured but limited improvement. Keep it. |
| Triton PTX errors in logs | Custom build didn't fix it — rollback to `--enforce-eager` |
| Container hangs during graph warmup | Graph capture failing — rollback |

### Rollback

```bash
docker stop qwen35 && docker rm qwen35
# Restore from pre-triton snapshot:
/home/claude/spark-config.sh apply pre-triton
```

### If the Build Fails

The Triton build is complex. If it fails:
- Check the LLVM build log for missing NVPTX target
- The community repo https://github.com/eelbaz/dgx-spark-vllm-setup may have pre-built wheels
- Monitor triton-lang/triton#2922 for official ARM64 NVPTX support
- The Marlin FP8 gain (10-20%) and other Step 3 optimizations still apply regardless
```

---

## Step 4: Evaluate Embedding Model Memory Reduction

### Background

The Qwen3-Embedding-4B model uses 15.8 GB at `--gpu-memory-utilization 0.13`. The model weights themselves are ~8 GB (4B params in BF16). The remaining ~8 GB is KV cache and overhead for the embedding pipeline.

If the embedding workload is consistently low-volume (which the dashboard shows — mostly idle with occasional bursts), you could reduce the util to 0.10 to reclaim ~4 GB.

### When to Do This

Only if you're still seeing memory pressure after NOW_PLAN Step 1 (reducing LLM from 0.72 to 0.65).

### Procedure

```bash
docker stop qwen3-embed && docker rm qwen3-embed

docker run -d \
  --name qwen3-embed \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3-Embedding-4B \
    --served-model-name qwen3-embedding-4b \
    --runner pooling \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.10 \
    --enforce-eager

# Wait for health
until curl -sf http://localhost:8001/health > /dev/null 2>&1; do sleep 5; done

# Test embedding
curl -s http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-4b","input":"test embedding"}' | python3 -m json.tool | head -5
```

### Risk

If `0.10` isn't enough for the model weights + pooling overhead, the container will fail to start. Check logs for OOM. If it fails, go back to `0.13`.

---

## Step 5: Install DCGM Exporter (Better GPU Monitoring)

### Background

The current custom GPU exporter on port 9400 can't read memory stats due to GB10's UMA architecture. NVIDIA's DCGM (Data Center GPU Manager) exporter may have better support for unified memory reporting, and exposes a much richer set of GPU metrics (SM occupancy, tensor core utilization, NVLink stats, memory bandwidth).

### When to Do This

When you have a maintenance window. This adds a new container.

### Procedure

```bash
# Check if dcgm-exporter supports GB10
docker pull nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04

docker run -d \
  --name dcgm-exporter \
  --restart unless-stopped \
  --gpus all \
  --cap-add SYS_ADMIN \
  -p 9401:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04

# Wait 30 seconds for DCGM to initialize
sleep 30

# Check if metrics are populated (especially memory)
curl -s http://localhost:9401/metrics | grep -E "DCGM_FI_DEV_FB_(TOTAL|USED|FREE)"
# If these show non-zero values, DCGM can read UMA memory — success
# If still zero, GB10 UMA isn't supported by DCGM either — stop the container
```

### Add to Prometheus

If DCGM works, add to Prometheus config on homeserver:
```yaml
  - job_name: "spark-dcgm"
    static_configs:
      - targets: ["spark.k4jda.net:9401"]
```

### Rollback

```bash
docker stop dcgm-exporter && docker rm dcgm-exporter
```

---

## Step 6: Configure Ethernet Static IP (Before Plugging In Cable)

### Background

Plugging in the ethernet cable without a static IP causes a full network outage — WiFi, Tailscale, and SSH all drop simultaneously. This was confirmed on 2026-03-28. The root cause is NetworkManager activating the "Wired connection 3" profile via DHCP, which disrupts routing across all interfaces.

**This step must be completed BEFORE plugging in the ethernet cable.**

### Prerequisites

This requires the `davistroy` user with interactive sudo (the `claude` user cannot run `nmcli` with sudo). Run these commands from the Spark console or as davistroy via SSH.

### Procedure

```bash
# 1. Configure static IP for ethernet (192.168.10.33, keep WiFi on .32)
sudo nmcli connection modify "Wired connection 3" \
  ipv4.method manual \
  ipv4.addresses "192.168.10.33/24" \
  ipv4.gateway "192.168.10.1" \
  ipv4.dns "192.168.10.1" \
  connection.autoconnect-priority 0

# 2. Verify the config was saved
nmcli connection show "Wired connection 3" | grep -E "(ipv4.method|ipv4.addresses|ipv4.gateway|ipv4.dns|autoconnect)"
# Expected: method=manual, addresses=192.168.10.33/24, gateway=192.168.10.1

# 3. NOW plug in the ethernet cable

# 4. Verify both interfaces are up
nmcli device status
# Expected: wlP9s9=connected (UBNT), enP7s7=connected (Wired connection 3)

# 5. Verify routing (ethernet should have lower metric = preferred)
ip route
# Expected: two default routes — ethernet at metric ~100, WiFi at metric 600

# 6. Verify both IPs respond
ping -c 2 192.168.10.32  # WiFi
ping -c 2 192.168.10.33  # Ethernet

# 7. Verify Tailscale
tailscale status
```

### Verification from Remote (run from workstation)

```bash
# All three paths should work
ssh davistroy@192.168.10.32 echo "WiFi OK"
ssh davistroy@192.168.10.33 echo "Ethernet OK"
ssh davistroy@spark.k4jda.net echo "Tailscale OK"
```

### Rollback (if ethernet causes problems)

```bash
# Unplug the ethernet cable (physical)
# OR disable ethernet via NM:
sudo nmcli connection down "Wired connection 3"
sudo nmcli connection modify "Wired connection 3" connection.autoconnect no
```

### Network Reference

| Interface | IP | Purpose | Metric |
|-----------|-----|---------|--------|
| wlP9s9 (WiFi) | 192.168.10.32 | Current primary, SSID "UBNT" | 600 |
| enP7s7 (Ethernet) | 192.168.10.33 | New primary (lower latency) | ~100 |
| tailscale0 | 100.124.10.120 | Overlay, works on either | N/A |

---

## Step 7: Ethernet Network Optimization (After Cable Is Connected)

### Background

Once the Spark is on ethernet (Step 6 completed), there are additional network optimizations.

### Procedure

```bash
# Check current ethernet settings
ethtool enP7s7

# Enable jumbo frames if your switch supports it (reduces CPU overhead for large responses)
# Only do this if ALL devices on the same VLAN support jumbo frames
sudo ip link set enP7s7 mtu 9000

# If that works, make it persistent via NetworkManager:
sudo nmcli connection modify "Wired connection 3" 802-3-ethernet.mtu 9000

# Enable TCP BBR congestion control (better throughput on modern networks)
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
echo "net.ipv4.tcp_congestion_control=bbr" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf

# Verify
sysctl net.ipv4.tcp_congestion_control
```

### Risk

Jumbo frames ONLY if your switch/router supports them. Mismatched MTU causes packet drops and mysterious connectivity issues. BBR is safe on any network.

---

## Step 8: Add ChromaDB and Neo4j to Spark (Co-located with LLM Stack)

### Background

The contact-center-lab pipeline uses ChromaDB (vector store for embeddings) and Neo4j (knowledge graph). Both are CPU/RAM services — no GPU needed. With ~10 GiB available RAM after the 0.65 gpu-memory-utilization change, there should be room for both.

**Current pipeline status:** Neither service is containerized yet. The pipeline outputs JSON files (`vector_db_atoms.jsonl`, `graph_nodes.json`, `graph_edges.json`) ready for import. Running these on the Spark co-located with the LLM stack would enable the chatbot to query them directly.

**Scale:** ~11K articles producing ~50-100K nodes/relationships. ChromaDB collection `kb_atoms` uses 384-d embeddings from `all-mpnet-base-v2` (sentence-transformers, not the Qwen embedding model).

### Memory Budget

| Service | Docker Limit | Est. Actual | GPU | Notes |
|---------|-------------|-------------|-----|-------|
| qwen35 (LLM) | — | ~3 GiB host + 88 GiB GPU | Yes | Already running |
| qwen3-embed | — | ~0.2 GiB host + 16 GiB GPU | Yes | Already running |
| GLiNER | — | ~0.1 GiB host + 2.4 GiB GPU | Yes | Already running |
| node-exporter | — | ~17 MiB | No | Already running |
| **ChromaDB** | **2 GiB** | **~500 MiB** | **No** | 11K articles × 384-d embeddings ≈ small |
| **Neo4j** | **4 GiB** | **~2.5 GiB** | **No** | heap=1g + pagecache=1g + JVM overhead |
| **Total new** | **6 GiB limit** | **~3 GiB actual** | | Leaves ~7 GiB free |

**Note on Neo4j memory:** Neo4j's JVM pre-allocates heap at startup. The formula is `heap + pagecache + OS ≈ total`. For ~100K nodes, heap=1g + pagecache=1g is sufficient. If the graph grows past 500K nodes, increase to heap=2g + pagecache=2g (requires 6 GiB limit).

**ChromaDB scaling risk:** HNSW index must fit entirely in RAM. If the collection grows to 1M+ embeddings, it needs ~4 GiB. At 11K articles this is not a concern.

### Pre-Flight (from Step 0 data)

Before deploying, verify from the Step 0 metrics:
- Available RAM > 8 GiB (need headroom beyond the ~3 GiB these services will use)
- Swap < 500 MiB (system isn't already under memory pressure)
- If either condition fails, do NOT proceed — address memory first

### Existing Container State

Both containers already exist on the Spark (stopped, with data volumes intact):

| Container | Image | Port | Volumes | Status |
|-----------|-------|------|---------|--------|
| chromadb | chromadb/chroma:latest | 8003 | `chromadb-data` | Exited (stopped) |
| neo4j | neo4j:5-community | 7474, 7687 | `neo4j-data`, `neo4j-logs` | Exited (stopped) |

**Tuning goal:** Give ChromaDB and Neo4j enough memory to perform well — not artificially constrained — while keeping enough headroom for the LLM stack. The right allocations depend on Step 0's available RAM measurement. The standalone Neo4j config (heap=4g + pagecache=2g) may be fine if we have headroom, or may need right-sizing if we're tight.

### Approach: Measure, Then Set

Don't guess memory limits — start both services with their existing configs (no Docker memory limits), run them under realistic load alongside the LLM stack, measure what they actually use, then decide whether limits are needed.

**Pre-flight gate:** Available RAM (from Step 0) must be > 8 GiB before starting. If < 8 GiB, fix memory pressure first.

### Phase 1: Start and Measure Idle

```bash
# 1. Record memory baseline before starting
free -h | tee /tmp/mem_before_chromaneo.txt

# 2. Start both with existing configs
docker start chromadb neo4j

# 3. Wait for health
until curl -sf http://localhost:8003/api/v1/heartbeat > /dev/null 2>&1; do sleep 3; done
echo "ChromaDB healthy"
until curl -sf http://localhost:7474 > /dev/null 2>&1; do sleep 5; done
echo "Neo4j healthy"

# 4. Let Neo4j JVM stabilize (60s), then measure idle state
sleep 60
echo "=== IDLE STATE ==="
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
free -h
```

### Phase 2: Load Test All Services Concurrently

```bash
# 5. Exercise ChromaDB
curl -s http://localhost:8003/api/v1/collections | python3 -m json.tool | head -20
# If kb_atoms exists, run similarity searches to exercise HNSW index

# 6. Exercise Neo4j with representative queries
curl -s http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN count(n) AS nodeCount"}]}' \
  -u neo4j:pipeline-knowledge-graph

curl -s http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n)-[r]->(m) RETURN type(r), count(r) ORDER BY count(r) DESC LIMIT 10"}]}' \
  -u neo4j:pipeline-knowledge-graph

# 7. Run LLM inference simultaneously (simulates real pipeline load)
for i in $(seq 1 10); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3.5-35b\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain concept $i of distributed systems in detail\"}],\"max_tokens\":300}" > /dev/null &
done
wait

# 8. Measure peak after concurrent load
echo "=== PEAK AFTER CONCURRENT LOAD ==="
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
free -h
cat /proc/swaps

# 9. Check LLM wasn't affected
curl -s http://localhost:8000/metrics | grep "vllm:num_preemptions_total"
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Count from 1 to 10"}],"max_tokens":50}'
# Compare tok/s against baseline (~23 tok/s)
```

### Phase 3: Decide on Limits

Based on Phase 2 measurements:

- **If available RAM > 4 GiB under concurrent load AND swap didn't increase AND LLM tok/s is unchanged** → Keep existing configs. No limits needed. Done.
- **If available RAM dropped below 4 GiB OR swap increased** → Right-size with Docker memory limits:
  - Set each service's `--memory` to measured peak + 25% headroom
  - For Neo4j, also tune heap/pagecache if the existing 4g+2g is oversized for the actual dataset
  - Recreate containers with limits (data volumes are Docker named volumes — preserved on `docker rm`)
- **If LLM preemptions > 0 OR tok/s degraded** → Stop ChromaDB/Neo4j, investigate whether memory contention affected UMA GPU allocations

### What to Watch For

| Signal | Meaning | Action |
|--------|---------|--------|
| Available RAM < 4 GiB under load | Memory tight but may be OK | Set Docker memory limits at measured peak + 25% |
| Swap increasing during test | Active memory pressure | Right-size Neo4j heap/pagecache down |
| LLM preemptions > 0 | KV cache evictions from RAM pressure | Stop ChromaDB/Neo4j, increase gpu-memory-util |
| ChromaDB/Neo4j OOM killed | Exceeded Docker limit (if set) | Increase --memory based on measured peak |
| LLM tok/s degraded > 10% | CPU or memory contention | Investigate — may need to reduce Neo4j thread count |

### Rollback

```bash
docker stop chromadb neo4j
# Data persists in Docker named volumes (chromadb-data, neo4j-data, neo4j-logs)
# To fully remove: docker rm chromadb neo4j
```

### Service Reference (after deployment)

| Service | Port | Health Check | API |
|---------|------|-------------|-----|
| qwen35 (LLM) | 8000 | `/health` | OpenAI-compatible `/v1/` |
| qwen3-embed | 8001 | `/health` | OpenAI-compatible `/v1/` |
| GLiNER (NER) | 8002 | `/` | `/v1/ner` |
| ChromaDB | 8003 | `/api/v1/heartbeat` | ChromaDB REST API |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | `/` on 7474 | Cypher via Bolt |

---

## Step 9: Docker Compose — Declarative Service Management

### Background

All containers are currently managed with individual `docker run` commands, which means:
- Startup order isn't enforced on reboot (all containers race with `--restart unless-stopped`)
- Configuration is scattered across memory files and CLAUDE.md
- No health-check-based auto-recovery (if vLLM hangs without crashing, it stays hung)
- No log rotation configured (vLLM logs grow unbounded under sustained load)

A `docker-compose.yml` makes the entire stack declarative, reproducible, and properly sequenced.

### What to Include

```yaml
# /home/claude/docker-compose.yml
services:
  qwen35:           # LLM — must start first
  qwen3-embed:      # Embedding — after qwen35 healthy
  gliner:           # NER — after qwen3-embed healthy
  chromadb:         # Vector store — independent
  neo4j:            # Graph DB — independent
  node-exporter:    # Metrics — independent
  gpu-exporter:     # GPU metrics — independent
```

Each service should include:
- **Health check:** `curl -sf http://localhost:PORT/health` with appropriate interval/retries
- **Depends_on with condition: service_healthy** — enforces startup order
- **Log rotation:** `logging: { options: { max-size: "100m", max-file: "3" } }`
- **Memory limits** (for ChromaDB/Neo4j, based on Step 8 measurements)
- **Restart policy:** `unless-stopped`
- All volume mounts using absolute paths (never `~`)

### Procedure

1. Create `docker-compose.yml` based on current running configs (`docker inspect` each container)
2. Stop all containers
3. `docker compose up -d`
4. Verify all services healthy and startup order correct
5. Test reboot behavior: services should come up in order automatically

### Risk

Low — this is a configuration change, not a behavior change. The same containers run the same way, just managed declaratively. Test by stopping and restarting the stack.

---

## Step 10: OS Cleanup — Disable Unnecessary Services

### Background

The Spark runs several desktop/workstation services that aren't needed for a headless inference server:

| Service | Purpose | RAM | Action |
|---------|---------|-----|--------|
| snap gnome/mesa/gtk/snap-store | Desktop UI packages | ~200 MB combined | Remove snaps |
| `avahi-daemon` | mDNS/DNS-SD (Bonjour) | ~5 MB | Disable |
| `multipathd` | SAN multipath storage | ~10 MB | Disable |
| `dgx-dashboard` + `dgx-dashboard-admin` | NVIDIA DGX web dashboard | ~50 MB | Keep (useful for monitoring) |
| `nvidia-dgx-telemetry` | NVIDIA telemetry | ~20 MB | Evaluate — may be sending data to NVIDIA |

### Procedure

```bash
# Disable unnecessary services
sudo systemctl disable --now avahi-daemon
sudo systemctl disable --now multipathd

# Remove desktop snaps (frees RAM and disk)
sudo snap remove snap-store
sudo snap remove gnome-46-2404
sudo snap remove gtk-common-themes
sudo snap remove mesa-2404
sudo snap remove firmware-updater

# Verify
systemctl list-units --type=service --state=running | wc -l
free -h
```

### Risk

Low. These services aren't used for inference. The snaps can be reinstalled if needed. Leave `dgx-dashboard` services — they provide useful system monitoring via NVIDIA's web UI.

---

## Step 11: Chunked Prefill Tuning

### Background

vLLM's chunked prefill splits long prompt processing into chunks, allowing decode steps for other requests to interleave. The current default is `max_num_batched_tokens=2048`.

Pipeline data shows:
- Average prompt length: ~1,382 tokens
- Average generation length: ~831 tokens
- Under 19-way concurrency, TTFT is 577ms and decode dominates (71.7s mean)

Since decode is the bottleneck (not prefill), increasing the chunk size would process prompts in fewer chunks, reducing TTFT slightly without meaningful decode impact.

### Test

```bash
# Add to qwen35 docker run command:
--max-num-batched-tokens 4096

# Measure TTFT improvement vs ITL impact under concurrent load
# If TTFT improves and ITL doesn't regress > 10%, keep it
```

### Risk

Low. If ITL degrades significantly, revert to 2048.

---

## Step 12: Data Backup Strategy

### Background

ChromaDB and Neo4j hold processed pipeline data that takes hours to regenerate. A simple backup strategy prevents data loss.

### Procedure

```bash
# Create backup script
cat > /home/claude/backup-data.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/home/claude/backups/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup ChromaDB (stop not required — consistent snapshot via docker)
docker run --rm -v chromadb-data:/data -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/chromadb-data.tar.gz -C /data .

# Backup Neo4j (stop for consistency)
docker stop neo4j
docker run --rm -v neo4j-data:/data -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/neo4j-data.tar.gz -C /data .
docker start neo4j

echo "Backup completed to $BACKUP_DIR"
ls -lh $BACKUP_DIR/
EOF
chmod +x /home/claude/backup-data.sh
```

### Schedule

Run before any major configuration change. Consider a weekly cron job once the stack is stable.

---

## Testing Cadence

| Optimization | Check frequency | How to check |
|-------------|----------------|--------------|
| Remove `--enforce-eager` | Monthly with new vLLM nightly | Try without flag, watch for Triton errors |
| Remove `--no-async-scheduling` | Monthly with new vLLM nightly | Concurrent request stress test |
| vLLM nightly update | Biweekly | Pull new image, full validation |
| Embedding memory reduction | Once (after NOW_PLAN) | Reduce to 0.10, check startup success |
| DCGM exporter | Once | Test if GB10 memory reporting works |
| Ethernet static IP (Step 6) | Done (configured 2026-03-28) | Plug in cable and verify |
| Ethernet MTU / BBR (Step 7) | Once (after ethernet connected) | ethtool check, iperf3 test |
| ChromaDB + Neo4j (Step 8) | Once (after memory headroom confirmed) | Start, load test, measure, then set limits |
| Docker Compose (Step 9) | Once (after all services validated) | Create compose, test stack restart |
| OS Cleanup (Step 10) | Once | Disable services, remove snaps |
| Chunked prefill (Step 11) | Once, re-evaluate with new vLLM | Benchmark TTFT vs ITL tradeoff |
| Data backup (Step 12) | Weekly after stack stable | Run backup script |

---

## Step 13: Configuration Snapshots and Backup

### Background

The `spark-config.sh` script captures the complete running state of the Spark into a named profile that can recreate the entire stack from scratch. Each profile includes:
- `docker-compose.yml` — all container definitions (images, args, volumes, ports, env vars)
- `sysctl.conf` — kernel tuning parameters
- `network/*.nmconnection` — NetworkManager connection files
- `models.json` — cached model inventory with sizes
- `restore.sh` — executable script to apply the config
- `system.json` — system metadata (kernel, driver, CUDA, RAM)

### When to Snapshot

Take a new snapshot:
- **After completing each LATER_PLAN step** — so you can roll back to the last known-good state
- **Before any risky change** (vLLM update, driver change, etc.)
- **When creating a new use-case profile** (e.g., "chatbot-v1" vs "pipeline-v1")

### Usage

```bash
# On the Spark:
/home/claude/spark-config.sh snapshot <name> "<description>"
/home/claude/spark-config.sh list
/home/claude/spark-config.sh show <name>
/home/claude/spark-config.sh diff <name>
/home/claude/spark-config.sh apply <name>    # Interactive confirmation

# Backup to homeserver (relay through workstation — no direct Spark→homeserver SSH):
# From workstation:
scp -i ~/.ssh/id_claude_code -r claude@spark.k4jda.net:/home/claude/spark-configs/<name> /tmp/<name>
scp -i ~/.ssh/id_claude_code -r /tmp/<name> claude@homeserver.k4jda.net:/mnt/user/appdata/spark-configs/

# Disaster recovery — pull from homeserver:
# Reverse the scp commands above
```

### Multi-Profile Strategy

| Profile | Description | When to Use |
|---------|-------------|-------------|
| `pipeline-v1` | Current config — LLM + embed + GLiNER, tuned for batch pipeline processing | Default for pipeline runs |
| `pipeline-v2` | After LATER_PLAN optimizations (enforce-eager removed, etc.) | After optimization validated |
| `full-stack` | Pipeline + ChromaDB + Neo4j running | When chatbot needs all services |
| `chatbot-v1` | Future — optimized for low-latency single-user chatbot queries | TBD |
| `maintenance` | Minimal — no LLM, just monitoring | System updates, driver changes |

Switching profiles: `spark-config.sh apply <profile-name>` stops all containers and starts the new profile's stack.

### Current State

- `pipeline-v1` — captured 2026-03-28, backed up to homeserver at `/mnt/user/appdata/spark-configs/pipeline-v1/`
- Script deployed at `/home/claude/spark-config.sh`

---

## Updated Execution Order

| Order | Step | Rationale |
|-------|------|-----------|
| 1st | Step 0: Review 24h usage data | Gate — determines parameter adjustments |
| 2nd | Step 10: OS cleanup | Free RAM/resources before adding services |
| — | **Snapshot: `pre-optimization`** | Baseline before any container changes |
| 3rd | Step 3: Update vLLM build + Marlin + prefix caching | Native SM121 kernels, Marlin FP8, prefix caching, embed sleep mode |
| 4th | Step 1b: Build custom Triton → remove --enforce-eager | **The big one** — potential 2-8x decode throughput |
| 5th | Step 2: Remove `--no-async-scheduling` | Better concurrency handling (re-test on new build) |
| 6th | Step 11: Chunked prefill tuning | Improve TTFT under concurrency |
| — | **Snapshot: `pipeline-v2`** | Capture optimized LLM config |
| 7th | Step 4: Embedding memory reduction | Quick test, independent |
| 8th | Step 5: DCGM exporter | Additive, no risk |
| 9th | Step 8: ChromaDB + Neo4j | Measure-first deployment |
| — | **Snapshot: `full-stack`** | All services running and validated |
| 10th | Step 9: Docker Compose | Codify final config declaratively |
| 11th | Step 12: Data backup | Set up after services stable |
| 12th | Step 6: Ethernet (plug in cable) | Already configured, just connect |
| 13th | Step 7: Ethernet optimization | Last — network changes after everything else |
| — | **Snapshot: final config name + backup to homeserver** | Complete optimized state preserved |

---

## Monitoring Checklist for All Changes

After any change, verify all of the following within 24 hours:

- [ ] All services remain healthy (check Grafana dashboard)
- [ ] No preemptions (`vllm:num_preemptions_total` stays at 0)
- [ ] Swap usage stays low (< 1 GiB)
- [ ] GPU temperature stays under 75°C
- [ ] Available RAM > 4 GiB
- [ ] Inference quality unchanged (spot-check a few responses)
- [ ] Grafana data continues flowing (no scrape gaps)
- [ ] Docker log sizes reasonable (`docker inspect --format '{{.LogPath}}' <container>` then check size)
