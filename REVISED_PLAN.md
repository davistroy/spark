# REVISED PLAN: Closing the SM121 Performance Gap

**Date:** 2026-03-29
**Supersedes:** LATER_PLAN.md (kept for reference)
**Based on:** Lab notebook entries 001-003, research synthesis, forensic analysis of previous session

---

## Current State

| Item | Value |
|------|-------|
| Image | cu130-known-good-20260306 (vLLM v0.17.0rc1) |
| CUDA graphs | Active (51 piecewise + 35 full) |
| MoE backend | Marlin FP8 (forced via env var) |
| Attention | FlashInfer |
| SM121 native kernels | **MISSING** (CMake bug #38126) |
| Async scheduling | Disabled (--no-async-scheduling) |
| Prefix caching | Disabled |
| Throughput | ~17-19 tok/s per-request at c8, ~140 tok/s aggregate |
| Target | 50+ tok/s per-request |

## What Failed Previously (don't repeat)

| Attempt | Failure | Root Cause | Fix |
|---------|---------|-----------|-----|
| Custom Docker build (Dockerfile.sm121) | cmake exit 255 | NVFP4 `cvt.e2m1x2` unsupported on SM121 | Disable NVFP4, cherry-pick #38126 |
| hellohal2064 image | 4-sec crash loop | Wrong model (80B not 35B), no volume mount, CUDA 13.1 vs 13.0 | Override env vars + mount model, but image is for wrong model family |
| v0.17.1 official image | Killed after 7 min | Premature — likely Triton JIT compiling | Retry with VLLM_TEST_FORCE_FP8_MARLIN=1 and 30 min patience |
| Async scheduling removal | Rolled back | Unknown crash or precautionary | Re-test with explicit monitoring |

---

## Execution Plan

### Phase 1: During Pipeline Run (NOW — CPU only, no GPU impact)

#### 1A. Fix and rebuild custom SM121 Docker image
**Why:** The build was 49/350 successful before hitting the NVFP4 bug. Fix is one line.
**Time:** ~30 min build on ARM64 (4 cores)
**Risk:** Zero to pipeline — CPU-only Docker build.

Steps:
1. Patch CMakeLists.txt in `/home/claude/vllm-build/` to disable NVFP4 for SM121
   - Either cherry-pick #38126's arch guard fix
   - Or add `sm_121` exclusion to NVFP4 guard (simpler, since we don't use NVFP4)
2. Update Dockerfile.sm121 to pass correct cmake flags
3. Build with `docker build -f Dockerfile.sm121 -t vllm-custom:sm121-v2 .`
4. Verify SM121 kernels exist: `cuobjdump` check for scaled_mm, MoE, MLA cubins
5. Verify NVFP4 was excluded (no crash on import)

#### 1B. Investigate hellohal2064 SM121 artifacts (background agent running)
**Why:** Even if the image is for a different model, the compiled .so files contain SM121 kernels that are model-agnostic.
**Time:** 5 min
**Risk:** Zero — read-only inspection.

Steps:
1. Check Triton version inside image
2. List compiled .so files with SM121 cubins
3. Determine if kernel .so files could be extracted and injected into v0.17.1

#### 1C. Prepare v0.17.1 launch command
**Why:** v0.17.1 was killed prematurely. It deserves a proper test with Marlin forcing.
**Time:** 5 min to prepare
**Risk:** Zero — preparation only.

Prepare exact docker run command with:
- `VLLM_TEST_FORCE_FP8_MARLIN=1` (skips Triton JIT, fast startup)
- `VLLM_FLASHINFER_MOE_BACKEND=latency`
- Fresh Triton cache path (not the v0.17.0rc1 cache)
- All other args from optimized-stable
- Wait script: 30 min timeout before declaring failure

#### 1D. Prepare docker-compose.yml (LATER Step 9)
**Why:** Codifies the entire stack. Useful regardless of which image wins.
**Time:** 15 min
**Risk:** Zero — file creation only.

---

### Phase 2: Pipeline Completes (requires qwen35 restart)

Execute in this order. Take a snapshot before each step.

#### 2A. Benchmark current config (single-request)
**Why:** Need clean baseline before changing anything. The "14.3 tok/s" vs "23 tok/s" discrepancy needs resolution.
**Time:** 2 min
**Risk:** None — single API call.

```bash
# Single-request benchmark (no concurrency)
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Write a detailed 500-word essay about the history of sailing."}],"max_tokens":600,"stream":false}' > /tmp/benchmark_baseline.json
# Calculate: 600 tokens / elapsed seconds = tok/s
```

#### 2B. Test custom SM121 image (vllm-custom:sm121-v2)
**Why:** This is the PRIMARY path to 50 tok/s. Native SM121 kernels.
**Time:** ~3 min (model load) + benchmark
**Risk:** Medium — new image, may have issues. Rollback: `spark-config.sh apply optimized-stable`

```bash
/home/claude/spark-config.sh snapshot pre-sm121-v2 "Before SM121 custom image test"
docker stop qwen35 && docker rm qwen35
docker run -d --name qwen35 --restart unless-stopped \
  --gpus all --ipc host --shm-size 64gb -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -v /home/<user>/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton:/root/.triton \
  vllm-custom:sm121-v2 \
  Qwen/Qwen3.5-35B-A3B \
    --served-model-name qwen3.5-35b \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.65 \
    --quantization fp8 --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --no-async-scheduling
# Monitor for SM121 kernel loading in logs
# Benchmark and compare against 2A baseline
```

**Success criteria:** No "GPU does not have native support for FP8" warning in logs. Throughput > 30 tok/s single-request.

#### 2C. Test v0.17.1 with Marlin forcing (if 2B fails)
**Why:** v0.17.1 is stable for Qwen3.5, just needs patience and proper env vars.
**Time:** ~3 min with Marlin forcing (no Triton JIT)
**Risk:** Low — well-tested version, just wasn't given enough time before.

#### 2D. Test async scheduling removal
**Why:** With 15-18 concurrent pipeline requests, async scheduling improves TTFT and GPU utilization.
**Time:** 10 min (stress test: 5 concurrent x 10 rounds)
**Risk:** Medium — the V1 crash may return. Monitor logs for `sampled_token_ids`.

#### 2E. Enable prefix caching
**Why:** Pipeline workloads share system prompts. Prefix caching avoids redundant prefill.
**Time:** Container restart + verification
**Risk:** Low — vLLM may auto-disable for hybrid Mamba models.

#### 2F. Reduce gpu-memory-utilization to 0.60
**Why:** KV cache at 1.6% usage. Freeing more host RAM helps with swap pressure.
**Time:** Container restart
**Risk:** Very low — still 80x headroom over peak KV usage.

---

### Phase 3: Optimization (after Phase 2 validated)

- Embed sleep mode (free 16 GB GPU when idle)
- OS cleanup (disable snap services, avahi, multipathd)
- Docker Compose (codify final config)
- Data backup strategy
- Ethernet (plug in cable — already configured)

---

## Parallel Execution Map

```
NOW (pipeline running):
├── 1A: Fix + rebuild SM121 Docker image ←── THE BIG ONE
├── 1B: Inspect hellohal SM121 artifacts (agent running)
├── 1C: Prepare v0.17.1 launch command
└── 1D: Write docker-compose.yml

AFTER PIPELINE:
├── 2A: Baseline benchmark
├── 2B: Test SM121 image ←── if build succeeds
│   ├── SUCCESS → snapshot pipeline-v3, proceed to 2D
│   └── FAIL → 2C: try v0.17.1
├── 2D: Test async scheduling
├── 2E: Prefix caching
└── 2F: Reduce memory util

LATER:
└── Phase 3 optimizations
```

---

## Key Insight: The Build Fix

The previous build failed at target [50/350] with:
```
ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'
```

This is NVFP4 microscaling — SM120 only, not SM121. The cmake set `-DENABLE_NVFP4_SM120=1` because it can't distinguish SM120 from SM121 (the #38126 bug).

**Fix:** Disable NVFP4 in the build. We use FP8 quantization, not NVFP4. The critical kernels (`scaled_mm`, MoE, MLA) were compiling fine before the NVFP4 error killed the build.

Two approaches:
1. **Minimal fix:** Patch CMakeLists.txt to exclude SM121 from NVFP4 arch guard
2. **Full fix:** Cherry-pick #38126 commits (0f82451 + 971766c) which fix all arch guards properly

Option 1 is faster and sufficient for our needs. Option 2 is more correct.
