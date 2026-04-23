# Implementation Plan: DGX Spark — Consolidated

**Created:** 2026-04-23
**Supersedes:** IMPLEMENT_SPARK_UPDATES.md, IMPLEMENT_SPARK_IMPROVEMENTS.md, IMPLEMENT_QWEN36_UPGRADE.md
**Baseline:** 53.5 tok/s c1, vLLM v0.19.0, Qwen3.5-35B-A3B FP8, gpu-memory-utilization 0.65
**System:** DGX Spark (GB10, SM121, 128 GB unified memory, Driver 580.142, CUDA 13.0)
**Status:** Phase 1 ready to execute

---

## Executive Summary

This plan consolidates three prior plans into a single source of truth. Two workstreams remain: a **model quality upgrade** (Qwen3.5 → Qwen3.6) and a **throughput optimization** (cu132 + MTP speculative decoding). These are complementary and sequenced so the model upgrade happens first (low risk, immediate quality gains) and the throughput experiment follows (higher risk, targets the 53.5 → 70+ tok/s gap).

### What's Already Done (Archived)

| Prior Plan | Status | Key Outcomes |
|---|---|---|
| IMPLEMENT_SPARK_UPDATES.md | **100% complete** | v0.19.0 deployed (+10% c1), ethernet cleaned, concurrency set to c12/600s, Gemma 4 evaluated (not adopted for pipeline) |
| IMPLEMENT_SPARK_IMPROVEMENTS.md Phase 1 | **FAILED** | MTP degrades on cu130. FLASHINFER/prefix-caching have zero effect on cu130. |
| IMPLEMENT_SPARK_IMPROVEMENTS.md Phase 3 | **+2%, not adopted** | cu132 base image tested WITHOUT MTP. Marginal gain doesn't justify image switch alone. |

### Key Insight Driving This Plan

Community benchmarks show 70-81 tok/s with FP8+MTP=2 — but on **cu132 builds**, not cu130. Our prior experiments tested MTP on cu130 (failed) and cu132 without MTP (+2%). The combination of cu132+MTP was **never tested**. That's the remaining throughput path.

---

## Plan Overview

| Phase | Focus | Key Deliverable | Est. Time | Dependencies |
|-------|-------|----------------|-----------|--------------|
| 1 | Qwen3.6 Model Upgrade | Better model quality, same throughput | ~2 hours | None |
| 2 | Memory Budget Tuning | More concurrent capacity | ~45 min | Phase 1 adopted |
| 3 | cu132 + MTP Throughput Experiment | Target 70+ tok/s | ~3 hours | Phase 1 adopted |
| 4 | Operational Improvements | Tool calling fix, quality baseline, naming | ~1 hour | Independent |

### Constraints

- Remote access only (Tailscale: `spark.k4jda.net`). No physical console for reboot.
- HF cache is root-owned — must download via Docker container.
- 5-model config constrains GPU memory: qwen35 (0.65), qwen3-embed (0.10), bge-m3 (0.05), gliner (~2GB), ce-service (~500MB).
- Container restart = ~150s LLM downtime (warm Triton cache).

### Live Container Config (verified 2026-04-23)

```
Image: vllm/vllm-openai:v0.19.0-aarch64-cu130
Model: Qwen/Qwen3.5-35B-A3B
Served as: qwen3.5-35b
Flags: --max-model-len 32768 --gpu-memory-utilization 0.65 --quantization fp8
       --kv-cache-dtype fp8 --reasoning-parser qwen3 --language-model-only
       --enable-auto-tool-choice --tool-call-parser qwen3_coder
Env: VLLM_FLASHINFER_MOE_BACKEND=latency
Volumes: /home/davistroy/.cache/huggingface:/root/.cache/huggingface
         /home/claude/.cache/triton:/root/.triton
IPC: host, SHM: 64GB
```

### Global Rollback

Any phase can roll back to this known-working config:

```bash
docker stop qwen35 && docker rm qwen35
docker run -d \
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
    --tool-call-parser qwen3_coder
```

<!-- BEGIN PHASES -->

---

## Phase 1: Qwen3.6 Model Upgrade

**Estimated Time:** ~2 hours (including download)
**Dependencies:** None
**Risk:** Low — same architecture, same image, same flags. Weight swap only.

### Background

Qwen3.6-35B-A3B uses the identical GDN hybrid architecture as Qwen3.5 (`model_type: "qwen3_5_moe"`, 40-layer, 256 experts, 3B active). Improvements are training-only: agentic coding (SWE-bench +8pts to 73.4), reasoning (AIME26 92.7), and multimodal (skippable via `--language-model-only`). vLLM v0.19.0 loads it as the same model class.

**Critical constraint:** `--served-model-name qwen3.5-35b` is hardcoded in 6+ downstream consumers (contact-center-lab config, CFA scripts, Grafana dashboard). Must preserve during experiment.

### Work Item 1.1: Download Qwen3.6 weights (non-disruptive)
**Status: PENDING**

```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net \
  "docker run --rm \
    -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.19.0-aarch64-cu130 \
    python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-35B-A3B')
print('Download complete')
\""
```

Downloads ~72 GB (26 safetensors shards). Runs alongside live model — zero disruption.

**Acceptance:** 26 shards present, config.json shows `model_type: "qwen3_5_moe"`.

---

### Work Item 1.2: Verify download integrity
**Status: PENDING**

Verify shard count, config.json model_type, and architectures list via a temporary container reading the HF cache.

**Acceptance:** 26 shards, `qwen3_5_moe` model_type, `Qwen3_5MoeForConditionalGeneration` architecture.

---

### Work Item 1.3: Container swap to Qwen3.6
**Status: PENDING**
**Pre-flight:**
- Confirm no active workload: `curl -s http://spark.k4jda.net:8000/metrics | grep 'vllm:num_requests_running'` → 0
- Record current container ID

```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net \
  "docker stop qwen35 && docker rm qwen35 && \
   docker run -d \
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
    Qwen/Qwen3.6-35B-A3B \
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
      --tool-call-parser qwen3_coder"
```

**Only change:** `Qwen/Qwen3.5-35B-A3B` → `Qwen/Qwen3.6-35B-A3B`.

Monitor startup: watch for Triton cache hits (warm = ~150s) vs recompilation (cold = 15-30+ min).

**Acceptance:** `/health` returns 200. No CUDA OOM. No model loading errors.

---

### Work Item 1.4: Throughput benchmark (c1, c4, c8)
**Status: PENDING**

Run identical methodology to Entry 022 (post-power-cycle clean benchmark).

| Metric | Baseline (Qwen3.5) | Pass Criterion |
|--------|-------------------|----------------|
| c1 tok/s | 53.5 | >= 50 (within 5%) |
| c4 aggregate | 140.4 | >= 126 (within 10%) |
| c8 aggregate | 216.0 | >= 194 (within 10%) |

**Acceptance:** All three pass criteria met.

---

### Work Item 1.5: Quality smoke test
**Status: PENDING**

5 representative prompts:
1. JSON compliance — structured output with required fields
2. Instruction following — multi-step extraction with format constraints
3. Reasoning — chain-of-thought with `<think>` block
4. Tool calling — function call via `qwen3_coder` parser
5. Long-form generation — technical writing quality

**Acceptance:** No structural regressions (broken JSON, missed instructions, malformed tool calls). Minor wording differences acceptable.

---

### Work Item 1.6: Adopt/rollback decision gate
**Status: PENDING**

| Criterion | Required |
|-----------|----------|
| c1 >= 50 tok/s | Yes |
| c4/c8 within 10% of baseline | Yes |
| All 5 quality tests pass | Yes |
| Thinking mode functional | Yes |
| No errors in container logs | Yes |

**All pass → adopt.** Proceed to Work Item 1.7.
**Any fail → rollback.** Execute global rollback command. Document failure in LAB_NOTEBOOK.

---

### Work Item 1.7: Update documentation (adopt path only)
**Status: PENDING**

- **SPARK_BASELINE.md:** model field, benchmark numbers, version tracking
- **LAB_NOTEBOOK.md:** full experiment entry (download, swap, benchmarks, quality, decision)
- **Memory: spark-device.md:** update qwen35 container section with actual live config (fixes stale pre-2026-04-11 documentation)
- **Memory: MEMORY.md:** update index line

---

### Phase 1 Exit Criteria

- [ ] Qwen3.6 running and validated, OR rollback executed and documented
- [ ] Benchmark numbers recorded
- [ ] Documentation updated
- [ ] Both model weights cached (Qwen3.5 + Qwen3.6) for instant rollback

---

## Phase 2: Memory Budget Tuning

**Estimated Time:** ~45 min
**Dependencies:** Phase 1 adopted (benchmark against Qwen3.6, not 3.5)
**Risk:** Medium — memory pressure possible at higher utilization

### Background

GPU memory utilization is 0.65. KV cache usage peaks at 1.6-3% under pipeline load — massively over-provisioned. vLLM issue #37121 confirms GDN hybrid models have ~7x KV cache overestimation. Increasing utilization to 0.70 or 0.75 could improve concurrent request capacity without risk, since actual cache usage is so low.

### Work Item 2.1: Increase gpu-memory-utilization to 0.70
**Status: PENDING**

Same container config as Phase 1 adopted config, with one change:
```diff
- --gpu-memory-utilization 0.65
+ --gpu-memory-utilization 0.70
```

Memory budget at 0.70: ~85.1 GiB for qwen35 + ~12.2 qwen3-embed + ~6 bge-m3 + ~2 gliner + ~0.5 ce-service = ~105.8 GiB. Leaves ~15.8 GiB free (safe).

**Acceptance:** Container starts without OOM. Swap stays < 200 MB under c8 load for 10 min.

---

### Work Item 2.2: Benchmark at 0.70
**Status: PENDING**

Run c1, c4, c8 benchmarks.

**Acceptance:** No throughput regression vs Phase 1. No swap growth. Temperature < 70C.

---

### Work Item 2.3: Test 0.75 (conditional)
**Status: PENDING**

If 0.70 is stable, test 0.75. Memory budget at 0.75: ~91.2 GiB for qwen35. Total ~112.4 GiB. Leaves ~9.2 GiB free — tighter margin.

**Acceptance:** Same criteria as 2.1. If swap grows or system memory drops below 8 GiB, revert to 0.70.

---

### Work Item 2.4: Update documentation
**Status: PENDING**

Update SPARK_BASELINE.md with new utilization value and benchmark numbers. LAB_NOTEBOOK entry.

### Phase 2 Exit Criteria

- [ ] Optimal utilization identified (0.70 or 0.75) and deployed
- [ ] Benchmarks confirm no regression
- [ ] Documentation updated

---

## Phase 3: cu132 + MTP Throughput Experiment

**Estimated Time:** ~3 hours (requires pipeline idle)
**Dependencies:** Phase 1 adopted (test against Qwen3.6, not 3.5)
**Risk:** Medium — cu132 image exists but MTP was never tested on it

### Background

The community's 70-81 tok/s numbers come from cu132 builds with MTP=2. Our prior experiments isolated the variables but never combined them:
- MTP on cu130 → **degraded** (cu130 lacks native MTP kernel codegen)
- cu132 without MTP → **+2%** (marginal, kernel codegen improvement alone is small)
- cu132 WITH MTP → **never tested** ← this is the experiment

The `vllm-cu132-test:latest` image (26.1 GB) already exists on the Spark from the Phase 3 cu132 build.

### Work Item 3.1: Verify cu132 image still works
**Status: PENDING**

```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net \
  "docker run --rm vllm-cu132-test:latest python3 -c 'import vllm; print(vllm.__version__)'"
```

Check that `vllm-cu132-test:latest` loads. Verify vLLM version (should be 0.19.1rc1.dev219+cu132).

**Acceptance:** Image loads, version confirmed.

---

### Work Item 3.2: Test cu132 + MTP=2 with Qwen3.6
**Status: PENDING**

Stop production qwen35. Start cu132 container with MTP speculative decoding:

```bash
docker stop qwen35 && docker rm qwen35
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton-cu132:/root/.triton \
  vllm-cu132-test:latest \
  Qwen/Qwen3.6-35B-A3B \
    --served-model-name qwen3.5-35b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization <Phase 2 winner> \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**Changes from Phase 1 config:**
- Image: `vllm-cu132-test:latest` (was `v0.19.0-aarch64-cu130`)
- Triton cache: `/home/claude/.cache/triton-cu132:/root/.triton` (separate cache for cu132 kernels)
- Added: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
- gpu-memory-utilization: from Phase 2 winner

**Note:** First startup may need full Triton recompilation (15-30+ min on cu132). Use separate cache dir to preserve cu130 Triton cache for rollback.

**Acceptance:** Container starts, MTP confirmed active in logs, `/health` returns 200.

---

### Work Item 3.3: Benchmark cu132 + MTP
**Status: PENDING**

Run c1, c4, c8, c16 benchmarks.

| Metric | Phase 1 Baseline | Community Target | Adopt Threshold |
|--------|-----------------|-----------------|-----------------|
| c1 tok/s | ~53.5 | 70-81 | >= 65 (+20%) |
| c4 aggregate | ~140 | — | >= 140 (no regression) |

**Acceptance:** c1 >= 65 tok/s AND no concurrent throughput regression.

---

### Work Item 3.4: Adopt/rollback decision
**Status: PENDING**

| Result | Action |
|--------|--------|
| c1 >= 65 tok/s, no regressions | Adopt cu132+MTP as production |
| c1 55-65, no regressions | Consider adopting; document tradeoff |
| c1 < 55 or regressions | Rollback to Phase 1/2 config (cu130, no MTP) |
| MTP causes errors/crashes | Rollback, document. Performance ceiling is ~53-55 tok/s on current setup. |

---

### Work Item 3.5: Update documentation (adopt path only)
**Status: PENDING**

Update SPARK_BASELINE.md, LAB_NOTEBOOK, spark-device.md. If cu132+MTP adopted, update CLAUDE.md with new operational rules (separate Triton cache for cu132).

### Phase 3 Exit Criteria

- [ ] cu132+MTP tested and decision made
- [ ] If adopted: production running cu132+MTP, docs updated
- [ ] If not adopted: rollback to Phase 1/2 config, failure documented with root cause
- [ ] Performance ceiling documented either way

---

## Phase 4: Operational Improvements

**Estimated Time:** ~1 hour
**Dependencies:** None — independent of Phases 1-3
**Risk:** Low

### Work Item 4.1: Tool calling parser upgrade
**Status: PENDING**

Test `--tool-call-parser qwen3_xml` with enhanced jinja template (Dickson fix, Apr 13 forum post). Reported to fix 6-hour session stability issues with tool calling. Can test on a spare port without affecting production.

**Acceptance:** Tool calling works reliably for 10+ sequential calls without parser errors.

---

### Work Item 4.2: Quality baseline with spark-evals
**Status: PENDING**

Review DanTup/spark-evals methodology. Run Inspect AI quality evals against current production config to establish a quality baseline for future quant format decisions.

**Acceptance:** Quality scores recorded for current model+quant combination.

---

### Work Item 4.3: Model name standardization (planning only)
**Status: PENDING**

Plan a coordinated rename from `qwen3.5-35b` to a generic name (e.g., `spark-llm`) across all consumers:
- contact-center-lab/pipeline/config.yaml
- contact-center-lab/pipeline/tests
- cfa/pipeline/scripts
- spark-monitor-dashboard.json (15+ Prometheus queries)

This is a planning item — the actual rename is a separate coordinated change.

**Acceptance:** File list and change plan documented. Not executed here.

### Phase 4 Exit Criteria

- [ ] Tool calling parser tested
- [ ] Quality baseline established (or deferred with rationale)
- [ ] Model naming plan documented

<!-- END PHASES -->

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Triton cache miss on Qwen3.6 (Phase 1) | Low | Medium — 30+ min startup | Identical tensor shapes → cache should hit. Rollback immediate. |
| Quality regression in Qwen3.6 (Phase 1) | Low | High — pipeline reliability | 5-prompt smoke test before production use |
| Memory pressure at 0.70/0.75 (Phase 2) | Low | Medium — swap growth | Monitor for 10 min under load. Revert if swap > 200 MB. |
| MTP crashes on cu132 (Phase 3) | Medium | Low — rollback to cu130 | Keep cu130 Triton cache. Separate cache dirs. |
| cu132 image incompatible with Qwen3.6 (Phase 3) | Low | Low — use cu130 | Test without MTP first, then add MTP. Isolate variables. |

---

## Scope Boundaries

### In Scope
- Qwen3.6 model upgrade (download, swap, benchmark, validate)
- Memory utilization tuning (0.65 → 0.70/0.75)
- cu132 + MTP throughput experiment
- Tool calling parser test
- Quality baseline establishment

### Out of Scope
- Vision/multimodal enablement (`--language-model-only` stays)
- Hybrid INT4+FP8 custom checkpoint build
- SGLang migration
- Driver upgrade (580.142 → newer)
- Model name rename execution (Phase 4.3 is planning only)
- KV cache overestimation fix (vLLM #37121 — waiting for upstream fix)

### Archived Plans
- `IMPLEMENT_SPARK_UPDATES.md` — 100% complete (v0.19.0, ethernet, Gemma 4). Archive to `docs/archive/`.
- `IMPLEMENT_SPARK_IMPROVEMENTS.md` — Phase 1 failed, Phase 3 +2% not adopted. Surviving items folded into this plan's Phases 2-3.
- `IMPLEMENT_QWEN36_UPGRADE.md` — Folded into this plan's Phase 1.
