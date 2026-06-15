# Implementation Plan: Autoresearch LLM Model Evaluation Study

**Created:** 2026-05-16
**Completed:** 2026-05-18 ~17:30 UTC
**Branch:** main
**Status:** COMPLETE (26/27 items; D.6 u=0.70 sub-run intentionally skipped after D.5 results clear)
**Prior plan:** Reference only — `IMPLEMENTATION_PLAN.md` (Performance Sprint, COMPLETE 2026-04-30, 18/18 items)

**Outcome:** `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quantized) recommended for production. GLM-4.7-Flash and Coder-Next-FP8 both rejected (40-70% slower than Qwen3.6 with equivalent quality). Production NOT changed — recommendation deferred for user approval. See `MODEL_EVALUATION_2026_05.md`.

**Context:** Research output (LAB_NOTEBOOK Autoresearch LLM Selection, 2026-05-16) recommends three models for evaluation against the current Qwen3.6-35B-A3B (BF16 + on-the-fly FP8) baseline: (1) `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quantized), (2) `Qwen/Qwen3-Coder-Next-FP8` (80B/3B hybrid), (3) `zai-org/GLM-4.7-Flash` (30B/3B MoE+MLA). User selected: swap-on-demand deployment, full study of all three, research-grade testing per model, contiguous-block execution. Ultra-plan analysis grouped work into 5 change sets (A–E) with CS-A as prerequisite.

**Goal:** Produce measured-on-Spark comparison across the three models and current baseline, against retire AR-1/AR-2/AR-3/AR-4 workloads. Output: comparative report driving a production decision (replace baseline, workload-route, or status quo).

**Scope:**
- Build model-agnostic swap harness reusing existing benchmark infrastructure (`/home/claude/{benchmarks,inspect-evals,concurrent-bench.py,stress-test.py}`)
- New AR-style task suite (only net-new test code)
- Three model evaluations, each with full research-grade test suite
- Comparative report + LAB_NOTEBOOK entries + memory updates
- Restoration of current production state between each evaluation

**Exclusions:**
- Concurrent multi-model serving on different ports (explicitly rejected by user)
- Downstream consumer code changes for non-Qwen tool-call format (follow-up if GLM adopted)
- Workload routing infrastructure (separate design if multi-model adoption indicated)
- Honorable-mention models (Nemotron, Devstral, etc.)
- Driver/firmware upgrades
- Kernel-level tuning (Entry 057 closed; no viable path on SM121)
- External dependency: vllm-project/vllm#37554 (FP8 KV hybrid GDN+attention bug)

**Risk Summary:**

| Phase | Risk | Probability | Impact | Rollback |
|-------|------|-------------|--------|----------|
| A (Harness) | docker-compose env refactor breaks production | Low | High | `docker-compose.yml.pre-eval-study` backup; `docker compose -f <backup> up -d qwen35` |
| A (Harness) | Parity gate fails (harness bug) | Low | Medium | Block CS-B; debug before proceeding |
| B (Qwen3.6-FP8) | Pre-quant hang on cu132 (v0.19.1rc1) | Low | High | Stop container; revert; flush Triton if poisoned |
| B (Qwen3.6-FP8) | FP8 KV quality probe inconclusive | Medium | Low | Document as "did not falsify"; preserve current config |
| C (GLM-4.7) | MLA whitelist patch absent in cu132 image | Medium | High | Pull `scitrera/spark-vllm:glm47` fallback |
| C (GLM-4.7) | Tool-call JSON malformed | Low | Medium | Document parser issue; exclude tool-call tests for GLM |
| D (Coder-Next) | cu132 image lacks `qwen3_next` arch | Medium | High | Build derived image OR pull cu130 nightly + accept Triton recompile |
| D (Coder-Next) | OOM with 0.80 util concurrent with bge-m3/gliner | High | Medium | Profile pre-stops bge-m3 + gliner (confirmed acceptable) |
| D (Coder-Next) | 4h soak fails partway | Medium | Medium | Document partial; don't extrapolate |
| E (Synthesis) | No clear winner | Medium | Low | Report status quo as outcome |

**Hard stop conditions** (immediate revert to production):
- GPU driver hang requiring host reboot
- Triton cache produces incorrect outputs (kernels compile but math wrong)
- Cumulative evaluation downtime exceeds 30% of wall-clock time

**Execution notes:**
- All SSH operations: `ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net`
- Production state preserved across all phases; swap-on-demand for LLM only
- Container restart with model load + Triton warm: ~6–12 min (model-dependent)
- Benchmark protocol: `throughput_bench.py` (600 max_tokens, c1/c4/c8/c16, 3 runs/level) extended with quality + AR + soak
- Every experiment → LAB_NOTEBOOK.md entry before proceeding to next sub-run
- Current baseline (2026-05-10 soak): mean 43.1s latency, 12.06 tok/s throughput, 100% success, c1=65.9 tok/s
- Hardware/driver invariant: driver 580.142, kernel 6.17.0-1014 (or current), cu132+MTP image at `vllm-cu132-test:latest`
- HF cache path: `/home/davistroy/.cache/huggingface` (NEVER `~`); Triton cache: `/home/claude/.cache/triton-cu132` (cu132-specific)
- **CRITICAL**: Per CLAUDE.md, never modify production docker-compose.yml in place without backup; always test on idle state; one variable at a time when debugging

---

## Phase A: Test Infrastructure & Safety Net

**Goal:** Build a model-agnostic swap harness that reuses existing benchmark scripts, adds an AR-style task suite, and produces a parity-validated measurement protocol before any model swap begins.

**Estimated time:** ~6 hours.

### Work Item A.1 — Backup current production state

**Status:** Pending
**Depends on:** None

**Task:** Create timestamped backup of `docker-compose.yml`, snapshot current GPU memory baseline, capture current vLLM version and image hash, log current consumer service IDs.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'cd /home/claude && cp docker-compose.yml docker-compose.yml.pre-eval-study'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker inspect qwen35 --format "{{.Config.Image}}" > /home/claude/llm-eval-prestate.txt'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker logs qwen35 2>&1 | grep -i "vllm.*version" | head -3 >> /home/claude/llm-eval-prestate.txt'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> /home/claude/llm-eval-prestate.txt'
```

**Acceptance criteria:** Backup file exists; prestate.txt records image, vLLM version, GPU memory baseline. Verify with `cat /home/claude/llm-eval-prestate.txt`.

### Work Item A.2 — Create `/home/claude/llm-eval/` harness directory

**Status:** Pending
**Depends on:** A.1

**Task:** Create directory layout: `llm-eval/profiles/`, `llm-eval/ar_tasks/`, `llm-eval/results/<run_id>/`. Add `README.md` documenting harness invocation.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'mkdir -p /home/claude/llm-eval/{profiles,ar_tasks,results,scripts}'
```

**Acceptance criteria:** Directory structure exists; readable by `claude` user.

### Work Item A.3 — Parameterize qwen35 service in docker-compose.yml

**Status:** Pending
**Depends on:** A.1

**Task:** Refactor qwen35 service `command:` block to read from env vars with current production values as `.env` defaults. New `/home/claude/.env` holds: `LLM_IMAGE`, `LLM_MODEL`, `LLM_QUANTIZATION`, `LLM_KV_DTYPE`, `LLM_TOOL_PARSER`, `LLM_REASONING_PARSER`, `LLM_SPEC_TOKENS`, `LLM_GPU_UTIL`, `LLM_MAX_BATCHED_TOKENS`, `LLM_EXTRA_ARGS`. Profile env files override these.

**Critical preservation:** Container name `qwen35`, port 8000, served-model-name `spark-llm`, healthcheck timing — these are downstream-consumer-visible and MUST NOT change. Volume mounts and Triton cache path also preserved.

**Files modified:** `/home/claude/docker-compose.yml`, new `/home/claude/.env`

**Acceptance criteria:**
- `docker compose config` validates without errors
- `docker compose up -d qwen35` (with default env) produces a container indistinguishable from prior production: same image, same `served-model-name`, same MTP=2, same `--kv-cache-dtype fp8`
- `/health` returns 200 within current cold-start budget (~6.5 min)

### Work Item A.4 — Adapt benchmark scripts for parameterization

**Status:** Pending
**Depends on:** A.2

**Task:** Modify three existing scripts to read served-model-name from env or argv (currently some hardcode `qwen3.5-35b`):
- `/home/claude/benchmarks/throughput_bench.py` — add `--model-name` CLI arg, output JSON to stdout
- `/home/claude/concurrent-bench.py` — replace hardcoded `qwen3.5-35b` with env-read or argv
- `/home/claude/stress-test.py` — same parameterization; verify 4h durations work (currently proven at 30 min only)

**Note:** Keep originals untouched as `*.orig.bak`; modifications go in `/home/claude/llm-eval/scripts/`.

**Acceptance criteria:** Each script runs against current production with `--model-name spark-llm`; produces equivalent output to original; new JSON output schema documented in `llm-eval/scripts/README.md`.

### Work Item A.5 — Define common metrics schema

**Status:** Pending
**Depends on:** A.2

**Task:** Create `/home/claude/llm-eval/metrics_schema.json` defining the JSON contract every test produces. Required fields:
- Run metadata: `run_id`, `started_at`, `ended_at`, `model`, `image`, `vllm_version`, `kv_dtype`, `mtp_n`, `gpu_util`, `extra_args`
- Throughput: per concurrency level c1/c4/c8/c16 — `tok_per_sec_aggregate`, `tok_per_sec_per_request_mean`, `p50_latency_s`, `p99_latency_s`, `success_rate`
- Quality: `swe_bench_subset_score`, `gpqa_d_probe_score`, `ar_task_pass_rate` (split by AR-1/2/3/4 subtypes)
- Stability: `soak_duration_s`, `crashes`, `restarts`, `gpu_mem_peak_gib`, `gpu_mem_drift_gib_per_hour`, `mtp_acceptance_rate`
- Errors: `oom_count`, `timeout_count`, `parse_errors`

**Acceptance criteria:** Schema file exists; harness orchestrator (A.7) writes valid JSON conforming to it.

### Work Item A.6 — Build AR-style task suite

**Status:** Pending
**Depends on:** A.2

**Task:** Construct ~30 fixture tasks in `/home/claude/llm-eval/ar_tasks/`:
- 10 AR-1-style: multi-file invariant reasoning. Example: "Here is a CRN preservation invariant and a scalar↔vec parity check. Given this 2K-LOC optimizer module spread across 3 files, identify the line that breaks parity beyond 0.001 tolerance." Each task has expected reasoning trace and final answer.
- 10 AR-2/3/4-style: single-file convergence. Example: "Here is a 500-LOC pytest. It fails with assertion X. Edit the source file to make it pass without changing the test." Expected pass = test runs green.
- 10 tool-call/YAML: structured output generation. Example: "Generate a YAML scenario with N=5 actors, M=12 events conforming to this schema." Expected pass = YAML parses and validates against schema.

**Files:** `ar_tasks/ar1_*.json`, `ar_tasks/ar2_*.json`, etc. Each fixture: `{prompt, expected_pattern, tolerance, eval_strategy}`.

**Runner:** `/home/claude/llm-eval/scripts/run_ar_tasks.py` — issues request to `http://localhost:8000/v1/chat/completions`, scores per fixture, outputs JSON pass/fail per task.

**Acceptance criteria:**
- 30 fixtures exist
- Runner against current production yields a non-zero pass rate (sanity: model can do *some* of the tasks)
- Runner output conforms to metrics_schema.json's `ar_task_pass_rate` shape

### Work Item A.7 — Build full-suite orchestrator

**Status:** Pending
**Depends on:** A.3, A.4, A.5, A.6

**Task:** Create `/home/claude/llm-eval/scripts/run_full_suite.sh` taking a profile name and orchestrating the full research-grade test sequence:
1. `docker compose --env-file profiles/<name>.env up -d qwen35`
2. Wait for `/health` 200 (max 15 min) — abort on timeout
3. Smoke: single chat completion
4. Throughput: c1/c4/c8/c16 × 3 runs each (existing `throughput_bench.py`)
5. Quality: SWE-bench subset via `inspect_evals`, GPQA-D probes (subset of fixtures from `inspect-evals/`)
6. AR tasks: full 30-fixture suite
7. Soaks (separate runs at c=1, c=4, c=8, 1h each)
8. 4h stability test at c=4 (concurrent baseline)
9. Emit consolidated `results/<run_id>/summary.json` conforming to schema
10. `docker compose stop qwen35` (do NOT remove; for fast next-iteration)

**Acceptance criteria:** Script runs end-to-end against the *current production profile* (A.8 below); summary.json validates against schema.

### Work Item A.8 — Parity gate: validate harness against 2026-05-10 baseline

**Status:** Pending
**Depends on:** A.7

**Task:** Create `/home/claude/llm-eval/profiles/current_baseline.env` carrying the current production env values exactly. Run `run_full_suite.sh current_baseline`. Compare results against 2026-05-10 soak: mean latency 43.1s ± 5.2s, throughput 12.06 tok/s, 100% success rate, c1=65.9 tok/s, MTP acceptance 80.7%.

**Verification point 1 (VP1):** Harness reproduces 2026-05-10 metrics within ±10%. If outside this window, harness has a bug — STOP, debug, retry. Do not proceed to CS-B until parity gate passes.

**Acceptance criteria:**
- Throughput at c=1 within 65.9 ± 6.6 tok/s
- Soak success rate ≥ 99.5% (parity with 100%)
- p99 latency within 62.3s ± 6.2s
- MTP acceptance ≥ 75% (parity with 80.7%)

**Output:** `/home/claude/llm-eval/results/parity_gate_2026-05-16.json`; LAB_NOTEBOOK entry documenting parity result.

---

## Phase B: Qwen/Qwen3.6-35B-A3B-FP8 Evaluation (pre-quantized)

**Goal:** Test research's #1 recommendation against current on-the-fly FP8 baseline. Three sub-experiments isolate (a) pre-quant vs on-the-fly, (b) FP8 KV cache vs BF16 KV cache quality impact on AR-1, (c) MTP=1 vs MTP=2 throughput tradeoff. Pre-existing Entry 054-055 rejection re-validated under current v0.19.1rc1.

**Estimated time:** ~14 hours.

### Work Item B.1 — Download Qwen/Qwen3.6-35B-A3B-FP8

**Status:** Pending
**Depends on:** A.8

**Task:** Stop qwen35, download model to HF cache, verify file count and sizes.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose stop qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'HF_HOME=/home/davistroy/.cache/huggingface python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3.6-35B-A3B-FP8\")"'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'du -sh /home/davistroy/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8'
```

**Acceptance criteria:** Disk usage ~36 GB; safetensors file count matches HF manifest; no download errors.

### Work Item B.2 — Sub-run 1: Pre-quant FP8, BF16 KV cache, MTP=1

**Status:** Pending
**Depends on:** B.1

**Task:** Create `/home/claude/llm-eval/profiles/qwen36_fp8_bf16kv_mtp1.env`:
```
LLM_IMAGE=vllm-cu132-test:latest
LLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
LLM_QUANTIZATION=
LLM_KV_DTYPE=auto
LLM_TOOL_PARSER=qwen3_coder
LLM_REASONING_PARSER=qwen3
LLM_SPEC_TOKENS=1
LLM_GPU_UTIL=0.70
LLM_MAX_BATCHED_TOKENS=32768
LLM_EXTRA_ARGS=--enable-prefix-caching --load-format fastsafetensors
```

Run full suite: `cd /home/claude/llm-eval && ./scripts/run_full_suite.sh qwen36_fp8_bf16kv_mtp1`.

**Verification point 2 (VP2):** Cold start ≤ 8 min; c=1 throughput within ±20% of current 65.9 (pre-quant expected modestly lower per Entry 054-055; this is the BF16-KV variant which is novel).

**Acceptance criteria:** Full suite completes; results JSON conforms to schema; LAB_NOTEBOOK entry with comparison to current baseline.

### Work Item B.3 — Sub-run 2: Pre-quant FP8, FP8 KV cache, MTP=1

**Status:** Pending
**Depends on:** B.2

**Task:** Create profile `qwen36_fp8_fp8kv_mtp1.env` (same as B.2 but `LLM_KV_DTYPE=fp8`). Run full suite.

**Note:** Skip the full 4h stability test if B.2 already showed reject signal; document this decision in LAB_NOTEBOOK.

**Verification point 3 (VP3):** AR-1 task pass rate documented vs B.2. Research claim: 4/8 failures with FP8 KV vs 1/8 with BF16. With 10 AR-1 fixtures, expect (loosely): FP8 KV ≤ 50% pass; BF16 KV ≥ 80% pass. Any direction is informative.

**Acceptance criteria:** Sub-run completes; FP8-KV vs BF16-KV delta on AR-1 explicitly recorded; comparison table in LAB_NOTEBOOK entry.

### Work Item B.4 — Sub-run 3: Pre-quant FP8, BF16 KV cache, MTP=2

**Status:** Pending
**Depends on:** B.3

**Task:** Create profile `qwen36_fp8_bf16kv_mtp2.env` (same as B.2 but `LLM_SPEC_TOKENS=2`). Run *abbreviated* suite — throughput c1/c4/c8/c16 + 1h c=4 soak only (skip full quality + AR; only the MTP setting changed; quality is invariant in MTP).

**Verification point 4 (VP4):** MTP=1 vs MTP=2 acceptance and throughput delta documented. Research argument: MTP=2 "tanks acceptance" on single-layer MTP. Our prior data: 80.7% acceptance at MTP=2. Resolve empirically.

**Acceptance criteria:** Throughput + acceptance numbers recorded; explicit MTP-N recommendation written into the per-model section of comparative report (Phase E).

### Work Item B.5 — Restore current production state

**Status:** Pending
**Depends on:** B.4

**Task:** Stop and remove qwen35 container; redeploy with default `.env` (current production). Verify spark-llm endpoint responds normally.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose rm -f qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose up -d qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'timeout 600 bash -c "until curl -sf http://localhost:8000/health; do sleep 10; done"'
```

**Verification point 5 (VP5):** `/health` returns 200; spot-check via `quick-bench.sh`-style request yields throughput within ±10% of baseline c=1.

**Acceptance criteria:** Production restored; downstream consumer health check passes for all 5 services depending on `spark-llm`.

### Work Item B.6 — LAB_NOTEBOOK entry: Qwen3.6-FP8 pre-quant evaluation

**Status:** Pending
**Depends on:** B.5

**Task:** Write LAB_NOTEBOOK entry (`Entry 069` or next available) summarizing all three sub-runs. Required content:
- Pre-quant vs on-the-fly throughput delta (validates or invalidates Entry 054-055 under v0.19.1rc1)
- FP8 KV vs BF16 KV AR-1 quality delta (validates or invalidates research gotcha #1)
- MTP=1 vs MTP=2 acceptance and throughput delta (resolves Spark-prior vs research recommendation)
- Decision: adopt/reject for Phase E synthesis

**Acceptance criteria:** Entry committed; cross-references to research output and prior entries (054-055); MEMORY.md updated with one-line index.

---

## Phase C: zai-org/GLM-4.7-Flash Evaluation

**Goal:** Evaluate GLM-4.7-Flash for AR-4 throughput specialist and tool-call workloads. Validates MLA whitelist patch presence; documents driver-580.142 artifact behavior; measures FP8 single-stream tok/s.

**Estimated time:** ~6 hours.

### Work Item C.1 — Verify MLA whitelist patch in cu132 image

**Status:** Pending
**Depends on:** B.6

**Task:** Inspect `vllm-cu132-test:latest` for presence of `glm4_moe_lite` in MLA whitelist.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker run --rm --entrypoint python3 vllm-cu132-test:latest -c "
import vllm
import inspect, importlib
m = importlib.import_module(\"vllm.config\")
src = inspect.getsource(m)
print(\"MLA_PRESENT\" if \"glm4_moe_lite\" in src else \"MLA_ABSENT\")
"'
```

**Decision branch:**
- `MLA_PRESENT`: skip C.2; use `vllm-cu132-test:latest` for GLM
- `MLA_ABSENT`: proceed to C.2 (build derived image)

**Acceptance criteria:** Output is unambiguously one of the two markers.

### Work Item C.2 — Build derived image with MLA patch (conditional)

**Status:** Pending (skip if C.1 = MLA_PRESENT)
**Depends on:** C.1

**Task:** Create `/home/claude/llm-eval/Dockerfile.glm47-patch`:
```dockerfile
FROM vllm-cu132-test:latest
RUN python3 -c "
import vllm.config as c, inspect, re
src_path = inspect.getfile(c)
with open(src_path) as f: src = f.read()
# Add glm4_moe_lite to MLA whitelist (per joshua8.ai/glm-4-7-flash-vllm-128k-setup)
patched = re.sub(
    r'(MLA_SUPPORTED_MODELS\s*=\s*\\{)',
    r'\\1\"glm4_moe_lite\", ',
    src
)
with open(src_path, 'w') as f: f.write(patched)
"
```

Build: `docker build -f Dockerfile.glm47-patch -t vllm-cu132-test:glm47 /home/claude/llm-eval/`.

**Fallback if build fails:** `docker pull scitrera/spark-vllm:glm47` (research-cited pre-patched image); document image substitution in profile.

**Acceptance criteria:** Image `vllm-cu132-test:glm47` exists; inspecting it for `glm4_moe_lite` returns MLA_PRESENT.

### Work Item C.3 — Download zai-org/GLM-4.7-Flash

**Status:** Pending
**Depends on:** C.1 (and C.2 if applicable)

**Task:** Stop qwen35; download model.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose stop qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'HF_HOME=/home/davistroy/.cache/huggingface python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"zai-org/GLM-4.7-Flash\")"'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'du -sh /home/davistroy/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash'
```

**Acceptance criteria:** ~30 GB on disk; safetensors file count matches manifest.

### Work Item C.4 — Run GLM-4.7-Flash full suite

**Status:** Pending
**Depends on:** C.3

**Task:** Create `/home/claude/llm-eval/profiles/glm47.env`:
```
LLM_IMAGE=vllm-cu132-test:glm47        # or vllm-cu132-test:latest if C.1 = MLA_PRESENT
LLM_MODEL=zai-org/GLM-4.7-Flash
LLM_QUANTIZATION=fp8
LLM_KV_DTYPE=auto
LLM_TOOL_PARSER=glm47
LLM_REASONING_PARSER=glm45
LLM_SPEC_TOKENS=
LLM_GPU_UTIL=0.70
LLM_MAX_BATCHED_TOKENS=16384
LLM_EXTRA_ARGS=--attention-backend triton --speculative-draft-attention-backend triton --enable-prefix-caching
```

Run `./scripts/run_full_suite.sh glm47`.

**Verification point 6 (VP6):**
- KV cache size at 128K ≤ 10 GB (MLA active; failure mode is 29 GB)
- c=1 throughput ≥ 35 tok/s (research projects 37-40)
- Tool-call JSON validates in 3 sanity probes
- No visual artifacts in 30+ token responses at contexts 1K/10K/100K

**Acceptance criteria:** Full suite completes; MLA-active flag confirmed in vLLM startup logs; all VP6 sub-criteria met OR explicitly documented failure mode.

### Work Item C.5 — Restore current production state

**Status:** Pending
**Depends on:** C.4

**Task:** Identical to B.5.

**Verification point 7 (VP7):** Production restored; consumer health checks pass.

**Acceptance criteria:** Same as B.5.

### Work Item C.6 — LAB_NOTEBOOK entry: GLM-4.7-Flash evaluation

**Status:** Pending
**Depends on:** C.5

**Task:** LAB_NOTEBOOK entry covering:
- MLA whitelist patch status (pre-existing or built)
- Measured throughput vs research projection
- Tool-call format compatibility notes (for Phase E follow-up flagging)
- Driver 580.142 artifact assessment

**Acceptance criteria:** Entry committed; MEMORY.md indexed.

---

## Phase D: Qwen/Qwen3-Coder-Next-FP8 Evaluation

**Goal:** Evaluate the largest model (~82 GB, 80B/3B hybrid GDN+Gated Attention). Validates research claim of stable FP8 KV cache; documents non-thinking-model behavior on AR-1; measures throughput at gpu_util=0.80 with auxiliary services stopped.

**Estimated time:** ~10 hours.

### Work Item D.1 — Verify qwen3_next architecture support

**Status:** Pending
**Depends on:** C.6

**Task:** Inspect cu132 image for `Qwen3NextForCausalLM` model class.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker run --rm --entrypoint python3 vllm-cu132-test:latest -c "
from vllm.model_executor.models import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print(\"QWEN3_NEXT_PRESENT\" if \"Qwen3NextForCausalLM\" in archs else \"QWEN3_NEXT_ABSENT\")
print(f\"vllm={__import__(\\\"vllm\\\").__version__}\")
"'
```

**Decision branch:**
- `QWEN3_NEXT_PRESENT`: skip D.2; use `vllm-cu132-test:latest`
- `QWEN3_NEXT_ABSENT`: proceed to D.2

**Acceptance criteria:** Output is unambiguously one of the two markers; vLLM version logged.

### Work Item D.2 — Provision newer vLLM image for qwen3_next (conditional)

**Status:** Pending (skip if D.1 = QWEN3_NEXT_PRESENT)
**Depends on:** D.1

**Task:** Two options, pick one:
- Option A: `docker pull vllm/vllm-openai:cu130-nightly` (research-recommended; accepts Triton cache pollution risk — use separate cache dir `/home/claude/.cache/triton-cu130-nightly`)
- Option B: build derived `vllm-cu132-test:coder-next` from latest vLLM source (riskier; may break MTP for other models)

**Recommendation:** Option A. The new image is used for Phase D only; cu132 image preserved untouched for production restoration.

**Acceptance criteria:** Image present; pre-flight test confirms `Qwen3NextForCausalLM` registered.

### Work Item D.3 — Stop auxiliary services (bge-m3, gliner)

**Status:** Pending
**Depends on:** D.1 (and D.2 if applicable)

**Task:** Per Phase 2 analysis and user confirmation, temporarily stop bge-m3 and gliner during Coder-Next test slot to free GPU memory for the 82 GB model at gpu_util=0.80.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose stop bge-m3 gliner'
```

**Note:** Keep chromadb and neo4j running (CPU-only, no GPU contention). qwen3-embed remains running (needed for retire ETL during this window). If embed contention occurs, document and stop embed too.

**Acceptance criteria:** bge-m3 and gliner show `Exited`; embed still healthy; GPU memory baseline recorded.

### Work Item D.4 — Download Qwen/Qwen3-Coder-Next-FP8

**Status:** Pending
**Depends on:** D.3

**Task:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose stop qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'HF_HOME=/home/davistroy/.cache/huggingface python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-Coder-Next-FP8\")"'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'du -sh /home/davistroy/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next-FP8'
```

**Acceptance criteria:** ~82 GB on disk; all safetensors present.

### Work Item D.5 — Run Coder-Next full suite at gpu_util=0.80

**Status:** Pending
**Depends on:** D.4

**Task:** Create `/home/claude/llm-eval/profiles/coder_next_u80.env`:
```
LLM_IMAGE=vllm-cu132-test:latest      # or vllm/vllm-openai:cu130-nightly if D.2 applicable
LLM_MODEL=Qwen/Qwen3-Coder-Next-FP8
LLM_QUANTIZATION=
LLM_KV_DTYPE=fp8
LLM_TOOL_PARSER=qwen3_coder
LLM_REASONING_PARSER=
LLM_SPEC_TOKENS=
LLM_GPU_UTIL=0.80
LLM_MAX_BATCHED_TOKENS=16384
LLM_EXTRA_ARGS=--attention-backend flashinfer --enable-prefix-caching --load-format fastsafetensors
```

**Critical:** Do NOT enable `--calculate-kv-scales` (per vllm#37554 corruption risk on hybrid GDN+attention).

Run `./scripts/run_full_suite.sh coder_next_u80`.

**Verification point 8 (VP8):**
- Cold start ≤ 12 min
- c=1 throughput ≥ 35 tok/s (research projects 43)
- AR-2/3/4 task pass rate ≥ AR-1 pass rate (research claims this model wins on coding loops)
- 4h soak: zero OOM, zero crashes
- FP8 KV cache stability: no quality drift over 4h window

**Acceptance criteria:** Full suite completes; all VP8 sub-criteria met OR documented failure.

### Work Item D.6 — Sub-experiment: Coder-Next at gpu_util=0.70 with full stack

**Status:** Pending
**Depends on:** D.5

**Task:** Restart bge-m3 and gliner. Run abbreviated suite (throughput c1/c4/c8 + 1h c=4 soak only, no quality re-run) at `LLM_GPU_UTIL=0.70` to determine whether Coder-Next is viable concurrent with full stack.

**Profile:** Copy `coder_next_u80.env` to `coder_next_u70.env`; change `LLM_GPU_UTIL=0.70` only.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose stop qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose up -d bge-m3 gliner'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'cd /home/claude/llm-eval && ./scripts/run_full_suite.sh coder_next_u70 --abbreviated'
```

**Verification point 9 (VP9):** Determines if Coder-Next can be swap-deployed without stopping auxiliary services; informs Phase E production decision.

**Acceptance criteria:** Run completes OR OOM is documented with telemetry; throughput delta vs D.5 (0.80 util) recorded.

### Work Item D.7 — Restore current production state

**Status:** Pending
**Depends on:** D.6

**Task:** Stop and remove qwen35; verify bge-m3 + gliner + embed all running; redeploy qwen35 with default `.env`.

**SSH commands:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose stop qwen35 && docker compose rm -f qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose ps'  # verify aux services state
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose up -d qwen35 bge-m3 gliner'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'timeout 900 bash -c "until docker compose ps | grep -E \"qwen35.*healthy\"; do sleep 10; done"'
```

**Verification point 10 (VP10):** All 8 production services healthy; spot-check throughput at c=1 within ±10% of baseline 65.9 tok/s.

**Acceptance criteria:** Full production stack restored; comparative 10-min spot-check passes.

### Work Item D.8 — LAB_NOTEBOOK entry: Qwen3-Coder-Next-FP8 evaluation

**Status:** Pending
**Depends on:** D.7

**Task:** LAB_NOTEBOOK entry covering:
- Architecture support status (cu132 native or required image swap)
- gpu_util=0.80 vs 0.70 throughput tradeoff
- Non-thinking-model AR-1 behavior (research warned about losing `<think>` trace)
- FP8 KV cache stability assessment (research claim validation)
- Auxiliary service impact (downtime cost during swap window)

**Acceptance criteria:** Entry committed; MEMORY.md indexed.

---

## Phase E: Comparative Synthesis & Production Decision

**Goal:** Produce comparative report, update project memory, decide on production change (or no-change).

**Estimated time:** ~4 hours.

### Work Item E.1 — Aggregate results into comparison matrix

**Status:** Pending
**Depends on:** D.8

**Task:** Consolidate metrics from all `llm-eval/results/` JSONs into a single comparison table. Required dimensions:
- Throughput: c=1, c=4, c=8, c=16 (aggregate and per-request)
- Latency: p50, p99 per concurrency
- Quality: SWE-bench subset, GPQA-D probes, AR-1/2/3/4 pass rates
- Stability: 4h soak success rate, GPU memory peak/drift, MTP acceptance
- Operational: cold start time, image patches required, KV-dtype tested, memory budget required
- 7-axis research rubric: re-scored with measured-on-Spark numbers

**Output:** `/home/davistroy/dev/personal/spark/MODEL_EVALUATION_2026_05.md` — markdown report with comparison matrix, per-model deep-dive, weighted scoring.

**Acceptance criteria:** Report exists; every metric in metrics_schema.json appears for every model (or N/A documented).

### Work Item E.2 — Write per-model and cross-cutting LAB_NOTEBOOK summary

**Status:** Pending
**Depends on:** E.1

**Task:** Final LAB_NOTEBOOK entry: "Autoresearch LLM Selection Study — Comparative Findings (2026-05-XX)". Includes:
- Pre-quant FP8 verdict (validates/invalidates Entry 054-055 under v0.19.1rc1)
- FP8 KV cache verdict for Qwen3.6 (validates/invalidates research gotcha #1)
- MTP=1 vs MTP=2 verdict
- GLM-4.7-Flash production-readiness assessment
- Qwen3-Coder-Next-FP8 production-readiness assessment
- Per-workload recommendation (AR-1, AR-2/3/4, throughput-bound)
- Recommended production action: keep status quo / replace baseline / workload-route

**Acceptance criteria:** Entry committed; MEMORY.md indexed with one-line entry per finding.

### Work Item E.3 — Update SPARK_BASELINE.md with measured comparison

**Status:** Pending
**Depends on:** E.1

**Task:** Update `/home/davistroy/dev/personal/spark/SPARK_BASELINE.md` model comparison section with measured-on-Spark numbers from the study. Replace any vendor-self-reported numbers with our measurements.

**Acceptance criteria:** Section updated; cross-link to MODEL_EVALUATION_2026_05.md.

### Work Item E.4 — Production decision and (conditional) docker-compose update

**Status:** Pending
**Depends on:** E.2

**Task:** Based on E.2 verdict, either:
- **No-change**: document decision in MODEL_EVALUATION_2026_05.md; leave `docker-compose.yml` reverted to pre-eval-study state; close plan
- **Replace baseline**: update default `.env` to point at the winning model; redeploy `qwen35` service; run 30-min validation soak; update spark-device.md memory
- **Workload-route (multi-profile)**: keep parameterized compose; document the per-workload profile choice; flag downstream consumer code changes as follow-up plan

**SSH commands for "replace baseline" case:**
```bash
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'cp /home/claude/docker-compose.yml /home/claude/docker-compose.yml.pre-baseline-change'
# Update .env to winning profile
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'docker compose up -d qwen35'
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net 'cd /home/claude/llm-eval && ./scripts/run_full_suite.sh new_baseline --abbreviated-soak'
```

**Verification point 11 (VP11) — only for "replace baseline" or "workload-route":** 30-min validation soak meets or exceeds 2026-05-10 reference (12.06 tok/s, 100% success, 43.1s mean latency).

**Acceptance criteria:**
- No-change: report documents rationale
- Change: post-change soak passes; spark-device.md updated; MEMORY.md indexed

### Work Item E.5 — Watchlist update

**Status:** Pending
**Depends on:** E.2

**Task:** Based on study findings, update the 60-day watchlist from the original research output. Promote/demote based on what we learned. Watchlist items:
- Qwen3.6-Coder-A3B (rumored coder-specialized variant)
- DeepSeek V4-Coder (multi-GPU; watch for sub-variants)
- GLM-4.8 / GLM-5.1-Air
- Nemotron-3-Nano (long-context option)
- Resolution of vllm-project/vllm#37554

**Output:** Append section to `MODEL_EVALUATION_2026_05.md` titled "60-day watchlist (post-evaluation refresh)".

**Acceptance criteria:** Section written; explicit "re-evaluate trigger" criteria per watchlist item.

### Work Item E.6 — Commit and tag

**Status:** Pending
**Depends on:** E.4, E.5

**Task:** Stage and commit changes to spark repo: `MODEL_EVALUATION_2026_05.md`, updated `SPARK_BASELINE.md`, `LAB_NOTEBOOK.md`, `MEMORY.md`, this `IMPLEMENTATION_PLAN_MODEL_EVAL.md` with all items checked. Commit message format per repo convention (see prior commits 6e51578, 7018e74). Optional: tag `eval-study-2026-05`.

**Acceptance criteria:** Commit pushed; PR opened if main is protected (current pattern: PR per major plan).

---

## Verification Checkpoints

Summary table for quick reference:

| VP | After Work Item | Checks |
|----|-----------------|--------|
| VP1 | A.8 | Harness parity gate (±10% of 2026-05-10 baseline) |
| VP2 | B.2 | Qwen3.6-FP8 pre-quant BF16-KV cold start + c=1 |
| VP3 | B.3 | FP8-KV vs BF16-KV AR-1 quality delta documented |
| VP4 | B.4 | MTP=1 vs MTP=2 acceptance and throughput delta |
| VP5 | B.5 | Production restored after Phase B |
| VP6 | C.4 | GLM MLA active, throughput ≥35, tool-call JSON valid, no artifacts |
| VP7 | C.5 | Production restored after Phase C |
| VP8 | D.5 | Coder-Next at u=0.80 (cold start ≤12 min, c=1 ≥35, 4h soak passes) |
| VP9 | D.6 | Coder-Next at u=0.70 with full stack viability |
| VP10 | D.7 | Production restored after Phase D (full 8-service stack) |
| VP11 | E.4 | (Conditional) Post-baseline-change 30-min validation soak |

---

## Post-Plan Actions (Follow-up scope)

Items intentionally NOT in this plan but flagged for separate consideration:

1. **Downstream consumer tool-call audit** — required only if GLM-4.7-Flash is adopted for any workload. Audit `/home/claude/ce-service/`, retire codebase, contact-center-lab consumers that issue tool-call requests against `spark-llm`.
2. **Workload routing infrastructure** — required only if Phase E concludes multi-model adoption. Design needed: how does the calling code select between Qwen3.6 (AR-1) and Coder-Next (AR-2/3/4) endpoints? Single endpoint with internal routing? Multiple endpoints with client-side selection?
3. **Driver/firmware reassessment** — last firmware update at Entry 050; check for newer firmware that might improve GLM-4.7 artifact behavior.
4. **vllm-project/vllm#37554 resolution watch** — when this bug closes, re-test Qwen3.6 FP8 KV cache; could unlock the current production config from research's gotcha #1.
5. **Honorable-mention model evaluation** — Nemotron-3-Nano (long-context), Devstral Small 2 (dense fallback) — only if Phase E suggests current pool insufficient.

---

## Summary of Outcomes (to be filled in post-execution)

### Key Results

(populated as items complete)

### Final Production State

(populated by E.4)

### Decision Record

(populated by E.2)
