# Implementation Plan: DGX Spark — MTP Validation, Image Eval, Ops Improvements, Rename

**Created:** 2026-04-24
**Branch:** spark-optimization-2026-04-24
**Status:** PENDING

**Context:** Spark-recon Entry 042 (2026-04-24) found multiple community reports that MTP speculative decoding degrades performance on Qwen3.6. Our MTP=2 was validated on Qwen3.5 and carried forward without re-benchmark. Additionally, eugr's spark-vllm-docker has a newer build (0.19.2rc1+cu132 with flashinfer_cutlass re-enabled), gliner memory bloat blocked the gpu_util 0.70 tuning attempt, and the legacy served-model-name `qwen3.5-35b` needs updating.

**Scope:** SSH operations on the DGX Spark + in-repo documentation/config changes. Does NOT cover contact-center-lab consumer updates (separate repo, flagged as follow-up).

**Risk Summary:**

| Phase | Risk | Rollback |
|-------|------|----------|
| 1 (MTP) | Removing MTP may halve c8 throughput | Re-add --speculative-config flag |
| 2 (Image) | New image may regress | docker tag current image before swap |
| 3 (Ops) | gpu_util 0.70 OOM under load | Set back to 0.65, restart |
| 4 (Rename) | Missed consumer → "model not found" | Change name back |

**Execution notes:**
- All SSH commands: `ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net`
- Container restart time: ~6 min (model load + Triton JIT warm)
- Benchmark tool: `benchmarks/throughput_bench.py` (600 max_tokens, 3 runs/level, c1/c4/c8/c16)
- Every experiment → LAB_NOTEBOOK.md entry before proceeding

---

## Phase 1: MTP Validation on Qwen3.6

**Goal:** Determine definitively whether MTP=2 helps or hurts Qwen3.6 on our cu132 build. This gates all subsequent config decisions.

**Current state:** MTP=2 active, acceptance rate 80.7%, benchmarks: c1=51.2, c4=160.8, c8=384.4, c16=576.0 tok/s.

### Work Item 1.1 — Tag current image for rollback ✅ Completed 2026-04-24

**Status:** COMPLETE 2026-04-24

**Task:** Tag the current production image so it can be restored if anything goes wrong during testing.

**SSH commands:**
```bash
docker tag vllm-cu132-test:latest vllm-cu132-test:pre-optimization-2026-04-24
docker images | grep vllm-cu132
```

**Acceptance:** `docker images` shows both `:latest` and `:pre-optimization-2026-04-24` tags pointing to the same image ID.

**Files:** None (remote only)

---

### Work Item 1.2 — Benchmark WITHOUT MTP ✅ Completed 2026-04-24

**Status:** COMPLETE 2026-04-24
**Depends on:** 1.1

**Task:** Stop the production container. Start a new qwen35 container identical to production but WITHOUT `--speculative-config` and `--max-num-batched-tokens 4096`. Run the full throughput benchmark suite.

**SSH commands:**
```bash
# Stop production
docker stop qwen35 && docker rm qwen35

# Start WITHOUT MTP (note: remove --max-num-batched-tokens and --speculative-config)
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
  --entrypoint python3 \
  vllm-cu132-test:latest \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
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

# Wait for health
until curl -sf http://localhost:8000/health; do sleep 10; done

# Run benchmark
python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model qwen3.5-35b --concurrency 1 4 8 16
```

**Acceptance:** Benchmark completes for all concurrency levels. Results recorded.

**Files:** LAB_NOTEBOOK.md (append entry)

---

### Work Item 1.3 — MTP A/B comparison and decision ✅ Completed 2026-04-24

**Status:** COMPLETE 2026-04-24
**Depends on:** 1.2

**Task:** Compare no-MTP results (1.2) against existing MTP results. Make adopt/drop decision using these criteria:

| Scenario | Decision |
|----------|----------|
| No-MTP c1 ≥ MTP c1 AND no-MTP c8 within 80% of MTP c8 | DROP MTP (simpler config, community-aligned) |
| MTP clearly wins c8 by >20% | KEEP MTP |
| Mixed results | Keep MTP — proven throughput at high concurrency |

**Reference baselines (MTP=2, Entry 039):**
- c1: 51.2 tok/s
- c4: 160.8 tok/s (aggregate)
- c8: 384.4 tok/s (aggregate)
- c16: 576.0 tok/s (aggregate)

**Acceptance:** Decision documented with rationale. If MTP dropped: production container restarted without MTP flags and verified healthy. If MTP kept: production container restored with MTP flags.

**Files:** LAB_NOTEBOOK.md (decision entry), SPARK_BASELINE.md (update if config changes), spark-device.md (update container command if config changes)

---

## Phase 2: Image Evaluation (eugr 0.19.2rc1+cu132)

**Goal:** Determine if eugr's newer build yields measurable improvement over our current image. Key differences: FlashInfer 0.6.8, flashinfer_cutlass re-enabled, PR #40191 torch fix.

**Prerequisite:** Phase 1 complete (we know the winning MTP config).

### Work Item 2.1 — Pull eugr's 0.19.2rc1 build ✅ Completed 2026-04-24

**Status:** COMPLETE 2026-04-24
**Depends on:** 1.3

**Task:** Pull the prebuilt vLLM and FlashInfer wheels from eugr's GitHub releases and build a test image, OR pull eugr's Docker image directly if available.

**What was done:** No GHCR image available. Cloned repo, build script auto-downloaded prebuilt wheels from GitHub releases, built runner image (Stage 6 only — no source compilation). Build time: 4:53.

**SSH commands used:**
```bash
git clone https://github.com/eugr/spark-vllm-docker.git /tmp/svd
cd /tmp/svd
bash build-and-copy.sh -t eugr-vllm-0192 --full-log
docker tag eugr-vllm-0192:latest eugr-vllm:test
```

**Result:**
- Image: `eugr-vllm-0192:latest` / `eugr-vllm:test` (same image ID: `83aec1653cd6`)
- Size: 19.3 GB
- vLLM: `0.19.2rc1.dev154+g1c2c1eb8b.d20260423.cu132`
- FlashInfer: 0.6.8 (cubin + jit_cache + python wheels)
- Base: `nvidia/cuda:13.2.0-devel-ubuntu24.04`
- Entrypoint: `/opt/nvidia/nvidia_entrypoint.sh` (needs `--entrypoint python3` override, same as current image)
- Python: 3.12, transformers 5.6.2, torch 2.11.0
- Note: Includes custom NCCL with mesh support (dgxspark-3node-ring), ray, fastsafetensors, instanttensor

**Acceptance:** `docker images | grep eugr` shows both tags. Entrypoint confirmed via `docker inspect`. Production containers untouched.

**Files:** None (remote only)

---

### Work Item 2.2 — Benchmark eugr image ✅ Completed 2026-04-24

**Status:** COMPLETE 2026-04-24
**Depends on:** 2.1

**Task:** Stop production container. Start with eugr's image using the winning MTP config from Phase 1. Benchmark.

**SSH commands:**
```bash
docker stop qwen35 && docker rm qwen35

# Start with eugr image + winning MTP config from Phase 1
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
  --entrypoint python3 \
  vllm-eugr-0192:test \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
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

# Wait + benchmark
until curl -sf http://localhost:8000/health; do sleep 10; done
python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model qwen3.5-35b --concurrency 1 4 8 16
```

**Note:** `--entrypoint python3` IS needed — eugr image uses NVIDIA base entrypoint (`/opt/nvidia/nvidia_entrypoint.sh`), same as our cu132 image. Confirmed via `docker inspect eugr-vllm:test`. Also: eugr image ships with transformers 5.6.2 (vs our image's transformers 4.x) — monitor for any behavioral differences.

**Acceptance:** Benchmark completes. Results compared against Phase 1 winner.

**Files:** LAB_NOTEBOOK.md (append entry)

---

### Work Item 2.3 — Image adopt/reject decision

**Status:** PENDING
**Depends on:** 2.2

**Task:** Compare eugr benchmark results against Phase 1 winner.

| Scenario | Decision |
|----------|----------|
| eugr ≥5% improvement at c1 or c8 | ADOPT eugr image |
| eugr within 5% | STAY on current image (avoid unnecessary change) |
| eugr regresses | REJECT, restore current image |

If ADOPT: update production container to eugr image. Tag appropriately.
If REJECT: restore from `vllm-cu132-test:pre-optimization-2026-04-24`.

**Acceptance:** Decision documented. Production container running on winning image, verified healthy.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md (update image if changed), spark-device.md (update container command if changed), CLAUDE.md (add rule if new image has different entrypoint behavior)

---

## Phase 3: Operational Improvements

**Goal:** Reclaim wasted GPU memory and determine the correct tool-calling parser for Qwen3.6.

### Work Item 3.1 — Restart gliner, verify memory reclamation

**Status:** PENDING

**Task:** Restart the gliner container to reclaim accumulated GPU memory (19.7 GiB → expected ~2 GiB). Prerequisite for retrying gpu_util 0.70.

**SSH commands:**
```bash
# Check current memory before restart
nvidia-smi

# Restart gliner
docker restart gliner

# Wait and verify
sleep 30
nvidia-smi

# Check free GPU memory specifically
python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'Free: {f/1024**3:.1f} GiB / Total: {t/1024**3:.1f} GiB')"
```

**Acceptance:** gliner memory usage drops from ~19.7 GiB to <4 GiB. Available GPU memory increases correspondingly.

**Files:** LAB_NOTEBOOK.md (document before/after)

---

### Work Item 3.2 — Retry gpu_util 0.70

**Status:** PENDING
**Depends on:** 3.1, Phase 2 complete (final image + MTP config known)

**Task:** Restart qwen35 with `--gpu-memory-utilization 0.70`. Now feasible since gliner memory reclaimed.

**SSH commands:**
```bash
docker stop qwen35 && docker rm qwen35

# Start with 0.70 — use final winning config from Phases 1-2
# ONLY change: --gpu-memory-utilization 0.65 → 0.70
[winning docker run command with 0.70]

until curl -sf http://localhost:8000/health; do sleep 10; done

# Quick benchmark
python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model qwen3.5-35b --concurrency 1 8

# Stability test (5 rounds, 1 min apart)
for i in $(seq 1 5); do
  python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model qwen3.5-35b --concurrency 8 --runs 1
  sleep 60
done
```

**Fallback if startup fails:**
```bash
docker rm qwen35
# Restart with 0.65
```

**Alternative if 0.70 fails even after gliner restart:** Try `--gpu-memory-utilization 0.6770` with `-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

**Acceptance:** Container starts at 0.70. num_gpu_blocks increased. 5-round stability test passes.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md (update gpu_util + blocks), spark-device.md (update command)

---

### Work Item 3.3 — Check Qwen3.6 chat template format

**Status:** PENDING

**Task:** Determine whether Qwen3.6's tokenizer_config.json uses XML or JSON format for tool calls. Zero-risk pre-test — no container restart needed.

**SSH commands:**
```bash
python3 -c "
import json, glob
files = glob.glob('/home/davistroy/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/tokenizer_config.json')
tc = json.load(open(files[0]))
tmpl = tc.get('chat_template', 'NOT FOUND')
if '<tool_call>' in tmpl or 'tool_call>' in tmpl:
    print('FORMAT: XML')
elif 'tool_calls' in tmpl:
    print('FORMAT: JSON')
else:
    print('FORMAT: UNKNOWN — manual inspection needed')
print('---')
for line in tmpl.split(chr(10)):
    if 'tool' in line.lower():
        print(line.strip()[:120])
"
```

**Decision:**

| Template Format | Correct Parser | Action |
|----------------|---------------|--------|
| JSON | `qwen3_coder` (current) | No change. Resolve watch item. Skip 3.4. |
| XML | `qwen3_xml` | Proceed to 3.4. |
| Unknown | — | Print full template, inspect manually. |

**Acceptance:** Format determined and documented.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md (resolve watch item if JSON)

---

### Work Item 3.4 — Live test qwen3_xml parser (conditional)

**Status:** PENDING
**Depends on:** 3.3 result = XML
**Skip if:** 3.3 result = JSON

**Task:** During a container cycle, test `qwen3_xml` parser with 10 tool-calling requests.

**SSH commands:**
```bash
# When restarting qwen35 for any reason, temporarily use --tool-call-parser qwen3_xml

# Run 10 tool-call requests
for i in $(seq 1 10); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model":"qwen3.5-35b",
      "messages":[{"role":"user","content":"What is the weather in Boston?"}],
      "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],
      "tool_choice":"auto"
    }' | python3 -c "
import json, sys
r = json.load(sys.stdin)
c = r['choices'][0]
fr = c['finish_reason']
tc = c['message'].get('tool_calls', [])
ok = fr == 'tool_calls' and len(tc) > 0 and tc[0]['function']['name'] == 'get_weather'
print(f'Run $i: finish_reason={fr}, tool_calls={len(tc)}, valid={ok}')
"
done
```

**Acceptance:** ≥9/10 succeed with valid tool calls. If <9/10: revert to `qwen3_coder`.

**Files:** LAB_NOTEBOOK.md, spark-device.md (update if parser changed)

---

## Phase 4: Model Rename

**Goal:** Change `--served-model-name` from `qwen3.5-35b` to `spark-llm` across this repo and on the remote Spark.

**Prerequisite:** Phases 1-3 complete (config stable, no more container restarts expected).

### Work Item 4.1 — Update benchmark script defaults

**Status:** PENDING

**Task:** Change default `--model` argument in both benchmark scripts.

**Changes:**
- `benchmarks/throughput_bench.py` line 99: `default="qwen3.5-35b"` → `default="spark-llm"`
- `benchmarks/quality_test.py` line 8: docstring `--model qwen3.5-35b` → `--model spark-llm`
- `benchmarks/quality_test.py` line 117: `default="qwen3.5-35b"` → `default="spark-llm"`

**Acceptance:** `grep -r "qwen3.5-35b" benchmarks/` returns no results.

**Files:** `benchmarks/throughput_bench.py`, `benchmarks/quality_test.py`

---

### Work Item 4.2 — Update SPARK_CONFIG.md

**Status:** PENDING

**Task:** Update all model name references.

**Changes:**
- Line 88: `qwen3.5-35b` → `spark-llm`
- Line 110: `--served-model-name qwen3.5-35b` → `--served-model-name spark-llm`
- Lines 206, 208: LiteLLM config `qwen3.5-35b` → `spark-llm`
- Line 355: curl example `qwen3.5-35b` → `spark-llm`

**Acceptance:** `grep "qwen3.5-35b" SPARK_CONFIG.md` returns no results.

**Files:** `SPARK_CONFIG.md`

---

### Work Item 4.3 — Update spark-dashboard.json

**Status:** PENDING

**Task:** Replace all 19 Prometheus metric label references.

**Changes:** Global find-and-replace `qwen3.5-35b` → `spark-llm` in all PromQL expressions.

**Acceptance:** `grep -c "qwen3.5-35b" spark-dashboard.json` returns 0. JSON is valid: `python3 -c "import json; json.load(open('spark-dashboard.json'))"`.

**Note:** Per CLAUDE.md — NEVER modify the live Grafana dashboard. Import updated JSON as a new dashboard.

**Files:** `spark-dashboard.json`

---

### Work Item 4.4 — Change served-model-name on remote Spark

**Status:** PENDING
**Depends on:** 4.1, 4.2, 4.3

**Task:** Restart qwen35 with `--served-model-name spark-llm`. Verify API responds.

**SSH commands:**
```bash
docker stop qwen35 && docker rm qwen35

# Restart with --served-model-name spark-llm (all other flags same)
[current winning docker run command with spark-llm]

until curl -sf http://localhost:8000/health; do sleep 10; done

# Verify new name
curl -s http://localhost:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])"
# Expected: spark-llm

# Verify old name fails
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"test"}],"max_tokens":5}'
# Expected: model not found error
```

**Acceptance:** `/v1/models` returns `spark-llm`. Old name returns error.

**Files:** spark-device.md, LAB_NOTEBOOK.md

---

### Work Item 4.5 — Update memory files and baseline

**Status:** PENDING
**Depends on:** 4.4

**Task:** Update spark-device.md container command and any SPARK_BASELINE.md / CLAUDE.md references.

**Files:** spark-device.md, SPARK_BASELINE.md, CLAUDE.md

**Acceptance:** `grep "qwen3.5-35b"` returns no results in any of these files (except historical context).

---

### Work Item 4.6 — Flag contact-center-lab for update

**Status:** PENDING
**Depends on:** 4.4

**Task:** Document required changes in contact-center-lab as follow-up. Do NOT modify the other repo from this plan.

**Consumer changes needed (for reference):**
- `pipeline/config.yaml` lines 24, 30, 77
- `pipeline/tests/unit/test_llm_client.py` lines 63, 98, 126
- `experiments/knowledge-base/augmented_knowledge_extraction_complete.ipynb` lines 336, 364
- `experiments/knowledge-base/servicenow_sampling_and_evaluation.ipynb` line 2011

**Acceptance:** Follow-up documented in LAB_NOTEBOOK.md with file paths and line numbers.

**Files:** LAB_NOTEBOOK.md
