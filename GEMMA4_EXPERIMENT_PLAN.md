# Gemma 4 vs Qwen3.5 — A/B Experiment Plan

**Date:** 2026-04-03
**Author:** Troy Davis + Claude
**Hardware:** DGX Spark GB10 SM121, 128 GB LPDDR5x, Driver 580.142, CUDA 13.0
**Execution:** Dedicated maintenance window (all production containers stopped)
**Baseline:** Qwen3.5-35B-A3B FP8, vllm-custom:sm121-inject, 48.6 tok/s single-request, 311.7 tok/s c16 aggregate

---

## Objective

Evaluate whether Gemma 4 models offer a compelling replacement or complement to the current Qwen3.5-35B-A3B configuration on a single DGX Spark. Test both the 26B-A4B MoE (throughput contender) and the 31B dense (quality contender) across throughput, output quality, tool calling, and multimodal dimensions.

Two evaluation tiers:
1. **Pipeline-specific** — can Gemma 4 serve as a drop-in replacement for the contact-center-lab pipeline (entity extraction, structured JSON, tool calling, 15-18 concurrent requests)?
2. **General capability** — how does Gemma 4 compare on reasoning, coding, long context, and multimodal tasks?

---

## Models Under Test

| Model | Architecture | Total Params | Active/Token | Disk (BF16) | HF Handle |
|-------|-------------|-------------|-------------|-------------|-----------|
| **Qwen3.5-35B-A3B** (control) | MoE | 35B | ~3B | ~35 GB (FP8 on-the-fly) | `Qwen/Qwen3.5-35B-A3B` |
| **Gemma 4 26B-A4B** | MoE (128 experts, 8+1 active) | 26B | 3.8B | ~49 GB | `google/gemma-4-26B-A4B-it` |
| **Gemma 4 31B** | Dense | 31B | 31B | ~62 GB | `google/gemma-4-31B-it` |
| **Gemma 4 31B NVFP4** | Dense, quantized | 31B | 31B | ~20 GB | `nvidia/Gemma-4-31B-IT-NVFP4` |

---

## Known Risks and Constraints

| Risk | Impact | Mitigation |
|------|--------|------------|
| **NVFP4 on SM121** — SM121 lacks `cvt.e2m1x2` microscaling instruction. NVFP4 may fail to load or produce incorrect output. | Exp 5 blocked | Fall back to AWQ int4 (`cyankiwi/gemma-4-31B-it-AWQ-4bit`, ~20 GB) |
| **Tool calling bug** — `Gemma4ToolParser.__init__()` takes wrong args (vLLM #38837). Fixed in later builds but may not be in `gemma4-cu130` image. | Exp 7 tool calling tests fail | Test tool calling early. If broken, skip tool tests for Gemma or use eugr's build which may include the fix. |
| **Power delivery throttling** — Spark silently throttles performance after extended operation. Multiple forum confirmations. | All throughput numbers unreliable | Power cycle (unplug USB-C + brick, 30s, reconnect) before ALL benchmark runs. |
| **Day-1 image maturity** — `gemma4-cu130` released April 2. No SM121 native kernels, no community optimization pass yet. | Gemma throughput numbers are floor, not ceiling | Note this in results. Re-evaluate in 2-3 weeks as community optimizes. |
| **Marlin FP8 MoE compatibility** — `VLLM_TEST_FORCE_FP8_MARLIN=1` is proven for Qwen's MoE (64 experts). Gemma uses 128 experts with different routing. May not apply. | Exp 3 FP8 path may underperform | Test with and without the env var. Document which backend vLLM selects. |

---

## Phase 0: Pre-Staging (Before Maintenance Window)

**Goal:** Download all model weights, pull Docker images, and prepare benchmark scripts so experiment time is spent testing, not waiting.

**Duration:** 2-4 hours (download-bound, can run in background during normal operations)

### 0A: Pull Docker images

```bash
# On Spark:
docker pull vllm/vllm-openai:gemma4-cu130
```

**Verify ARM64 resolution:**
```bash
docker inspect vllm/vllm-openai:gemma4-cu130 | grep Architecture
# Expect: "aarch64"
```

### 0B: Download model weights

All models download to `/home/davistroy/.cache/huggingface` (the shared HF cache).

```bash
# On Spark — run in tmux/screen so downloads survive SSH disconnects:

# Gemma 4 26B-A4B (~49 GB)
huggingface-cli download google/gemma-4-26B-A4B-it

# Gemma 4 31B (~62 GB)
huggingface-cli download google/gemma-4-31B-it

# Gemma 4 31B NVFP4 (~20 GB)
huggingface-cli download nvidia/Gemma-4-31B-IT-NVFP4

# AWQ int4 fallback (~20 GB) — only if NVFP4 fails later
# huggingface-cli download cyankiwi/gemma-4-31B-it-AWQ-4bit

# FP8 pre-quantized variant (~25 GB est.)
# huggingface-cli download protoLabsAI/gemma-4-26B-A4B-it-FP8
```

**Note:** Gemma 4 requires accepting Google's license on HuggingFace. Ensure the HF token on Spark has accepted the license for all google/* models before downloading.

### 0C: Clone/update eugr's spark-vllm-docker

```bash
cd /home/claude
git clone https://github.com/eugr/spark-vllm-docker 2>/dev/null || \
  (cd spark-vllm-docker && git pull)
```

Verify `gemma4-26b-a4b` recipe exists:
```bash
ls spark-vllm-docker/recipes/ | grep -i gemma
```

### 0D: Prepare benchmark script

Create a standardized benchmark script that runs the same tests against any model endpoint. Saves results to timestamped JSON for comparison.

```bash
# Benchmark parameters (consistent across all models):
SINGLE_REQUEST_TOKENS=600
CONCURRENT_LEVELS="1 4 8 16"
PROMPT_LENGTHS="128 512 2048"
WARMUP_REQUESTS=3
BENCHMARK_REQUESTS=5  # per configuration
```

### 0E: Prepare quality test prompts

Write the prompt suite to files so they're identical across all models. Two tiers:

**Pipeline-specific prompts** (save to `benchmark/pipeline/`):
- `p1_entity_extraction.json` — real pipeline input, expect structured JSON output
- `p2_json_schema.json` — complex nested JSON with specific schema
- `p3_tool_call.json` — function-call formatted request matching pipeline tools
- `p4_concurrent_mixed.json` — 16 mixed requests simulating production load

**General capability prompts** (save to `benchmark/general/`):
- `g1_reasoning.json` — multi-step logic problem
- `g2_coding.json` — Python function with edge cases
- `g3_long_context.json` — 4K+ token input, question about details near end
- `g4_multimodal.json` — image description task (Gemma only)
- `g5_instruction_following.json` — precise formatting instructions

### 0F: Disk space check

```bash
df -h /home/davistroy/.cache/huggingface
# Verify at least 200 GB free (current models ~51 GB + new ~131 GB + headroom)
```

---

## Phase 1: Baseline Re-Measurement (Start of Maintenance Window)

**Goal:** Establish clean Qwen3.5 baseline after power cycle. All subsequent comparisons reference these numbers.

**Duration:** ~30 minutes

### 1A: Power cycle the Spark

1. SSH in, verify no active pipeline runs
2. `sudo shutdown -h now`
3. Physically unplug USB-C cable AND power brick from wall
4. Wait 30 seconds
5. Reconnect power, wait for boot
6. SSH back in, verify all services healthy

### 1B: Start Qwen3.5 (control) with current production config

```bash
docker run -d \
  --name qwen35 \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton:/root/.triton \
  vllm-custom:sm121-inject \
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
```

Wait for `/health` 200 (expect ~90s for model load).

### 1C: Run full benchmark suite against Qwen3.5

Record:
- Single-request decode tok/s (3 warmup + 5 measured, max_tokens=600)
- TTFT at pp128, pp512, pp2048
- Concurrent throughput at c1, c4, c8, c16
- GPU memory (`nvidia-smi`)
- System memory and swap (`free -h`)

### 1D: Run quality test suite against Qwen3.5

Run all pipeline-specific and general capability prompts. Save raw outputs to `results/qwen35/` for later comparison.

### 1E: Stop Qwen3.5

```bash
docker stop qwen35 && docker rm qwen35
```

Verify GPU memory is fully released:
```bash
nvidia-smi
# Expect: 0 MiB used (or minimal driver overhead)
```

---

## Phase 2: Gemma 4 26B-A4B (MoE) — Throughput Experiments

**Goal:** Find the highest achievable throughput for the 26B MoE on this Spark.

**Duration:** ~90 minutes

### Exp 2A: BF16 Baseline (Official Image)

```bash
docker run -d \
  --name gemma4-26b \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4-cu130 \
  google/gemma-4-26B-A4B-it \
    --served-model-name gemma4-26b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --load-format safetensors
```

**Note:** Omit `--reasoning-parser` and `--tool-call-parser` for initial throughput test — add them in Exp 4 (tool calling may crash due to #38837).

**Measure:** Single-request tok/s, TTFT at pp128/512/2048, GPU memory.

**Expected:** ~23-25 tok/s (matching WilliamD's day-1 numbers). This is the floor.

```bash
docker stop gemma4-26b && docker rm gemma4-26b
```

### Exp 2B: FP8 On-The-Fly Quantization

Same as 2A, add `--quantization fp8`:

```bash
docker run -d \
  --name gemma4-26b-fp8 \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4-cu130 \
  google/gemma-4-26B-A4B-it \
    --served-model-name gemma4-26b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --quantization fp8 \
    --load-format safetensors
```

**Key question:** Does FP8 give Gemma the same 3.65x jump it gave Qwen? If so, expect ~85-90 tok/s. More likely a smaller gain due to 27% more active params.

If this works, also test with Marlin forcing:
```bash
# Add env var:
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
```

**Record:** Which MoE backend vLLM selects (check logs for "Using ... backend for MoE").

```bash
docker stop gemma4-26b-fp8 && docker rm gemma4-26b-fp8
```

### Exp 2C: eugr's spark-vllm-docker Recipe

```bash
cd /home/claude/spark-vllm-docker
./run-recipe.sh gemma4-26b-a4b --solo
```

**Purpose:** Native SM121 kernels + community optimization. eugr's builds achieve 52.85 tok/s on Qwen vs our 48.6 (8% faster). If the same optimization applies to Gemma 4, this is the production-ready path.

**Measure:** Same benchmark suite. Compare against Exp 2A/2B to quantify the kernel gap.

```bash
# Cleanup per eugr's conventions (check recipe docs)
```

### Phase 2 Decision Gate

| Result | Next Step |
|--------|-----------|
| Best Gemma 26B tok/s > 40 | Proceed to Phase 3 (31B) and Phase 4 (quality) |
| Best Gemma 26B tok/s 25-40 | Proceed, but note this is day-1 and will improve |
| Best Gemma 26B tok/s < 25 | Skip concurrency tests, proceed to Phase 3 and Phase 4 |

---

## Phase 3: Gemma 4 31B (Dense) — Throughput Experiments

**Goal:** Determine the fastest usable configuration for the dense 31B model.

**Duration:** ~60 minutes

### Exp 3A: NVFP4 (NVIDIA Native Quantization)

```bash
docker run -d \
  --name gemma4-31b-nvfp4 \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4-cu130 \
  nvidia/Gemma-4-31B-IT-NVFP4 \
    --served-model-name gemma4-31b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --load-format safetensors
```

**Critical risk:** SM121 may not support NVFP4 inference kernels. If the container crashes or produces NaN/garbage output, log the exact error and proceed to Exp 3B.

**Expected (if it works):** Somewhere between AWQ int4 (10.6 tok/s) and AWQ int8 (6.5 tok/s) — NVFP4 is 4-bit but with NVIDIA's native format, which may be faster than generic AWQ on Blackwell.

```bash
docker stop gemma4-31b-nvfp4 && docker rm gemma4-31b-nvfp4
```

### Exp 3B: AWQ Int4 (Fallback if NVFP4 Fails)

Only run if Exp 3A fails or produces incorrect output.

```bash
docker run -d \
  --name gemma4-31b-awq4 \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4-cu130 \
  cyankiwi/gemma-4-31B-it-AWQ-4bit \
    --served-model-name gemma4-31b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --load-format safetensors
```

**Expected:** ~10.6 tok/s (WilliamD's number).

```bash
docker stop gemma4-31b-awq4 && docker rm gemma4-31b-awq4
```

### Exp 3C: BF16 (Sanity Check Only)

Quick 5-minute test to confirm WilliamD's 3.7 tok/s. No full benchmark — just a single-request measurement. If it matches, skip further BF16 testing.

```bash
docker run -d \
  --name gemma4-31b-bf16 \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4-cu130 \
  google/gemma-4-31B-it \
    --served-model-name gemma4-31b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --load-format safetensors
```

**Expected:** ~3.7 tok/s. Not usable for interactive work. Proceed immediately.

```bash
docker stop gemma4-31b-bf16 && docker rm gemma4-31b-bf16
```

### Phase 3 Decision Gate

| Result | Interpretation |
|--------|---------------|
| NVFP4 works, >15 tok/s | Viable for batch/offline quality tasks |
| NVFP4 fails, AWQ int4 ~10 tok/s | Marginal — only worth it if quality is substantially better |
| All dense configs <8 tok/s | 31B not viable on single Spark. Note for dual-Spark future. |

---

## Phase 4: Quality A/B Testing

**Goal:** Compare output quality across all three models using identical prompts. This is the most important phase — throughput doesn't matter if quality regresses.

**Duration:** ~60 minutes (run best config for each model)

### Setup

Run each model one at a time using the fastest viable configuration identified in Phases 2-3:
- **Qwen3.5:** `vllm-custom:sm121-inject`, FP8, full production config (from Phase 1)
- **Gemma 4 26B:** Best config from Phase 2 (likely Exp 2C eugr's recipe, or 2B FP8)
- **Gemma 4 31B:** Best config from Phase 3 (likely NVFP4 or AWQ int4)

### Tier 1: Pipeline-Specific Tests

These test whether Gemma 4 can replace Qwen3.5 in the contact-center-lab pipeline.

| Test ID | Task | Input | Expected Output | Scoring |
|---------|------|-------|-----------------|---------|
| P1 | Entity extraction | Real pipeline transcript (~500 tokens) | JSON with entity types, spans, confidence | Precision/recall of entities, JSON validity |
| P2 | Complex JSON schema | Nested object with arrays, enums, optional fields | Valid JSON matching schema exactly | Schema validation pass/fail, field completeness |
| P3 | Tool calling | Function-call request with 3 available tools | Correct tool selected, parameters populated | Tool name correct, param types correct, no hallucinated params |
| P4 | Concurrent mixed load | 16 simultaneous requests (mix of P1-P3) | All complete without errors or degradation | Error rate, p50/p95 latency, output quality under load |

**Tool calling note:** If `--tool-call-parser gemma4` crashes due to #38837, run P3 and P4 without tool calling for Gemma models. Document the gap — tool calling maturity is a scoring criterion.

### Tier 2: General Capability Tests

| Test ID | Task | Scoring |
|---------|------|---------|
| G1 | Multi-step reasoning (math word problem with 4+ steps) | Correct final answer, clear chain-of-thought |
| G2 | Code generation (Python: parse CSV, handle edge cases, return typed dict) | Runs correctly, handles edge cases, type-safe |
| G3 | Long context (4K+ token input, question about a detail in the final 500 tokens) | Correct answer, no hallucination, cites relevant section |
| G4 | Multimodal — image description (Gemma only) | Accuracy of description, detail level |
| G5 | Instruction following (output exactly 3 bullet points, each under 15 words, no preamble) | Exact format compliance |

### Scoring Method

Each test scored 0-3:
- **3** — Perfect or near-perfect output
- **2** — Correct with minor issues (formatting, extra text, slight imprecision)
- **1** — Partially correct (right structure, wrong content, or vice versa)
- **0** — Failed (wrong answer, invalid JSON, crashed, refused)

Record raw outputs verbatim in `results/{model_name}/` for review.

---

## Phase 5: Tool Calling Deep Dive

**Goal:** Determine if Gemma 4 can handle the pipeline's tool calling requirements.

**Duration:** ~20 minutes

### Exp 5A: Tool Calling Smoke Test

Start the best Gemma 26B config with tool calling enabled:

```bash
# Add to docker run:
    --reasoning-parser gemma4 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4
```

If `Gemma4ToolParser.__init__()` error occurs:
1. Log the exact error
2. Check if eugr's build includes the #38837 fix
3. If no fix available, score Gemma tool calling as **blocked** in the decision matrix

### Exp 5B: Tool Calling Accuracy (if 5A passes)

Run 10 tool-calling requests with varying complexity:
- Single tool, simple params
- Single tool, complex nested params
- Multi-tool selection (choose from 3 tools)
- Tool + reasoning (explain why you chose this tool)

Compare format and accuracy against Qwen3.5's `qwen3_coder` parser output.

---

## Phase 6: Concurrency Profile

**Goal:** Test the winning Gemma config under production-like concurrent load.

**Duration:** ~30 minutes

Run only the best-performing Gemma model from Phase 2 or 3.

| Concurrency | Measurements |
|-------------|-------------|
| 1 | Single-request decode tok/s, TTFT |
| 4 | Aggregate tok/s, per-request tok/s, TTFT p50/p95 |
| 8 | Aggregate tok/s, per-request tok/s, TTFT p50/p95 |
| 16 | Aggregate tok/s, per-request tok/s, TTFT p50/p95, preemptions |

Compare directly against Qwen3.5's concurrency curve from Phase 1.

**Key question:** Gemma 26B activates 3.8B params/token vs Qwen's ~3B (27% more). At high concurrency, this means proportionally more memory bandwidth contention. Does the per-request tok/s degrade faster than Qwen under load?

---

## Phase 7: Restore Production Config

**Goal:** Return the Spark to production state.

1. Stop all test containers
2. Verify GPU memory fully released (`nvidia-smi`)
3. Start production containers in order: qwen35 → qwen3-embed → gliner (with health check waits between each)
4. Verify all endpoints healthy
5. Run one quick inference request against each endpoint

---

## Results Template

### Throughput Summary

| Config | Single tok/s | c4 Agg | c8 Agg | c16 Agg | GPU Mem | TTFT pp128 | TTFT pp2048 |
|--------|-------------|--------|--------|---------|---------|------------|-------------|
| Qwen3.5 FP8 (control) | 48.6 | — | — | 311.7 | — | — | — |
| Gemma 26B BF16 | — | — | — | — | — | — | — |
| Gemma 26B FP8 | — | — | — | — | — | — | — |
| Gemma 26B eugr | — | — | — | — | — | — | — |
| Gemma 31B NVFP4 | — | — | — | — | — | — | — |
| Gemma 31B AWQ int4 | — | — | — | — | — | — | — |

### Quality Summary

| Test | Qwen3.5 (0-3) | Gemma 26B (0-3) | Gemma 31B (0-3) | Notes |
|------|---------------|-----------------|-----------------|-------|
| P1 Entity extraction | — | — | — | |
| P2 Complex JSON | — | — | — | |
| P3 Tool calling | — | — | — | |
| P4 Concurrent mixed | — | — | — | |
| G1 Reasoning | — | — | — | |
| G2 Code generation | — | — | — | |
| G3 Long context | — | — | — | |
| G4 Multimodal | N/A | — | — | |
| G5 Instruction following | — | — | — | |

### Decision Matrix

| Criterion | Weight | Qwen3.5 | Gemma 26B MoE | Gemma 31B Dense |
|-----------|--------|---------|---------------|-----------------|
| Single-request tok/s | 25% | — | — | — |
| 16-concurrent aggregate tok/s | 15% | — | — | — |
| JSON/structured output quality | 20% | — | — | — |
| Tool calling reliability | 15% | — | — | — |
| Reasoning quality | 10% | — | — | — |
| Multimodal capability | 5% | 0 | — | — |
| Context window (256K native) | 5% | 32K cfg | 256K | 256K |
| Operational maturity | 5% | high | day-1 | day-1 |
| **Weighted Total** | **100%** | — | — | — |

---

## Possible Outcomes

| Outcome | Action |
|---------|--------|
| **Gemma 26B wins on throughput AND quality** | Replace Qwen3.5 as primary model. Update SPARK_CONFIG.md, container commands, pipeline config. |
| **Gemma 26B ties on quality, slower on throughput** | Keep Qwen3.5. Revisit in 2-3 weeks after community optimization. |
| **Gemma 31B wins on quality, too slow for interactive** | Consider two-model strategy: Qwen3.5 (or Gemma 26B) for interactive, Gemma 31B for batch/offline. Requires scheduling logic. |
| **Gemma 4 multimodal opens new use cases** | Keep Qwen3.5 for text pipeline, evaluate Gemma 26B as a second model for multimodal tasks when pipeline is idle. |
| **All Gemma configs underperform** | Stay on Qwen3.5. Document results for community. Revisit when vLLM Gemma 4 support matures. |

---

## References

- [Gemma 4 Day-1 DGX Spark Benchmarks (WilliamD)](https://forums.developer.nvidia.com/t/gemma-4-day-1-inference-on-nvidia-dgx-spark-preliminary-benchmarks/365503)
- [Gemma 4 vLLM Version Compatibility](https://forums.developer.nvidia.com/t/gemma-4-models-which-vllm-version-any-prs-spotted/365490)
- [NVIDIA Spark vLLM Playbook](https://build.nvidia.com/spark/vllm)
- [vLLM Gemma 4 Blog](https://vllm.ai/blog/gemma4)
- [Google DeepMind Gemma 4](https://deepmind.google/models/gemma/gemma-4/)
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
- [protoLabsAI Gemma 4 26B FP8](https://huggingface.co/protoLabsAI/gemma-4-26B-A4B-it-FP8)
- [Spark Arena Leaderboard](https://spark-arena.com/)
- [vLLM #38837 — Gemma4ToolParser fix](https://github.com/vllm-project/vllm/issues/38837)
