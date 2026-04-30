# Implementation Plan: DGX Spark — Performance Sprint, Infrastructure Hardening, Research Scoping

**Created:** 2026-04-30
**Branch:** main
**Status:** IN_PROGRESS (8/16 items complete — Phase 0 + Phase 1 + Phase 2 done)
**Prior plan:** Archived to `docs/archive/IMPLEMENT_MTP_EUGR_OPS_RENAME-v1.md` (COMPLETE 2026-04-24)

**Context:** Spark-recon Entry 049 (2026-04-30) identified 6 actionable items: firmware just updated (Entry 050, ~6% gain expected), eugr v0.20.1rc1 available (2 minor versions ahead), pre-quant FP8 hang rule invalidated by 3 independent signals, vLLM-Tune kernel tuning reported +9.5% decode. Additionally, infrastructure items from LATER_PLAN remain unfinished: Docker Compose, OS cleanup, data backup. Ultra-plan analysis grouped these into 3 change sets with clear ordering dependencies.

**Scope:** SSH operations on the DGX Spark + in-repo documentation/config changes. Research items (NVFP4/INT4, Gemma 4) are scoped as documentation only — no system changes.

**Exclusions:**
- Contact-center-lab consumer updates (separate repo)
- Grafana dashboard modifications (existing dashboards work)
- Driver upgrade (staying on 580.142 — driver 590 has UMA memory leak)
- Shelly Plug remote power setup (separate project)

**Risk Summary:**

| Phase | Risk | Rollback |
|-------|------|----------|
| 0 (Backup) | Near zero | N/A |
| 1 (Baseline) | Near zero | N/A |
| 2 (eugr eval) | Medium — c8/c16 regression possible (seen in v0.19.2rc1) | Restore tagged production image |
| 3 (Pre-quant FP8) | Medium — silent hang possible (seen on v0.19.0) | Swap back to BF16 model + `--quantization fp8` |
| 4 (vLLM-Tune) | Low — additive config, removable | Remove config volume mount |
| 5 (Infra) | Low — codifying existing behavior | Individual `docker run` commands in spark-device.md |
| 6 (Research) | Zero — documentation only | N/A |

**Execution notes:**
- All SSH commands: `ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net`
- Container restart time: ~6 min (model load + Triton JIT warm)
- Benchmark tool: `benchmarks/throughput_bench.py` (600 max_tokens, 3 runs/level, c1/c4/c8/c16)
- Every experiment → LAB_NOTEBOOK.md entry before proceeding
- Current baseline (pre-firmware, 2026-04-24): c1=59.9, c4=166.2, c8=373.8, c16=564.0 tok/s

---

## Phase 0: Safety Net

**Goal:** Back up non-recoverable data before any container experiments. This phase is a prerequisite for all subsequent work.

### Work Item 0.1 — Create data backup script ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** None

**Task:** Create `/home/claude/backup-data.sh` that backs up ChromaDB and Neo4j Docker volumes to timestamped local directories. Neo4j requires a brief stop for consistent snapshot.

**SSH commands:**
```bash
cat > /home/claude/backup-data.sh << 'SCRIPTEOF'
#!/bin/bash
set -euo pipefail

BACKUP_DIR=/home/claude/backups/$(date +%Y%m%d-%H%M%S)
mkdir -p "$BACKUP_DIR"

echo "=== Backing up ChromaDB ==="
docker run --rm -v chromadb-data:/data -v "$BACKUP_DIR":/backup alpine \
  tar czf /backup/chromadb-data.tar.gz -C /data .

echo "=== Stopping Neo4j for consistent backup ==="
docker stop neo4j
docker run --rm -v neo4j-data:/data -v "$BACKUP_DIR":/backup alpine \
  tar czf /backup/neo4j-data.tar.gz -C /data .
docker run --rm -v neo4j-logs:/data -v "$BACKUP_DIR":/backup alpine \
  tar czf /backup/neo4j-logs.tar.gz -C /data .
docker start neo4j

echo "=== Backup complete ==="
ls -lh "$BACKUP_DIR/"
echo "Total: $(du -sh "$BACKUP_DIR" | cut -f1)"
SCRIPTEOF
chmod +x /home/claude/backup-data.sh
```

**Acceptance:** Script exists and is executable. Dry-read the script to verify volume names match (`chromadb-data`, `neo4j-data`, `neo4j-logs` — confirmed via `docker system df`).

**Files:** None (remote only)

---

### Work Item 0.2 — Run initial backup ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** 0.1

**Task:** Execute the backup script. Verify backup sizes are reasonable (total ~1 GB based on `docker system df` showing 1.068 GB in volumes).

**SSH commands:**
```bash
/home/claude/backup-data.sh

# Verify backup integrity (spot-check)
tar tzf /home/claude/backups/*/chromadb-data.tar.gz | head -5
tar tzf /home/claude/backups/*/neo4j-data.tar.gz | head -5
```

**Acceptance:** Backup completes without error. Total size ~1 GB. Neo4j restarts and returns to healthy state (`curl -sf http://localhost:7474`).

**Files:** LAB_NOTEBOOK.md (append brief entry)

---

## Phase 1: Post-Firmware Baseline

**Goal:** Establish clean throughput baseline after today's firmware update (Entry 050). All subsequent experiments reference these numbers.

**Prerequisite:** Phase 0 complete.

### Work Item 1.1 — Post-firmware benchmark ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** 0.2

**Task:** Run the full throughput benchmark suite against the current production config. The system was just rebooted for firmware recovery — this is the cleanest possible test environment (40C GPU, 0% utilization, 41 min uptime).

**SSH commands:**
```bash
# Verify health
curl -sf http://localhost:8000/health && echo "LLM OK"

# Full benchmark
python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model spark-llm --concurrency 1 4 8 16

# Record GPU state
nvidia-smi
free -h
```

**Expected results:** ~63.5 tok/s c1 (59.9 × 1.06 from firmware gain). c4/c8/c16 should scale proportionally.

**Acceptance:** Benchmark completes for all concurrency levels. Results recorded in LAB_NOTEBOOK.md as Entry 051.

**Files:** LAB_NOTEBOOK.md (Entry 051), SPARK_BASELINE.md (update if numbers change)

---

### Work Item 1.2 — OS cleanup (parallel with 1.1) ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** None (independent of all other items)

**Task:** Remove unused desktop snaps and free RAM. **CRITICAL: Keep `firmware-updater` snap** — just used today for firmware update.

**SSH commands:**
```bash
# Remove desktop snaps (NOT firmware-updater, NOT core/snapd)
sudo snap remove gnome-46-2404
sudo snap remove gtk-common-themes
sudo snap remove mesa-2404

# Verify firmware-updater is still present
snap list | grep firmware

# Check RAM improvement
free -h

# Check disk freed
df -h /
```

**What NOT to remove:**
- `firmware-updater` — needed for future firmware updates (just used today)
- `bare`, `core22`, `core24`, `snapd` — required by snap ecosystem and firmware-updater
- `dgx-dashboard`, `dgx-dashboard-admin` — useful monitoring
- `nvidia-dgx-telemetry` — keep for now (provides system telemetry)
- `avahi-daemon`, `multipathd` — not shown as running in `systemctl` output; skip

**Acceptance:** `snap list` shows firmware-updater still present. gnome-46-2404, gtk-common-themes, mesa-2404 removed. RAM available increases (~100-200 MB).

**Files:** LAB_NOTEBOOK.md (brief entry), CLAUDE.md (add note if relevant)

---

## Phase 2: Image Evaluation (eugr v0.20.1rc1+cu132)

**Goal:** Determine if eugr's latest build yields measurable improvement. Key differences from our v0.19.1rc1: FlashInfer 0.6.9 (up from 0.6.8-ish), experimental b12x support, vLLM v0.20.1rc1. Previous rejection (Entry 045-046) was v0.19.2rc1 — different build, specific KV cache and CUDA graph regressions.

**Prerequisite:** Phase 1 complete (clean post-firmware baseline established).

### Work Item 2.1 — Pull and build eugr v0.20.1rc1 image ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** 1.1

**Task:** Pull the latest eugr/spark-vllm-docker and build the runner image. Tag current production image for rollback.

**SSH commands:**
```bash
# Tag current production image for rollback
docker tag vllm-cu132-test:latest vllm-cu132-test:pre-eugr-v0201

# Update or clone eugr repo
cd /home/claude/spark-vllm-docker && git pull || \
  git clone https://github.com/eugr/spark-vllm-docker.git /home/claude/spark-vllm-docker

# Check what version is available
cat /home/claude/spark-vllm-docker/README.md | head -30

# Build runner image (Stage 6 only — prebuilt wheels, no source compile)
cd /home/claude/spark-vllm-docker
bash build-and-copy.sh -t eugr-vllm-0201 --full-log 2>&1 | tee /tmp/eugr-build.log

# Tag for testing
docker tag eugr-vllm-0201:latest eugr-vllm:v0201-test

# Verify
docker images | grep eugr
```

**Acceptance:** `docker images` shows `eugr-vllm-0201:latest` and `eugr-vllm:v0201-test`. Rollback tag `vllm-cu132-test:pre-eugr-v0201` exists.

**Files:** None (remote only)

---

### Work Item 2.2 — Pre-flight: clean GPU state ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** 2.1

**Task:** Ensure clean GPU state before testing. Entry 045 showed eugr's stricter `request_memory()` check failed when gliner had bloated to 19.7 GiB. Stop gliner, bge-m3, and ce-service before swapping qwen35 to avoid memory contention.

**SSH commands:**
```bash
# Check current GPU state
nvidia-smi

# Stop auxiliary GPU containers
docker stop gliner bge-m3 ce-service
docker rm gliner bge-m3 ce-service

# Verify GPU memory freed
nvidia-smi

# Check for orphan processes holding GPU memory
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
# If orphan PIDs exist: docker run --rm --pid=host --privileged alpine kill -9 <PID>
```

**Acceptance:** Only qwen35 and qwen3-embed show GPU memory usage. No orphan PIDs.

**Files:** None (remote only)

---

### Work Item 2.3 — Benchmark eugr image ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** 2.2

**Task:** Stop production container. Start with eugr image using identical flags. Full c1/c4/c8/c16 benchmark.

**SSH commands:**
```bash
docker stop qwen35 && docker rm qwen35

# Start with eugr image — IDENTICAL FLAGS to production (spark-device.md)
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
  eugr-vllm:v0201-test \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
    --served-model-name spark-llm \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.70 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

# Monitor startup (watch for errors, MoE backend, KV cache tokens)
for i in $(seq 1 60); do
  HEALTH=$(curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "HEALTHY" || echo "loading")
  echo "[$i] $HEALTH"
  if [ "$HEALTH" = "HEALTHY" ]; then break; fi
  sleep 10
done

# Capture startup log details
docker logs qwen35 2>&1 | grep -iE "MoE backend|attention|FP8|cache|graph|version|specul" | head -20

# Full benchmark
python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model spark-llm --concurrency 1 4 8 16
```

**Key metrics to compare against Phase 1 baseline:**
- KV cache token count (production: 1,142,736 — Entry 046 showed eugr v0.19.2rc1 had 929,936, 8.2% fewer)
- CUDA graph mode (production: FULL_AND_PIECEWISE — eugr v0.19.2rc1 had PIECEWISE only)
- MoE backend (expect TRITON; check if FLASHINFER_CUTLASS is available/selected)
- FlashInfer version (expect 0.6.9)
- Startup time

**Acceptance:** Benchmark completes for all concurrency levels.

**Results (2026-04-30, vLLM v0.20.1rc1.dev96+gefdc95674):**
| Level | eugr tok/s | production tok/s | Delta |
|-------|-----------|-----------------|-------|
| c1    | 57.7      | 59.9            | -3.7% |
| c4    | 176.5     | 166.2           | +6.2% |
| c8    | 384.2     | 373.8           | +2.8% |
| c16   | 607.1     | 564.0           | +7.6% |

Startup: 342s. KV cache: 45.26 GiB / 2,656,829 tokens. CUDA graph mode: PIECEWISE (FULL_AND_PIECEWISE unavailable with FlashInfer+speculative decode in v0.20.1). MoE backend: TRITON. Startup time: 342s (vs ~364s production).

**Files:** LAB_NOTEBOOK.md (Entry 052)

---

### Work Item 2.4 — eugr adopt/reject decision ✅ Completed 2026-04-30

**Status:** COMPLETE 2026-04-30
**Depends on:** 2.3

**Task:** Compare eugr benchmark against Phase 1 baseline. Apply decision criteria.

| Scenario | Decision |
|----------|----------|
| eugr c1 within 5% AND (c8 OR c16 improves > 3%) | ADOPT |
| eugr within 5% at all levels | STAY on current image (avoid unnecessary change) |
| eugr c8 OR c16 regresses > 3% | REJECT |

**If ADOPT:**
```bash
docker tag eugr-vllm:v0201-test vllm-cu132-test:latest
# Take snapshot
/home/claude/spark-config.sh snapshot post-eugr-v0201 "eugr v0.20.1rc1 adopted"
```

**If REJECT:**
```bash
docker stop qwen35 && docker rm qwen35
# Restore production image
docker run -d ... vllm-cu132-test:pre-eugr-v0201 ... [production flags from spark-device.md]
```

**Decision (2026-04-30): REJECT** — eugr v0.20.1rc1 regresses against post-firmware baseline on all levels. Root cause: FlashInfer + speculative decode forces PIECEWISE-only CUDA graphs in v0.20.1rc1 (FULL_AND_PIECEWISE unsupported). Production restored from `vllm-cu132-test:pre-eugr-v0201` (same as `:latest`).

Comparison vs post-firmware baseline (Entry 051):
- c1: 57.7 vs 65.9 = -12.5%
- c4: 176.5 vs 174.7 = +1.0%
- c8: 384.2 vs 394.3 = -2.6%
- c16: 607.1 vs 634.0 = -4.2%

Matches REJECT criterion: "c8 OR c16 regresses > 3%" (c16: -4.2%).

Re-test when FlashInfer backend gains FULL_AND_PIECEWISE support with speculative decode.

**Acceptance:** Decision documented with rationale. Production container running on winning image, verified healthy.

**Files:** LAB_NOTEBOOK.md (Entry 053)

---

## Phase 3: Model Evaluation (Pre-Quantized FP8)

**Goal:** Test pre-quantized FP8 weights (`Qwen/Qwen3.6-35B-A3B-FP8`) on the winning image from Phase 2. Hang rule invalidated by 3 independent signals (Seth Hobson Arena entry, forum reports, model repo).

**Prerequisite:** Phase 2 complete (winning image determined).

### Work Item 3.1 — Download pre-quant FP8 model

**Status:** PENDING
**Depends on:** 2.4

**Task:** Download the pre-quantized FP8 model weights if not already cached.

**SSH commands:**
```bash
# Check if already cached
ls /home/davistroy/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/ 2>/dev/null && echo "CACHED" || echo "NEED DOWNLOAD"

# If not cached — download (~25 GB)
huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8

# Verify
ls -la /home/davistroy/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/*/
```

**Acceptance:** Model files exist in HF cache.

**Files:** None (remote only)

---

### Work Item 3.2 — Benchmark pre-quant FP8 with MARLIN_ATOMIC_ADD

**Status:** PENDING
**Depends on:** 3.1

**Task:** Swap to pre-quantized FP8 model. Remove `--quantization fp8` (weights are already quantized). Also test `VLLM_MARLIN_USE_ATOMIC_ADD=1` (Seth's Arena config; our own startup logs recommended it).

**CRITICAL: 10-minute startup timeout.** If no `/health` 200 within 600 seconds, the hang bug is still present on this vLLM version. Kill and revert.

**SSH commands:**
```bash
docker stop qwen35 && docker rm qwen35

# Start with pre-quant FP8 model + MARLIN_ATOMIC_ADD
# Changes vs production:
#   - Model: Qwen/Qwen3.6-35B-A3B → Qwen/Qwen3.6-35B-A3B-FP8
#   - REMOVED: --quantization fp8 (weights already quantized)
#   - ADDED: -e VLLM_MARLIN_USE_ATOMIC_ADD=1
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton-cu132:/root/.triton \
  --entrypoint python3 \
  [WINNING_IMAGE_FROM_PHASE_2] \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name spark-llm \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

# === 10-MINUTE TIMEOUT ===
TIMEOUT=600
START=$(date +%s)
while true; do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "HEALTHY after $(($(date +%s) - START))s"
    break
  fi
  ELAPSED=$(($(date +%s) - START))
  if [ $ELAPSED -gt $TIMEOUT ]; then
    echo "TIMEOUT after ${TIMEOUT}s — HANG CONFIRMED"
    docker logs qwen35 2>&1 | tail -30
    docker stop qwen35 && docker rm qwen35
    echo "Reverting to on-the-fly FP8..."
    # Restore production config
    break
  fi
  echo "[$ELAPSED/${TIMEOUT}s] waiting..."
  sleep 15
done

# If healthy — benchmark
python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model spark-llm --concurrency 1 4 8 16
```

**Acceptance:** Either: (a) benchmark completes and results recorded, OR (b) hang confirmed and production config restored.

**Files:** LAB_NOTEBOOK.md (Entry 053)

---

### Work Item 3.3 — Pre-quant adopt/reject decision

**Status:** PENDING
**Depends on:** 3.2

**Task:** Compare pre-quant FP8 results against Phase 1/2 baseline.

| Scenario | Decision |
|----------|----------|
| Pre-quant starts AND c1 within 5% of on-the-fly | ADOPT (simpler, faster startup) |
| Pre-quant starts AND c1 regresses > 5% | Test without MARLIN_ATOMIC_ADD to isolate variables |
| Pre-quant hangs (timeout) | REJECT, document vLLM version constraint |

**If ADOPT:** Update production config. Also test MARLIN_ATOMIC_ADD separately if pre-quant was tested with it (isolate which change helped).

**If REJECT:** Restore on-the-fly FP8 from Phase 2 winning config.

**Acceptance:** Decision documented. Production container running on winning model, verified healthy.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md, spark-device.md, CLAUDE.md (update pre-quant hang rule if invalidated)

---

## Phase 4: Kernel Tuning (vLLM-Tune)

**Goal:** Apply auto-tuned Triton MoE kernel configs for GB10 FP8. Both our image and eugr's show "Using default MoE config" — tuned configs should improve decode throughput.

**Prerequisite:** Phase 3 complete (winning image + model determined). Run on the final production config.

### Work Item 4.1 — Research vLLM-Tune integration

**Status:** PENDING
**Depends on:** 3.3

**Task:** Investigate vLLM-Tune (SerraphimSerapis, NVIDIA forum). Determine: installation method, how it runs, what it produces, how configs are mounted into the container.

**Research steps:**
1. Search NVIDIA DGX Spark forum for "vLLM-Tune" posts by serapis
2. Check if it's a GitHub repo, pip package, or Docker image
3. Determine input requirements (running model? GPU access? model config?)
4. Determine output format (JSON config files? volume mount location?)
5. Check compatibility with our cu132+MTP config

**Acceptance:** Integration method documented. Either: (a) proceed to 4.2 with clear steps, OR (b) flag as blocked with specific reason.

**Files:** LAB_NOTEBOOK.md (research entry)

---

### Work Item 4.2 — Run vLLM-Tune and benchmark

**Status:** PENDING
**Depends on:** 4.1

**Task:** Run vLLM-Tune to generate optimized kernel configs for GB10 FP8 (E=256, N=512). Mount configs into the container. Benchmark before/after.

**SSH commands:** (to be determined by 4.1 research)

```bash
# Expected pattern (actual commands depend on 4.1 findings):
# 1. Run tuning tool
# 2. Mount generated config: -v /path/to/tuned-config:/path/in/container
# 3. Restart qwen35 with config mount
# 4. Verify "Using tuned MoE config" (no more default warning)
# 5. Benchmark c1 and c8 minimum

python3 ~/benchmarks/throughput_bench.py --url http://localhost:8000 --model spark-llm --concurrency 1 8
```

**Decision criteria:** Adopt if decode tok/s improves > 3%. Reject if regression or no measurable difference.

**Rollback:** Remove the config volume mount and restart.

**Acceptance:** Before/after benchmark documented. Config adopted or rejected with rationale.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md, spark-device.md (add volume mount if adopted)

---

## Phase 5: Infrastructure Hardening

**Goal:** Codify the winning configuration as a Docker Compose stack with health checks, startup ordering, and log rotation. Restore auxiliary containers stopped in Phase 2.

**Prerequisite:** Phases 2-4 complete (final production config is settled).

### Work Item 5.1 — Restore auxiliary containers

**Status:** PENDING
**Depends on:** 4.2 (or 3.3 if Phase 4 is skipped/blocked)

**Task:** Restart gliner, bge-m3, ce-service (stopped in Work Item 2.2 for clean GPU state). Verify all endpoints healthy.

**SSH commands:**
```bash
# Restart in order (GPU services first)
docker start gliner
sleep 30
curl -sf http://localhost:8002/v1/ner -X POST \
  -H "Content-Type: application/json" \
  -d '{"texts":["test"],"labels":["PERSON"],"threshold":0.5}' && echo "GLiNER OK"

docker start bge-m3
until curl -sf http://localhost:8004/health > /dev/null 2>&1; do sleep 5; done
echo "BGE-M3 OK"

docker start ce-service
sleep 10
curl -sf http://localhost:8005/ce/health && echo "CE-Service OK"

# Full health check
for port in 8000 8001 8002 8003 8004 8005; do
  curl -sf http://localhost:$port/health > /dev/null 2>&1 || \
  curl -sf http://localhost:$port/ > /dev/null 2>&1 || \
  curl -sf http://localhost:$port/api/v1/heartbeat > /dev/null 2>&1 || \
  curl -sf http://localhost:$port/ce/health > /dev/null 2>&1
  echo "Port $port: $?"
done
```

**Acceptance:** All 6 service endpoints healthy. nvidia-smi shows expected GPU memory allocation.

**Files:** None (remote only)

---

### Work Item 5.2 — Create docker-compose.yml

**Status:** PENDING
**Depends on:** 5.1

**Task:** Create a Docker Compose file that captures the complete running state of all containers. Use `docker inspect` to extract exact flags for each container.

**SSH commands:**
```bash
# Extract current config for each container
for c in qwen35 qwen3-embed bge-m3 gliner ce-service chromadb neo4j node-exporter; do
  echo "=== $c ==="
  docker inspect $c --format '{{json .Config}}' | python3 -m json.tool > /tmp/inspect_$c.json
  docker inspect $c --format '{{json .HostConfig}}' | python3 -m json.tool > /tmp/hostconfig_$c.json
done
```

**Compose file requirements:**
- All 8 containers defined as services
- Health checks for every service (using existing health endpoints)
- `depends_on` with `condition: service_healthy`:
  - qwen35: no deps (starts first)
  - qwen3-embed: depends_on qwen35 healthy
  - gliner: depends_on qwen3-embed healthy
  - bge-m3, ce-service: depends_on qwen35 healthy
  - chromadb, neo4j, node-exporter: no deps
- Log rotation on all services: `logging: { driver: json-file, options: { max-size: "100m", max-file: "3" } }`
- All volume mounts with absolute paths (never `~`)
- Restart policy: `unless-stopped`
- Any vLLM-Tune config mounts from Phase 4

**Acceptance:** `docker compose config` validates without errors. All services match their current `docker inspect` output (same image, same flags, same mounts, same ports).

**Files:** `/home/claude/docker-compose.yml` (remote), LAB_NOTEBOOK.md

---

### Work Item 5.3 — Test Docker Compose migration

**Status:** PENDING
**Depends on:** 5.2

**Task:** Stop all containers. Start via `docker compose up -d`. Verify startup order and all health checks pass.

**SSH commands:**
```bash
# Take pre-compose snapshot
/home/claude/spark-config.sh snapshot pre-compose "Before Docker Compose migration"

# Stop all existing containers
docker stop qwen35 qwen3-embed bge-m3 gliner ce-service chromadb neo4j node-exporter
docker rm qwen35 qwen3-embed bge-m3 gliner ce-service chromadb neo4j node-exporter

# Start via compose
cd /home/claude
docker compose up -d

# Watch startup order
docker compose logs -f --tail=0 2>&1 | head -100 &
LOGPID=$!

# Wait for all services healthy (timeout 15 min for full stack)
TIMEOUT=900
START=$(date +%s)
while true; do
  HEALTHY=$(docker compose ps --format json | python3 -c "import json,sys; data=[json.loads(l) for l in sys.stdin]; print(sum(1 for d in data if d.get('Health','')=='healthy' or d.get('State')=='running'))")
  TOTAL=$(docker compose ps -q | wc -l)
  ELAPSED=$(($(date +%s) - START))
  echo "[$ELAPSED/${TIMEOUT}s] $HEALTHY/$TOTAL services up"
  if [ "$HEALTHY" -ge "$TOTAL" ]; then echo "ALL UP"; break; fi
  if [ "$ELAPSED" -gt "$TIMEOUT" ]; then echo "TIMEOUT"; break; fi
  sleep 15
done
kill $LOGPID 2>/dev/null

# Full health verification
curl -sf http://localhost:8000/health && echo "qwen35 OK"
curl -sf http://localhost:8001/health && echo "qwen3-embed OK"
curl -sf http://localhost:8004/health && echo "bge-m3 OK"
curl -sf http://localhost:8003/api/v1/heartbeat && echo "chromadb OK"
curl -sf http://localhost:8005/ce/health && echo "ce-service OK"

# Quick inference test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"spark-llm","messages":[{"role":"user","content":"Hello"}],"max_tokens":5}' | python3 -m json.tool | head -10

# Take post-compose snapshot
/home/claude/spark-config.sh snapshot compose-v1 "Docker Compose migration complete"
```

**Rollback if compose fails:**
```bash
docker compose down
# Restart individually using spark-device.md commands
```

**Acceptance:** All 8 services start in correct order via `docker compose up -d`. All health checks pass. Inference test returns valid response. Snapshot captured.

**Files:** LAB_NOTEBOOK.md, spark-device.md (note compose management)

---

## Phase 6: Research Documentation

**Goal:** Document the NVFP4/INT4 and Gemma 4 paths for future decision-making. No system changes.

### Work Item 6.1 — Scope NVFP4/INT4 quantization path

**Status:** PENDING
**Depends on:** None (can run anytime)

**Task:** Document what's required to pursue the INT4/NVFP4 tier (90+ tok/s). This is a decision-support document, not an action plan.

**Research:**
1. What model checkpoints exist? (RedHatAI NVFP4, PrismaQuant 4.75-bit, AWQ)
2. What vLLM builds are required? (DFlash, nightly cu130, flashinfer_cutlass)
3. What's the quality tradeoff? (PrismaQuant 88/100 vs FP8 91/100 — how was this measured?)
4. What quality evaluation framework exists? (DanTup/spark-evals, custom benchmarks)
5. What's the minimum viable experiment? (single model swap, no DFlash, just NVFP4?)

**Document in:** LAB_NOTEBOOK.md research entry. Include a decision matrix with prerequisites, effort, expected gain, and quality risk for each path.

**Decision gate:** Defer execution until: (a) DFlash lands in mainline vLLM, OR (b) quality eval framework exists, OR (c) throughput requirements change.

**Acceptance:** Research entry written. Path documented with clear prerequisites and decision criteria.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md (update watch items)

---

### Work Item 6.2 — Check Gemma 4 community status

**Status:** PENDING
**Depends on:** None (can run anytime)

**Task:** Quick research pass on Gemma 4 status since our April 11 benchmarks (Entry 020-021).

**Questions to answer:**
1. Is guided JSON / structured output fixed for Gemma 4 in vLLM? (check #39130 and related PRs)
2. Has the throughput gap narrowed? (community benchmarks, eugr recipe changes)
3. What did eugr's "Gemma 4 recipe fixes" in v0.20.1rc1 address?
4. Any new Gemma 4 quantized checkpoints? (FP8, NVFP4)

**Decision gate:** Schedule a dedicated maintenance window only if: guided JSON is confirmed fixed AND throughput exceeds 50 tok/s c1 on community benchmarks.

**Acceptance:** Status documented. Decision on whether to schedule a Gemma 4 experiment.

**Files:** LAB_NOTEBOOK.md, SPARK_BASELINE.md (update Gemma 4 reference section), GEMMA4_EXPERIMENT_PLAN.md (update if warranted)

---

## Verification Checkpoints

| After Phase | Verify |
|-------------|--------|
| 0 | Backup exists, Neo4j healthy |
| 1 | Post-firmware baseline recorded, snap list reduced |
| 2 | Winning image decided, production container healthy |
| 3 | Winning model decided, production container healthy |
| 4 | Kernel tuning evaluated, production container healthy |
| 5 | All 8 services running via Docker Compose, health checks pass |
| 6 | Research documented in LAB_NOTEBOOK.md |

## Post-Plan Actions

After all phases complete:
1. Take final snapshot: `spark-config.sh snapshot performance-sprint-2026-05 "Post performance sprint"`
2. Run `/spark-recon` to update SPARK_BASELINE.md watch items
3. Update memory files with any new learnings
4. Archive this plan to `docs/archive/`
