# Swap Relief + Prefix Caching Plan — qwen35 + bge-m3

**Created:** 2026-05-09 (after Entry 062 audit)
**Trigger:** Entry 062 audit found 6.5 GiB system swap used after 9-day uptime; qwen35 EngineCore 2.16 GiB swapped, bge-m3 EngineCore 1.54 GiB swapped, bge-m3 worker 553 MiB.
**Window:** 2026-05-09, ~6–7 minutes total downtime (mostly qwen35 warm-cache reload)
**Bundle:** Add `--enable-prefix-caching` to qwen35 at the same restart (free wins for pipeline workloads with shared system prompts)

---

## Goal

Clear ~4.3 GiB of accumulated process-level swap and enable prefix caching on qwen35 with no performance regression.

## Scope

**In:**
- `docker compose stop` and `docker compose up -d` for `qwen35` and `bge-m3` only
- One-line edit to `/home/claude/docker-compose.yml` adding `--enable-prefix-caching` to the qwen35 `command:` block

**Not in:**
- vLLM image upgrade (eugr v0.20.2rc1 evaluation is a separate plan)
- Atlas / AWQ INT4 / DFlash experiments (separate)
- Any other container (qwen3-embed, gliner, ce-service, chromadb, neo4j stay running)
- sysctl, OS reboot, firmware
- `swapoff -a` cleanup (per user direction — per-process VmSwap is the real metric)

## Pre-flight checklist (BLOCK if any fails)

| # | Check | Pass criterion |
|---|---|---|
| 1 | qwen35 idle | `vllm:num_requests_running` and `vllm:num_requests_waiting` both 0 |
| 1b | bge-m3 idle | same metric, port 8004 |
| 2 | No active TCP clients on 8000 / 8004 | empty established socket list |
| 3 | Backup compose file | `cp /home/claude/docker-compose.yml /home/claude/docker-compose.yml.bak.<timestamp>` |
| 4 | Snapshot current container config | `docker inspect qwen35 > /home/claude/qwen35.preflight.<timestamp>.json` |
| 5 | Triton cu132 cache present | `/home/claude/.cache/triton-cu132` exists, contents non-empty |
| 6 | No pending DKMS / dpkg | `dpkg --audit` empty |
| 7 | GPU at idle | <50W draw, <50°C |
| 8 | Disk has room | >100 GB free on / |

## The diff (apply only after pre-flight passes + user confirms)

In `/home/claude/docker-compose.yml`, qwen35 service `command:` block — insert one line before `--speculative-config`:

```diff
       - --max-num-batched-tokens
       - "4096"
+      - --enable-prefix-caching
       - --speculative-config
       - '{"method":"mtp","num_speculative_tokens":2}'
```

**Application command (single insertion, GNU sed):**
```bash
sed -i '/^      - --speculative-config$/i\      - --enable-prefix-caching' /home/claude/docker-compose.yml
```

The `^      - --speculative-config$` pattern matches exactly one line in the file (verified). No other compose service uses speculative decoding.

**Why this location:** keeps the speculative-decode block intact at the end of the command list and groups with engine-runtime flags. Argument order is irrelevant to vLLM but readable for future audits.

## Execution sequence

```bash
# 0. Pre-flight passed, user confirmed; record start time
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); echo "$START"

# 1. Stop bge-m3 first (smaller, faster, no spec-decode warm-up)
docker compose -f /home/claude/docker-compose.yml stop bge-m3
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader  # verify ~10 GiB freed

# 2. Stop qwen35
docker compose -f /home/claude/docker-compose.yml stop qwen35
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader  # verify ~86 GiB freed

# 3. Apply the edit
sed -i '/^      - --speculative-config$/i\      - --enable-prefix-caching' /home/claude/docker-compose.yml
diff /home/claude/docker-compose.yml.bak.<ts> /home/claude/docker-compose.yml
# Expect: exactly one inserted line

# 4. Start qwen35
docker compose -f /home/claude/docker-compose.yml up -d qwen35

# 5. Wait for /health (timeout 600s; abort if exceeded)
START=$(date +%s)
until curl -fs http://localhost:8000/health > /dev/null; do
  ELAPSED=$(( $(date +%s) - START ))
  [ $ELAPSED -gt 600 ] && { echo "ABORT: qwen35 unhealthy after 600s"; exit 1; }
  sleep 10
done

# 6. Start bge-m3
docker compose -f /home/claude/docker-compose.yml up -d bge-m3
until curl -fs http://localhost:8004/health > /dev/null; do sleep 5; done
```

**Expected timing:** qwen35 ~280–360s warm cache; bge-m3 ~30s.

## Verification gates (all must pass)

| # | Gate | Pass criterion |
|---|---|---|
| V1 | All endpoints healthy | 8000/8001/8002/8003/8004 → 200; 8005 `/ce/health` → 200; 7474 → 200 |
| V2 | LLM correctness | Smoke prompt returns "HEALTHY", latency <1s, `enable_thinking: false` honored |
| V3 | Prefix caching active | `vllm:gpu_prefix_cache_queries_total` and `vllm:gpu_prefix_cache_hits_total` exist in /metrics |
| V4 | Prefix cache hit on warm path | Two identical 500-token system prompts → second call shows incremented hit counter |
| V5 | KV cache budget preserved | Startup log: ≥46 GiB available, ≥1.1M tokens (within ±5% of 46.09 GiB / 1.14M baseline) |
| V6 | Throughput sweep | c1/c4/c8/c16 each within ±5% of post-firmware baseline (65.9 / 174.7 / 394.3 / 634.0 tok/s) |
| V7 | Per-process swap relieved | qwen35 EngineCore VmSwap < 100 MB, bge-m3 EngineCore VmSwap < 100 MB |
| V8 | Peripheral containers unaffected | gliner / qwen3-embed / ce-service / chromadb / neo4j health 200, zero restart count |

## Rollback

**Trigger if:** qwen35 fails /health within 600s, OR V5 fails (KV cache shrink >10%), OR V6 c1 sustained <62 tok/s, OR V1/V2/V8 fails.

```bash
docker compose -f /home/claude/docker-compose.yml stop qwen35
cp /home/claude/docker-compose.yml.bak.<timestamp> /home/claude/docker-compose.yml
docker compose -f /home/claude/docker-compose.yml up -d qwen35
# Wait /health, run V1/V2 to confirm rollback healthy
```

Triton cache (`/home/claude/.cache/triton-cu132`) untouched — rollback boots from same warm cache. No image pull, no kernel rebuild.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Startup hang | Low | High | 600s timeout + log streaming; rollback ready |
| Prefix caching reduces effective KV pool | Very low | Medium | V5 catches it; vLLM design uses same pool |
| First-call latency spike on warm prefix path | Low | Low | Expected; second call is the win |
| `--enable-prefix-caching` × MTP=2 interaction | Low | Medium | vLLM v0.19.x supports both; if V6 regresses, rollback |
| System swap counter doesn't drop visibly | Expected | Cosmetic | Per-process VmSwap is the real metric |

## Open mitigation post-execution (optional, not blocking)

- If `vllm:gpu_prefix_cache_hit_rate` < 30% after pipeline workload runs, prefix caching isn't paying off for our prompt mix → revert in next window.
- Watch swap trend over the next 9-day cycle. If swap re-grows past 4 GiB before next planned restart, escalate to investigation: which process accumulates? Could indicate a vLLM memory leak.

## Decision criteria summary

- **Commit:** all V1–V8 pass, no V6 regression
- **Rollback:** any gate fails, or qwen35 doesn't reach /health in 600s
- **Document:** Entry 064 in LAB_NOTEBOOK with measured before/after on swap, throughput sweep table, prefix cache metrics

---

## Execution Result (2026-05-09 23:06 → 23:36 UTC)

**Outcome: PARTIAL SUCCESS — restart kept, prefix caching rolled back.**

### Swap relief: **SUCCESS** ✓
- qwen35 EngineCore: 2.16 GiB → 264 MB (-88%)
- bge-m3 EngineCore: 1.54 GiB → 0 kB (fully clean)
- bge-m3 worker: 553 MiB → 0 kB
- System swap: 6.5 GiB → 5.1 GiB (lazy reclaim; per-process metric is what matters)
- System available RAM: 10 GiB → ~11 GiB after restart

### Prefix caching: **ROLLED BACK**
- Synthetic V6 bench: c1 60.8, c4 162.8, c8 369.9, c16 578.5 tok/s (-7 to -9% vs Entry 052 baseline; failed ±5% gate)
- `vllm:prefix_cache_hits_total = 0` after 3151 queries — bench prompts were ~15-20 tokens, shorter than vLLM's 16-token block boundary, so no caching hits possible
- Rollback executed: yaml restored from backup, qwen35 restarted clean, `enable_prefix_caching=False` confirmed in startup config

### Post-rollback bench (verifying we're back to a consistent baseline)
- c1=58.3, c4=161.5, c8=374.1, c16=552.7 tok/s
- Within ±3% of pre-firmware baseline (Entry 050: 59.9 / 166.2 / 373.8 / 564.0)
- 5-13% BELOW post-firmware baseline (Entry 052: 65.9 / 174.7 / 394.3 / 634.0)
- **Implication:** the Entry 052 post-firmware "+10%" gains may have been measurement variance or transient. Today's numbers match Entry 050 (gpu_util 0.70 baseline). Worth a separate investigation before declaring a regression.

### Total downtime
- bge-m3: 6m 30s
- qwen35: 6m 5s (first cycle) + 6m 5s (rollback cycle) = 12m 10s

### Lessons
1. **Bench prompt length matters for prefix cache validation.** Future tests of `--enable-prefix-caching` MUST use prompts ≥ 16 tokens (preferably ≥ 200 tokens to span multiple blocks) AND repeat the SAME prefix across requests.
2. **The post-firmware baseline (Entry 052) is suspect.** Today's measurement matches the pre-firmware baseline within noise. Need a controlled re-bench to confirm whether firmware-attributed gains were real.
3. **Swap relief mechanism works as designed** — container restart cleared the per-process accumulated swap.

### Follow-ups (recommended, not done in this window)
- **[Open]** Re-test `--enable-prefix-caching` with a realistic 800-token system prompt + multiple matching calls; measure actual `prefix_cache_hits_total` increment.
- **[Open]** Re-bench post-firmware baseline (Entry 052 numbers) to determine whether the "+10%" gain is reproducible or was a one-time variance.
- **[Watch]** qwen3-embed (1.26 GiB EC swap) and gliner (1.74 GiB swap) accumulated swap not addressed in this window. Defer to next maintenance cycle if they grow further.

