# CLAUDE.md

This file provides guidance to Claude Code when working with files in this repository.

## Operational Rules — Learning Capture

**These rules apply in every session. Do not skip them.**

### When a hardware configuration or performance issue is diagnosed and fixed

After any non-trivial finding about the DGX Spark system:

1. **Update `CLAUDE.md`** (this file) — add or update a bullet with the operational rule.
2. **Update the memory file** — write a detailed entry in `C:\Users\Troy Davis\.claude\projects\C--Users-Troy-Davis-dev-personal-spark\memory\` in the appropriate topic file.
3. **Update `MEMORY.md`** — add a concise bullet + link to the topic file so it survives context compaction.

### What counts as a "non-trivial finding"

- Any hardware capability, limitation, or configuration quirk discovered
- Any LLM inference performance characteristic (throughput, memory limits, concurrency)
- Any fix that took more than one attempt to get right

### Learning file locations

| File | Purpose | When to write |
|------|---------|---------------|
| `CLAUDE.md` (this file) | Operational rules, always enforced | Every session with new learnings |
| `memory/MEMORY.md` | Concise index, survives compaction | After each new topic file entry |
| `memory/hardware-learnings.md` | DGX Spark hardware, LLM performance | Hardware/performance findings |

### Verified operational rules (do not repeat these mistakes)

- **HF cache owned by root:** `/home/davistroy/.cache/huggingface` is root-owned (from vLLM docker volume mounts). Non-root processes must use a separate cache dir or set `HF_HOME` to a user-writable location.
- **PyTorch CUDA on GB10 (sm_121):** Standard PyTorch cu126 builds only support up to sm_90. Must use PyTorch **nightly** with cu130 index for Blackwell GPU support. cu128 detects the GPU but NVRTC JIT fails at inference time with `invalid value for --gpu-architecture`. Only cu130 has sm_121 kernel support.
- **GPU memory budget (3-model config):** LLM at 0.72, embed at 0.13, GLiNER ~2GB on CUDA. Total ~108.5 GB of ~121.6 GiB. Leaves ~13 GB for OS/buffers. Previous 2-model config (0.75/0.15) left only ~10 GB which caused 4.6 GB swap with GLiNER added.
- **Container startup order matters:** Start qwen35 first, wait for `/health` 200, then qwen3-embed, wait for health, then gliner. Simultaneous startup causes CUDA memory allocation races and transient hangs. The `--restart unless-stopped` policy doesn't enforce ordering on reboot.
- **davistroy has no passwordless sudo:** OS-level changes (sysctl, systemctl) require interactive sudo password entry — cannot be done via non-interactive SSH from remote Claude sessions.
- **Docker GPU access:** Use `--gpus all` (not `--runtime nvidia`) on DGX Spark — nvidia runtime not configured, GPU access is via device requests.
- **HF cache absolute path required:** Docker volume mounts must use `/home/davistroy/.cache/huggingface`, NOT `~/.cache/huggingface`. The `claude` user's `~` expands to `/home/claude` which has no model weights. Using the wrong path causes silent hang (container tries to download 35GB model).
- **Triton JIT cache must be persisted:** Add `-v /home/claude/.cache/triton:/root/.triton` to vLLM containers. Without this, TRITON Fp8 MoE kernel compilation restarts from scratch on every `docker rm` (takes 15-30+ min on ARM64/SM 12.1).
- **FP8 MoE backend:** `VLLM_FLASHINFER_MOE_BACKEND=latency` env var only affects unquantized MoE, NOT FP8 mode (vLLM v0.17.0rc1). For FP8, TRITON backend is auto-selected. FlashInfer FP8 CUTLASS crashes on SM 12.1. The `--moe-backend marlin` CLI flag forces Marlin backend as alternative.
- **Ethernet static IP configured:** enP7s7 set to 192.168.10.33/24 (manual, gateway 192.168.10.1). WiFi stays on 192.168.10.32. NM connection files stored in both `/run/NetworkManager/system-connections/` (runtime) and `/etc/NetworkManager/system-connections/` (persistent). NM on this system reads from `/run/` first — always write there AND `/etc/`.
- **GPU memory utilization 0.65:** Reduced from 0.72 on 2026-03-28. KV cache was 0.54% utilized with 0 preemptions. Freed ~9 GB host RAM. num_gpu_blocks=2466 (down from 2585).
- **sysctl tuning applied:** `vm.swappiness=1`, `vm.min_free_kbytes=262144`, TCP buffer sizes increased. Persisted in `/etc/sysctl.d/99-spark-tuning.conf`.
- **Grafana 12 dashboards — use direct datasource UIDs:** Template variable references (`${DS_PROMETHEUS}`) in datasource objects cause panels not to render in Grafana 12.4.2. Use `{"uid": "PBFA97CFB590B2093"}` directly. The working pattern (from CFA dashboard) omits the `"type"` field in datasource refs. Provisioned dashboard was fixed by replacing all `${DS_PROMETHEUS}` with the literal UID.
- **Homeserver curl broken:** `curl` on Unraid 7.2 homeserver has a bus error. Use `wget` for all HTTP operations. For POST: `wget -O /tmp/resp.txt --header="Authorization: Basic $AUTH" --header="Content-Type: application/json" --post-file=/tmp/payload.json URL`.

## Spark Configuration Safety Rules

**These rules are mandatory for ALL operations on the DGX Spark system. Violation of these rules has caused system outages and data loss (2026-03-27 driver rollback bricked system, 2026-03-28 wrong volume mount caused 20-min debug, 2026-03-28 Grafana dashboard destroyed). Do not skip them.**

### Classification: Recoverable vs Unrecoverable Operations

Before executing ANY change on the Spark, classify it:

| Category | Examples | Required Process |
|----------|----------|-----------------|
| **Recoverable** | Editing a git-tracked file, sysctl changes (revert by editing conf), docker env var tweak | Standard — verify after |
| **Unrecoverable without physical access** | System reboot after kernel/DKMS/driver changes, BIOS/UEFI changes, bootloader mods | STOP. Inform user. Require explicit confirmation that physical console access is available |
| **Unrecoverable data loss** | Deleting Grafana dashboards, dropping DB tables, removing Docker volumes with state, overwriting non-git configs | STOP. Create backup FIRST. Or create new resource instead of modifying existing one |
| **Extended downtime risk** | Container restart (model reload 90s+), driver changes, GPU memory reallocation | Confirm idle state. Have rollback plan ready |

### Pre-Flight Checks: Container Operations

1. **Read the documented configuration FIRST.** Before any `docker run`, `docker stop`, or `docker rm`:
   - Read `spark-device.md` for the current known-working container command
   - Read `container-restart-learnings.md` for known pitfalls
   - Copy the exact documented command as the starting point

2. **Diff before execute.** If changing any parameter:
   - Start from the documented known-working command
   - Change ONLY the specific parameter being intentionally modified
   - Show the diff to the user before running
   - Verify all volume mount paths are absolute (never use `~` in Docker volume mounts)
   - Verify HF cache path is `/home/davistroy/.cache/huggingface` (not `/home/claude/...`)

3. **Verify container startup.** After any container restart:
   - Watch logs AND GPU memory every 10s (not blind polling)
   - Confirm GPU memory grows beyond ~3 GB within 60s for qwen35
   - Wait for `/health` 200
   - If stuck, check `docker inspect` volume mounts BEFORE changing any other parameters

### Pre-Flight Checks: Reboot

1. **Never reboot after DKMS/driver/kernel changes without confirming physical console access.**
   - Check DKMS output for "Signing key" or "MOK" messages
   - If MOK enrollment may be triggered, STOP and inform user
   - Require explicit confirmation: "I have physical console access, proceed"

2. **Never reboot on evenings or weekends** unless user explicitly confirms physical access.

3. **Before any reboot verify:** No pending DKMS builds, no partial apt/dpkg operations (`dpkg --audit`).

### Pre-Flight Checks: External State

External state = anything not tracked in git (Grafana dashboards, databases, API resources, Docker volumes with data).

1. **Never modify or delete external state in place.** Backup first or create a NEW resource with a different identifier.
2. **Grafana:** NEVER delete or overwrite a dashboard. Create new dashboards with unique UIDs. Use direct datasource UIDs, not template variables.

### Debugging Protocol

1. **Diagnose before changing.** Check basics FIRST: volume mounts (`docker inspect`), port bindings, env vars. Compare running config against documented config. Read actual logs.
2. **Never shotgun-debug.** Don't try multiple config changes hoping one works. One variable at a time, only after understanding the actual problem.
3. **If stuck after 2 failed attempts, STOP and present analysis to the user.** What you tried, what symptoms are, what you think root cause is, what you want to try next.

### Volume Mount Reference (copy-paste, never reconstruct from memory)

```bash
# qwen35 / qwen3-embed — HF cache (model weights)
-v /home/davistroy/.cache/huggingface:/root/.cache/huggingface

# qwen35 — Triton JIT cache (compiled kernels)
-v /home/claude/.cache/triton:/root/.triton

# gliner — separate HF cache
-v /home/davistroy/gliner-env/hf-cache:/root/.cache/huggingface
```

NEVER use `~/.cache/huggingface` — the `claude` user's `~` is `/home/claude`, which has no model weights.

## Project Overview

Reference documentation for NVIDIA Jetson DGX Spark AI system — user manual and configuration notes.
