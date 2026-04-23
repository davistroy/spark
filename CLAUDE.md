# CLAUDE.md — DGX Spark

## Learning Capture — Every Session

After any non-trivial finding (hardware capability/limitation, LLM performance characteristic, or any fix > 1 attempt):

1. Update `CLAUDE.md` — add/update bullet.
2. Write detailed entry to `<user-home>\.claude\projects\C--Users-Troy-Davis-dev-personal-spark\memory\` in appropriate topic file.
3. Update `memory/MEMORY.md` — concise bullet + link.

### Learning Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Operational rules, always enforced |
| `memory/MEMORY.md` | Concise index, survives compaction |
| `memory/hardware-learnings.md` | Hardware, LLM performance |

## Verified Rules — Do Not Repeat

- **HF cache owned by root:** `/home/<user>/.cache/huggingface` is root-owned from vLLM docker volumes. Non-root processes must set `HF_HOME` to a user-writable location.
- **PyTorch CUDA on GB10 (sm_121):** Must use PyTorch **nightly** with cu130 index. cu128 detects GPU but NVRTC JIT fails at inference. Only cu130 has sm_121 kernel support.
- **GPU memory budget (3-model config):** LLM at 0.65, embed at 0.13, GLiNER ~2GB. Total ~108.5 GB of ~121.6 GiB.
- **Container startup order:** Start qwen35 first → wait for `/health` 200 → qwen3-embed → wait → gliner. Simultaneous startup causes CUDA memory races.
- **Primary user has no passwordless sudo** — OS-level changes require interactive sudo; cannot be done via non-interactive SSH.
- **Docker GPU access:** Use `--gpus all` (not `--runtime nvidia`).
- **HF cache absolute path required:** Use `/home/<user>/.cache/huggingface`, NOT `~/.cache/huggingface`. Wrong path causes silent hang (tries to download 35GB model).
- **Triton JIT cache must be persisted:** Add `-v /home/claude/.cache/triton:/root/.triton` to vLLM containers. Without this, kernel compilation restarts from scratch on every `docker rm` (15-30+ min on ARM64/SM 12.1).
- **FP8 MoE backend (v0.19.0):** TRITON auto-selected. `VLLM_TEST_FORCE_FP8_MARLIN=1` is NO LONGER NEEDED — removed 2026-04-11. Auto-select is correct.
- **Pre-quantized FP8 model hangs on v0.19.0:** `Qwen/Qwen3.5-35B-A3B-FP8` causes silent hang during load. Use on-the-fly FP8 quantization (`--quantization fp8` with `Qwen/Qwen3.5-35B-A3B`).
- **Ethernet static IP:** enP7s7 set to <spark-lan-ip>/24, route-metric=700 (WiFi=600 takes priority). NM reads from `/run/NetworkManager/system-connections/` first — write there AND `/etc/`. Delete NM auto-generated volatile profiles.
- **Ethernet switch port: USW Pro 24 Port 10.** Do NOT move cable. Moving triggers MAC flapping detection and silently drops all frames. If you must change ports, reboot the switch afterward.
- **Dual-homed routing hazard:** WiFi (.32) and Ethernet (.33) on same /24. Ethernet route-metric MUST be higher than WiFi (700 vs 600).
- **GPU memory utilization 0.65** (reduced from 0.72 on 2026-03-28). num_gpu_blocks=2466.
- **sysctl tuning applied:** `vm.swappiness=1`, `vm.min_free_kbytes=262144`, TCP buffers increased. Persisted in `/etc/sysctl.d/99-spark-tuning.conf`.
- **Grafana 12 dashboards:** Use direct datasource UIDs (`{"uid": "PBFA97CFB590B2093"}`), not template variable references (`${DS_PROMETHEUS}`). Omit `"type"` field. Template vars cause panels not to render in Grafana 12.4.2.
- **Homeserver curl broken:** Use `wget` for all HTTP ops on Unraid 7.2 homeserver. For POST: `wget -O /tmp/resp.txt --header="..." --post-file=/tmp/payload.json URL`.
- **Production model is Qwen3.6-35B-A3B** (adopted 2026-04-23). Container name and `--served-model-name` remain `qwen3.5-35b` for downstream compatibility.
- **`enable_thinking: false` placement:** Must be at request top level (`chat_template_kwargs`), NOT inside `extra_body`. Wrong placement silently fails to suppress thinking tokens, causing token exhaustion on short `max_tokens` budgets.
- **Production image is cu132+MTP** (adopted 2026-04-23). Image: `vllm-cu132-test:latest`. Requires `--entrypoint python3` override (cu132 image uses NVIDIA base entrypoint), `--num-speculative-tokens 2`, `--speculative-model [MTP]`, and `--max-num-batched-tokens 4096`.
- **Separate Triton caches per CUDA toolkit:** cu130 uses `/home/claude/.cache/triton:/root/.triton`, cu132 uses `/home/claude/.cache/triton-cu132:/root/.triton`. Never mix — rollback requires the original cache intact.

## Configuration Safety Rules — MANDATORY

**Violations have caused outages and data loss (2026-03-27 driver rollback bricked system; 2026-03-28 wrong volume mount; 2026-03-28 Grafana dashboard destroyed).**

### Operation Classification

| Category | Examples | Required Process |
|----------|----------|-----------------|
| **Recoverable** | git-tracked file edits, sysctl changes, docker env tweak | Standard — verify after |
| **Unrecoverable without physical access** | Reboot after kernel/DKMS/driver changes, BIOS/UEFI, bootloader | STOP. Inform user. Require explicit confirmation of physical console access. |
| **Unrecoverable data loss** | Deleting Grafana dashboards, dropping DB tables, removing Docker volumes, overwriting non-git configs | STOP. Backup FIRST. Or create new resource. |
| **Extended downtime risk** | Container restart (model reload 90s+), driver changes, GPU memory reallocation | Confirm idle state. Have rollback plan ready. |

### Pre-Flight: Container Operations

1. Read `spark-device.md` for current known-working container command BEFORE any `docker run/stop/rm`.
2. Start from documented command. Change ONLY the specific parameter. Show diff to user before running.
3. All volume mount paths must be absolute (never `~` in Docker mounts).
4. After restart: watch logs AND GPU memory every 10s; confirm GPU memory grows >3 GB within 60s for qwen35; wait for `/health` 200.

### Pre-Flight: Reboot

- **Never reboot after DKMS/driver/kernel changes without physical console confirmation.**
- Check for MOK messages. If MOK enrollment may trigger — STOP, inform user.
- **Never reboot evenings or weekends** without explicit physical access confirmation.
- Before any reboot: verify no pending DKMS builds, no partial dpkg (`dpkg --audit`).

### Pre-Flight: External State

- **Never modify or delete external state in place.** Backup first or create NEW resource with different identifier.
- **Grafana:** NEVER delete or overwrite a dashboard. Create new with unique UIDs.

### Debugging Protocol

1. Diagnose before changing. Check: volume mounts (`docker inspect`), port bindings, env vars. Compare running config against documented. Read actual logs.
2. Never shotgun-debug. One variable at a time, only after understanding root cause.
3. If stuck after 2 failed attempts, STOP and present analysis: what you tried, symptoms, root cause hypothesis, next step.

### Volume Mount Reference (copy-paste, never reconstruct)

```bash
# qwen35 / qwen3-embed — HF cache
-v /home/<user>/.cache/huggingface:/root/.cache/huggingface
# qwen35 — Triton JIT cache
-v /home/claude/.cache/triton:/root/.triton
# gliner — separate HF cache
-v /home/<user>/gliner-env/hf-cache:/root/.cache/huggingface
```

NEVER use `~/.cache/huggingface`.

## Project

Reference documentation for NVIDIA DGX Spark AI system — configuration notes and user manual.
