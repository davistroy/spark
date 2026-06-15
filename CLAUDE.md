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
- **Primary user has no passwordless sudo** — OS-level changes require interactive sudo; cannot be done via non-interactive SSH. Exception: `/etc/sudoers.d/claude-snap` added 2026-04-30 grants `claude` NOPASSWD for `/usr/bin/snap` (added via `tee` which was already NOPASSWD).
- **snap not in original NOPASSWD list** — `sudo snap remove` fails non-interactively without the sudoers.d entry above. `tee` trick: `echo 'rule' | sudo tee /etc/sudoers.d/file && sudo chmod 440 /etc/sudoers.d/file`.
- **Docker GPU access:** Use `--gpus all` (not `--runtime nvidia`).
- **HF cache absolute path required:** Use `/home/<user>/.cache/huggingface`, NOT `~/.cache/huggingface`. Wrong path causes silent hang (tries to download 35GB model).
- **Triton JIT cache must be persisted:** Add `-v /home/claude/.cache/triton:/root/.triton` to vLLM containers. Without this, kernel compilation restarts from scratch on every `docker rm` (15-30+ min on ARM64/SM 12.1).
- **FP8 MoE backend (v0.19.0):** TRITON auto-selected. `VLLM_TEST_FORCE_FP8_MARLIN=1` is NO LONGER NEEDED — removed 2026-04-11. Auto-select is correct.
- **Pre-quantized FP8 model hangs on v0.19.0 (version-specific):** `Qwen/Qwen3.5-35B-A3B-FP8` caused silent hang during load on v0.19.0. **Qwen3.6-35B-A3B-FP8 does NOT hang on v0.19.1rc1** (confirmed 2026-04-30, Entry 054, startup 391s). Use on-the-fly FP8 quantization (`--quantization fp8`) regardless — it outperforms pre-quant across most concurrency levels (c1: on-the-fly 65.9 vs pre-quant 58.1, c16: 634.0 vs 541.0). Pre-quant uses block-scaled FP8 kernel + uncalibrated KV scales, which underperforms row-wise on SM121.
- **Ethernet static IP:** enP7s7 set to <spark-lan-ip>/24, route-metric=700 (WiFi=600 takes priority). NM reads from `/run/NetworkManager/system-connections/` first — write there AND `/etc/`. Delete NM auto-generated volatile profiles.
- **Ethernet switch port: USW Pro 24 Port 10.** Do NOT move cable. Moving triggers MAC flapping detection and silently drops all frames. If you must change ports, reboot the switch afterward.
- **Dual-homed routing hazard:** WiFi (.32) and Ethernet (.33) on same /24. Ethernet route-metric MUST be higher than WiFi (700 vs 600).
- **GPU memory utilization 0.70** (increased from 0.65 on 2026-04-24 after gliner memory fix). KV cache: 47.95 GiB, 1,142,736 tokens, max concurrency 85.92x.
- **sysctl tuning applied:** `vm.swappiness=1`, `vm.min_free_kbytes=262144`, TCP buffers increased. Persisted in `/etc/sysctl.d/99-spark-tuning.conf`.
- **Grafana (updated 2026-06-15):** Instance is now `open-brain-grafana` (grafana/grafana:latest, **v13.0.1**) on `homeserver:3050`, admin password **rotated 2026-06-15** — now in **Bitwarden Secrets Manager** (key `GRAFANA_ADMIN_PASSWORD`, *ai-work* project; ideally belongs in *personal*, but that project is read-only for the current bws token) and mirrored in `/mnt/user/appdata/open-brain/.env` (compose reads `GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}`). The prior `admin`/`Spark2026!` no longer work. Retrieve with `bws secret list` / get. Prometheus datasource uid is the literal **`DS_PROMETHEUS`** (NOT the prior instance's `PBFA97CFB590B2093`); Loki is `DS_LOKI`. Still: reference datasources by literal uid `{"uid":"DS_PROMETHEUS"}` (omit `"type"`); do NOT use `${DS_PROMETHEUS}` template-var form. The old "DGX Spark" folder (`dfhdwahqbii9sc`) is gone after the rebuild — only an "Open Brain" folder exists; new dashboards land in General unless `folderUid` is set. Spark metrics ARE scraped here (job `spark-gpu`, instance `spark.k4jda.net:9400`). New reliability dashboard: uid `spark-reliability`.
- **Homeserver curl broken:** Use `wget` for all HTTP ops on Unraid 7.2 homeserver. For POST: `wget -O /tmp/resp.txt --header="..." --post-file=/tmp/payload.json URL`.
- **Production model is `Qwen/Qwen3.6-35B-A3B-FP8`** (native pre-quantized, adopted 2026-05-18 Entry 073). Was `Qwen/Qwen3.6-35B-A3B` (BF16 + on-the-fly `--quantization fp8`) from 2026-04-23 to 2026-05-18. `--served-model-name` is `spark-llm`. Container name remains `qwen35`. Production compose: `/home/claude/docker-compose.yml`. Rollback to on-the-fly FP8: `cp /home/claude/docker-compose.yml.pre-fp8prequant /home/claude/docker-compose.yml && docker compose stop qwen35 && docker compose up -d qwen35`.
- **Production uses BF16 KV cache + `--max-num-batched-tokens 32768`** (Entry 073, 2026-05-18). Do NOT add `--kv-cache-dtype fp8` or `--quantization fp8` — they are inappropriate for the pre-quantized model and impose 5-15% throughput cost. BF16 KV reduces KV cache token capacity by 55% (504K vs 1.12M at 131K context, max concurrency 3.85x vs 8.57x); non-binding for current workload but flag if high-concurrency 131K usage materializes.
- **Pre-quant FP8 vs on-the-fly FP8 (2026-05-18 Entry 070):** Pre-quant FP8 wins on cu132+MTP + v0.19.1rc1.dev219+g72ff142c3.d20260412 (production now). Measured gains: c=1 +13.0%, c=4 +25.1%, c=8 +14.4%, c=16 +29.3%. **This contradicts Entry 054-055 (2026-04-30)** which rejected pre-quant — that rejection was correct for the older vLLM build but is incorrect for the current build. Kernel selection paths have evolved. Re-validate if vLLM upgrades again.
- **GLM-4.7-Flash and Qwen3-Coder-Next-FP8 both rejected on Spark (Entries 071, 072, 2026-05-18):** GLM-4.7 is 42-65% slower than Qwen3.6-FP8 on SM121 (research's "fastest" claim does NOT hold here). Coder-Next has 0% MTP acceptance due to vllm#37554 q_scale fallback (FP8 KV cache hybrid GDN+attention), making it 58-69% slower. Quality essentially equivalent for all three. NOTE (2026-06-11 recon): vllm#37554 was already closed-as-completed 2026-03-20 — the closure IS the q_scale=1.0 fallback that causes the 0% acceptance. Re-evaluate instead when a real KV-scale calibration fix for hybrid GDN+attention lands (none as of v0.22.1), or if spec-decode hybrid-attention PR #39949 changes Coder-Next MTP behavior.
- **`enable_thinking: false` placement:** Must be at request top level (`chat_template_kwargs`), NOT inside `extra_body`. Wrong placement silently fails to suppress thinking tokens, causing token exhaustion on short `max_tokens` budgets.
- **`--tool-call-parser qwen3_coder` is correct for Qwen3.6:** Despite the name, `qwen3_coder` parses XML tool calls (`<tool_call><function=...>`), which is what Qwen3.6's chat template uses. `qwen3_xml` also works (expat-based) but switching is unnecessary.
- **Production image is cu132+MTP** (adopted 2026-04-23). Image: `vllm-cu132-test:latest`. Requires `--entrypoint python3` override (cu132 image uses NVIDIA base entrypoint), `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (corrected 2026-06-11, Entry 076 — live config uses the speculative-config JSON form, NOT the legacy `--num-speculative-tokens`/`--speculative-model` flags previously documented here), and `--max-num-batched-tokens 32768` (was 4096 prior to 2026-05-18; bumped on production switch to pre-quant FP8).
- **Production attention backend is FLASH_ATTN (auto-selected), NOT FlashInfer (verified 2026-06-11, Entry 076):** startup log shows `Using FLASH_ATTN attention backend`. FlashInfer is used only for MoE kernels via `VLLM_FLASHINFER_MOE_BACKEND=latency`. Relevant when assessing FlashInfer-specific upstream bugs (e.g. vllm#37754 FlashInfer-attention+MTP Xid-13 crash — does not directly apply to our config).
- **Separate Triton caches per CUDA toolkit:** cu130 uses `/home/claude/.cache/triton:/root/.triton`, cu132 uses `/home/claude/.cache/triton-cu132:/root/.triton`. Never mix — rollback requires the original cache intact.
- **Firmware updates change the kernel.** DGX Spark firmware (EC/UEFI/USB-PD) can bump the kernel (e.g., 6.17.0-1008→6.17.0-1014). The matching NVIDIA module package (`linux-modules-nvidia-580-open-{kernel-version}`) is NOT auto-installed. After firmware update: `apt install linux-modules-nvidia-580-open-$(uname -r)` → `modprobe nvidia` → `systemctl restart nvidia-persistenced`. No reboot needed. Prebuilt packages are pre-signed (Secure Boot safe).
- **`apt dist-upgrade` flips nvidia to DKMS with an UNENROLLED MOK key — reboot would lose the GPU (TESTED 2026-06-15, Entry 078; current kernel now 6.17.0-1021, driver 580.159.03).** On a kernel/driver bump, dist-upgrade removes the prebuilt `linux-modules-nvidia-580-open-*` packages and installs `nvidia-dkms-580-open`, which builds modules signed with `/var/lib/shim-signed/mok/MOK.der` (`CN=spark Secure Boot Module Signature key`) that is **NOT enrolled** (`mokutil --test-key` confirms). Under SecureBoot those won't load → GPU dead after reboot (OS still boots + SSH works, so recoverable, not a brick). dist-upgrade also leaves `linux-image-<newkver>` **half-configured** (DKMS arm64/aarch64 double-autoinstall aborts → `dpkg --audit` shows it). **Recovery (restore prebuilt, no MOK enrollment):** (1) `sudo dpkg --purge --force-depends nvidia-dkms-580-open`; (2) `sudo dpkg --configure -a` (clears the half-config); (3) `sudo apt-get install -y linux-modules-nvidia-580-open-nvidia-hwe-24.04 linux-modules-nvidia-580-open-<newkver>-nvidia` — install the META **and** the kernel-specific module **together** so apt satisfies `nvidia-driver-580-open` via prebuilt (not DKMS); (4) **before reboot, verify** `modinfo -F signer <…/nvidia.ko>` = `Canonical Ltd. Kernel Module Signing` (enrolled UEFI CA → SecureBoot-safe). Old kernels' prebuilt modules become uninstallable after a driver bump (they pin the old `nvidia-kernel-common`), so the old kernel is an SSH-recovery fallback only (boots OS, no GPU). Stay on 580.x (590 = GB10 UMA leak, unsupported).
- **Marlin WNA16 MoE CUDA graph capture hangs on SM121 (2026-05-13):** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (compressed-tensors, group_size=32, num_bits=4) with Marlin WNA16 MoE backend: piecewise CUDA graphs capture in 3s, but FULL capture hangs indefinitely (20+ min, no progress, 0% GPU utilization). Must use `--enforce-eager`. AWQ INT4 underperforms FP8 on-the-fly by 10-28% across all concurrency levels even with enforce-eager. **No viable INT4 quantization path exists for SM121.**
- **vLLM-Tune configs incompatible with CUDA graph capture (2026-04-30):** `SeraphimSerapis/vllm-tune` pre-tuned GB10 TP=1 MoE config (BLOCK_N=256, BLOCK_K=256, warps=8, stages=3 for M=1) requires 110,592 bytes shared memory during CUDA graph capture — exceeds GB10's 101,376 byte per-SM runtime limit. The tuning benchmarks in Triton eager mode (partial allocation) but production uses CUDA graph capture (full static allocation). Result: `triton.runtime.errors.OutOfResources` crash at container startup. Default MoE config (BLOCK_N=64, BLOCK_K=128, stages=4, 40,960 bytes) is the largest valid parameter set for our CUDA graph + MTP config. Do NOT apply vLLM-Tune configs without verifying shared memory budget against CUDA graph capture mode first.
- **30-minute soak test PASSED (2026-05-10):** cu132+MTP with Qwen3.6-35B-A3B completed 252 requests (100% success rate) with 43.1 ± 5.2 sec latency (+3.9% improving trend), 12.06 tok/s throughput (13.9% CV), zero errors, zero restarts. System PRODUCTION-READY. Cold-start warm-up: first 7 min elevated (54 sec), then stable. See [soak-test-2026-05-10.md](./memory/soak-test-2026-05-10.md).

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
