# Implementation Plan — Security Remediation + NVFP4/Arm-C Eval + Ops Housekeeping

**Created:** 2026-06-30
**Branch:** main
**Status:** PENDING (0/18 items)
**Based On:** spark-recon Entry 081 + spark-audit Entry 082 (2026-06-30); ultra-plan Phases 0–5 (this session).
**Relationship to prior plans:**
- `IMPLEMENTATION_PLAN.md` (Performance Sprint) — COMPLETE 2026-04-30, reference only.
- `IMPLEMENTATION_PLAN_MODEL_EVAL.md` — COMPLETE 2026-05-18, reference only.
- `IMPLEMENTATION_PLAN_SPARK_ROADMAP.md` — **partially complete; this plan ABSORBS its §5.3 (Arm C, PENDING) and P3 NVFP4 backlog** (NVFP4 trigger fired 2026-06-30) by coupling them into CS-B. Roadmap §3.2 (Grafana deploy, BLOCKED) and §5.4/5.5 (synthesis/adoption) remain owned by the roadmap.

> **Execute with:** `/implement-plan --input IMPLEMENTATION_PLAN_2026-06-30.md` — but **only Phase 2.3 (builder prune) and parts of Phase 3 prep are autonomous-agent-executable.** CS-A (security), CS-B eval (production downtime), and the davistroy-sudo items are operational/human-gated. See the Execution Mode legend.

---

## Execution Mode legend
- **AUTO** — autonomous-agent-safe (read-only or fully reversible, no production impact).
- **CLAUDE-OPS** — claude executes via NOPASSWD primitives, supervised (security/state-sensitive).
- **GATED** — guided SSH with production downtime; requires idle box + live supervision + (for adoption) user approval.
- **DAVISTROY** — requires interactive sudo by `davistroy` (`ssh-keygen -A` alt path, `usermod`).

## Scope
Six items from the 2026-06-30 recon/audit, grouped into three change sets:
- **CS-A** — CVE-2026-24218 SSH host-key remediation (confirmed factory-shared keys).
- **CS-B** — NVFP4-on-vLLM evaluation **coupled with** Arm C build upgrade (v0.23.x), gated/sandboxed.
- **CS-C** — Ops housekeeping: kernel-log visibility, secondary-service swap clear, docker reclaim.

## Out of Scope / Exclusions
- **Adopting** NVFP4 or v0.23.x into production — separate decision after gate + 12h soak + user approval (ADR-0001; SPARK_ROADMAP §5.5).
- Host driver/kernel/firmware changes (none required — current 580.159.03 / 6.17.0-1021 is target).
- Qwen3.7 (unreleased), Grafana dashboard deploy (SPARK_ROADMAP §3.2), Windows-laptop known_hosts (davistroy handles).
- Production `qwen35` is sandbox-protected throughout (eval uses `/home/claude/llm-eval/docker-compose.qwen35.yml`).

## Risk Summary

| Change set | Top risk | Rollback |
|---|---|---|
| CS-A | new keys malformed → sshd restart fails | keep current session open as net; restore `/home/claude/ssh-hostkey-backup-20260630/`; Tailscale (100.124.10.120, unpinned) fallback |
| CS-B | prod disruption / v0.23.x ~12h cuDNN bug / NVFP4-MoE graph crash (#2776) / 4-bit quality drop | `restore_production.sh` (Entry 080 proven); enforce-eager fallback; **no adoption without 12h soak + approval** |
| CS-C2 | brief embed/NER downtime (~30–90 s) | self-healing on restart; verify health |
| CS-C3 | prune deletes a needed image | builder-prune-only first; curated image prune with explicit keep-list |

---

## Phase 0 — Safety Net & Pre-Flight  *(AUTO)*

### 0.1 Idle + state confirmation
**Status:** PENDING · **Mode:** AUTO
**Do:** Confirm GPU idle (`nvidia-smi` 0% / <15 W, no `/v1` POSTs in prior 15 min) before any disruptive step; `dpkg --audit` clean; all 8 containers healthy; record current per-proc swap snapshot.
**Acceptance:** Idle confirmed; baseline snapshot recorded in LAB_NOTEBOOK.
**Verify:** `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader`

### 0.2 Record current fingerprints & image keep-list
**Status:** PENDING · **Mode:** AUTO
**Do:** Record factory host-key fingerprints (ed25519 `SHA256:ah+SGtSN3Zetimeo9Y3+CtffPifiVOzJ9YuFUfkBKNM`, ecdsa `…T/P4tDRW…`, rsa `…sJFmblk…`) for change-verification; record `docker images` keep-list for CS-C3.
**Acceptance:** Both recorded.

---

## Phase 1 — CS-A: CVE-2026-24218 SSH host-key remediation  *(CLAUDE-OPS; DAVISTROY alt)*

> Confirmed applicable (Entry 081): host-key mtime `2025-09-22 23:46` predates fs-birth `23:49`; `root@localhost`; never regenerated → factory-shared. No scheduled automation SSHes to the Spark; only this VM (`spark.k4jda.net` + `192.168.10.33`) and the Windows laptop pin the key.

### 1.1 Backup existing host keys
**Status:** PENDING · **Mode:** CLAUDE-OPS
**Do:** `sudo mkdir -p /home/claude/ssh-hostkey-backup-20260630 && sudo cp -a /etc/ssh/ssh_host_* /home/claude/ssh-hostkey-backup-20260630/`
**Acceptance:** 6 files (3 priv + 3 pub) backed up. **Constitution C3 satisfied (backup before in-place change).**

### 1.2 Generate + install fresh keys
**Status:** PENDING · **Mode:** CLAUDE-OPS (recommended) **or** DAVISTROY (alt)
**Recommended (claude, NOPASSWD primitives):**
```
mkdir -p /tmp/newhk
ssh-keygen -q -t ed25519 -f /tmp/newhk/ssh_host_ed25519_key -N "" -C ""
ssh-keygen -q -t rsa -b 3072 -f /tmp/newhk/ssh_host_rsa_key -N "" -C ""
ssh-keygen -q -t ecdsa -b 256 -f /tmp/newhk/ssh_host_ecdsa_key -N "" -C ""
# validate before install:
for k in /tmp/newhk/ssh_host_*_key.pub; do ssh-keygen -lf "$k"; done
sudo cp /tmp/newhk/ssh_host_* /etc/ssh/
sudo chown root:root /etc/ssh/ssh_host_*
sudo chmod 600 /etc/ssh/ssh_host_*_key && sudo chmod 644 /etc/ssh/ssh_host_*_key.pub
```
**Alt (davistroy interactive):** `sudo rm /etc/ssh/ssh_host_* && sudo ssh-keygen -A`
**Acceptance:** New keys present, valid, root-owned, correct perms; fingerprints differ from 0.2.
**Decision point:** confirm CLAUDE-OPS vs DAVISTROY with user before running.

### 1.3 Restart sshd (session-safe)
**Status:** PENDING · **Mode:** CLAUDE-OPS
**Do:** `sudo systemctl restart ssh` (existing session survives — sshd forks per connection). **Keep current session open** as the safety net until 1.4 verifies.
**Acceptance:** `sudo systemctl is-active ssh` = `active`.
**Rollback:** if not active → `sudo cp -a /home/claude/ssh-hostkey-backup-20260630/ssh_host_* /etc/ssh/ && sudo systemctl restart ssh`.

### 1.4 Re-pin this VM + verify
**Status:** PENDING · **Mode:** AUTO (local, no sudo)
**Do:** `ssh-keygen -R spark.k4jda.net; ssh-keygen -R 192.168.10.33; ssh-keygen -R 192.168.10.32` then reconnect `ssh -o StrictHostKeyChecking=accept-new claude@spark.k4jda.net 'hostname && sudo systemctl is-active ssh'`; record new fingerprints.
**Acceptance:** Fresh connection succeeds; pinned fingerprint == server's new `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`; differs from factory.

### 1.5 Hand off laptop re-pin
**Status:** PENDING · **Mode:** DAVISTROY
**Do:** Provide davistroy the Windows re-pin one-liner (`ssh-keygen -R` for the 4 host aliases + reconnect-and-accept). Record completion.
**Acceptance:** Laptop reconnects cleanly to new keys.

---

## Phase 2 — CS-C (quick): kernel-log visibility, swap clear, build-cache reclaim

### 2.1 Add claude to systemd-journal/adm
**Status:** PENDING · **Mode:** DAVISTROY
**Do:** `sudo usermod -aG systemd-journal,adm claude` (one-time; `usermod` not in claude NOPASSWD). Takes effect next login.
**Acceptance:** `id claude | grep -o systemd-journal`; new session: `journalctl -k -n 5` works without sudo; closes the Entry-082 Xid-visibility gap.

### 2.2 Clear secondary-service swap
**Status:** PENDING · **Mode:** GATED (brief embed/NER downtime)
**Do:** During idle, restart the swap-holding secondary services (NOT qwen35): `docker compose -f /home/claude/docker-compose.yml restart qwen3-embed bge-m3 gliner ce-service`. (Or fold into CS-B Phase 3.6 full-stack ordered restart.) Respect startup-order rule where applicable.
**Acceptance:** Per-proc swap for those services < 100 MB; `free -h` swap materially reduced (was ~7.6 GiB).
**Verify:** re-run the per-`/proc/*/status` VmSwap scan from Entry 082.

### 2.3 Docker build-cache prune (safe)
**Status:** PENDING · **Mode:** AUTO
**Do:** `docker builder prune -f` (reclaims ~53 GB build cache; does NOT touch images).
**Acceptance:** `docker system df` build-cache reclaimable drops; all tagged images intact.

---

## Phase 3 — CS-B: NVFP4 + Arm C evaluation  *(GATED — idle box + supervision; ADR-0001)*

> Staged to separate **quant** (NVFP4 vs FP8) from **build** (v0.23.x vs v0.19.1). Mirrors Entry 080 Arm B method. Production stopped during eval, restored via `restore_production.sh`. **Eval ≠ adoption.**

### 3.0 Harness prep (no production impact)
**Status:** PENDING · **Mode:** AUTO
**Do:** (a) Download `nvidia/Qwen3.6-35B-A3B-NVFP4` (~20 GB) into the HF cache via throwaway no-GPU container. (b) Add `VLLM_MARLIN_USE_ATOMIC_ADD` env passthrough to `/home/claude/llm-eval/docker-compose.qwen35.yml` (+ entrypoint), backward-compatible, back up `*.bak-20260630`. (c) Optionally extend the MTP spec-config path to carry `moe_backend:triton` for MTP=3. (d) Author profiles (below).
**Acceptance:** Model cached; compose/entrypoint changes diffed & backed up; profiles created.
**Resolves:** U4 (flag plumbing).

### 3.1 B1 — NVFP4 cheap probe on CURRENT build
**Status:** PENDING · **Mode:** GATED
**Profile `nvfp4_curbuild.env`:** `LLM_IMAGE=vllm-cu132-test:latest`, `LLM_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4`, `LLM_KV_DTYPE=fp8`, `LLM_SPEC_TOKENS=3`, `LLM_EXTRA_ARGS=--moe-backend marlin --attention-backend flashinfer`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`.
**Do:** Stop prod qwen35; launch profile; **smoke only** — observe whether NVFP4+Marlin loads with FULL CUDA graph capture (not forced eager) and serves `/health` 200.
**Acceptance (decision, not gate):** Records one of {loads-with-full-graphs | needs-enforce-eager (#2776) | unsupported-on-this-build}. **Resolves U1/U2 cheaply.** If it runs, capture a quick c1/c8 datapoint to isolate quant-vs-build.
**Then:** restore production (or proceed to 3.2 if continuing same window).

### 3.2 B2 — Obtain v0.23.x image (Arm C)
**Status:** PENDING · **Mode:** GATED · **Depends On:** 3.1
**Do:** Acquire a v0.23.x build — pull eugr prebuilt (`prebuilt-vllm-current` + FlashInfer 0.6.13) or build from `~/spark-vllm-docker`. **Verify the image contains FlashInfer (#164)** and registers `modelopt_fp4` + Marlin sm_121 kernels. Tag locally (e.g. `vllm-cu132-test:v0.23x`). Keep `:pre-eugr-v0201` rollback intact.
**Acceptance:** Image present, FlashInfer confirmed, quant methods include `modelopt_fp4`. **Resolves U5.**

### 3.3 B3 — Faithful NVFP4 eval on v0.23.x
**Status:** PENDING · **Mode:** GATED · **Depends On:** 3.2
**Profile `nvfp4_v023x.env`:** as 3.1 but `LLM_IMAGE=vllm-cu132-test:v0.23x`, + `--async-scheduling --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors` per Poveda recipe; tune `LLM_GPU_UTIL` (NVFP4 weights ~½ FP8 → test 0.70, optionally 0.80, U7).
**Do:** `run_full_suite nvfp4_v023x` → c1/c4/c8/c16 + AR(30) + 30-min soak.
**Acceptance:** Suite completes; metrics + AR + soak captured. **Resolves U6.** Note prefix-caching is ON (config-level, not single-variable comparison — documented).

### 3.4 B4 — Arm C build control on v0.23.x (FP8 + MTP=2)
**Status:** PENDING · **Mode:** GATED · **Depends On:** 3.2
**Profile `prod_mtp2_v023x.env`:** `prod_mtp2_n2.env` with `LLM_IMAGE=vllm-cu132-test:v0.23x`.
**Do:** `run_full_suite prod_mtp2_v023x` → attributes gains to **build** vs **quant**; fulfils SPARK_ROADMAP §5.3 Arm C.
**Acceptance:** Suite captured; build-vs-baseline delta computed.

### 3.5 B5 — Gate + decision matrix + ADR resolution
**Status:** PENDING · **Mode:** GATED (decision)
**Do:** Build the `{current, v0.23.x} × {FP8-MTP2, NVFP4}` matrix vs MTP baseline 406.9. Apply gates: **c8 ≥ 427.2 (+5%)**, **AR ≥ 28/30**, clean 30-min soak. Update ADR-0001 status (Accepted/Rejected). If a candidate passes, recommend **12h soak + user approval** before adoption (do NOT switch production here).
**Acceptance:** Decision recorded (LAB_NOTEBOOK Entry; ADR-0001 resolved); CLAUDE.md verified-rule amended if NVFP4 viable.

### 3.6 B6 — Restore production + verify
**Status:** PENDING · **Mode:** GATED · **Depends On:** 3.5
**Do:** `restore_production.sh`; verify qwen35 `/health` 200, model `Qwen3.6-35B-A3B-FP8`, MTP=2, 0 restarts; full-stack health in startup order (folds in 2.2 swap clear). Retain eval artifacts + harness changes (`*.bak-20260630`).
**Acceptance:** Live config == Entry 073 production; all 8 containers healthy.

---

## Phase 4 — CS-C3b: curated image prune  *(AUTO; post-eval)*

### 4.1 Curated docker image prune
**Status:** PENDING · **Mode:** AUTO · **Depends On:** Phase 3 complete
**Do:** `docker image prune` (dangling) + explicit removal of confirmed-stale tagged images only. **Keep-list:** `vllm-cu132-test:latest` (prod), `:pre-eugr-v0201` (rollback), `:v0.23x` (Arm C, if kept), `vllm-node:latest`, embedding/support images. Candidate-remove: `:glm47`, `eugr-vllm-0192/0201` (superseded) — confirm before removing.
**Acceptance:** ~100+ GB reclaimed; every keep-list image still present; `docker compose` stack unaffected.
**Verify:** `docker system df`; `docker images`; `docker compose -f /home/claude/docker-compose.yml ps` all healthy.

---

## Unknowns Register

| ID | Unknown | Severity | Resolved by | Resolution strategy |
|---|---|---|---|---|
| U1 | NVFP4+Marlin runs with full CUDA graphs on our build / v0.23.x? | High | 3.1, 3.3 | cheap probe before build investment |
| U2 | flashinfer #2776 NVFP4-MoE graph-capture crash bites us? | High | 3.1, 3.3 | enforce-eager fallback in smoke |
| U3 | v0.23.x ~12h cuDNN graph-corruption fixed? | Med | (adoption only) | 12h soak before any production switch |
| U4 | `--moe-backend marlin` flag valid + env/MTP=3 plumbing | Med | 3.0 | smoke resolved-command log |
| U5 | obtain v0.23.x: pull vs build; FlashInfer present (#164) | Med | 3.2 | verify image before B3 |
| U6 | NVFP4 4-bit quality vs FP8 | Low | 3.3 | AR(30) gate + optional expanded suite |
| U7 | gpu_mem_util tuning for NVFP4 (smaller weights) | Low | 3.3 | test 0.70/0.80 |
| U8 | CVE: new keys malformed → sshd fail | Low | 1.2/1.3 | validate `ssh-keygen -lf` + keep session open |

## Definition of Done (Runnable)

| Change set | Check | Command |
|---|---|---|
| CS-A | sshd active | `ssh … 'sudo systemctl is-active ssh'` → `active` |
| CS-A | key rotated | `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` ≠ `SHA256:ah+SGtSN…` |
| CS-A | client re-pinned | fresh `ssh claude@spark.k4jda.net hostname` succeeds without prompt |
| CS-B | NVFP4 loads | startup log: `modelopt_fp4` + Marlin MoE + FlashInfer + graph capture (not eager unless #2776) |
| CS-B | throughput gate | `run_full_suite` c8 agg **≥ 427.2** |
| CS-B | quality gate | AR ≥ 28/30 |
| CS-B | soak | `soak_test.py` 30-min 100% / 0 err / 0 restart |
| CS-B | prod restored | `docker inspect qwen35` model = `Qwen3.6-35B-A3B-FP8`, MTP=2 |
| CS-C | swap cleared | secondary-service VmSwap < 100 MB |
| CS-C | reclaim | `docker system df` build-cache + dangling reclaimed; keep-list intact |
| CS-C | journal | `id claude \| grep systemd-journal` + `journalctl -k -n5` no-sudo |

## Implementation Sequence
1. **Phase 0** (pre-flight) → **Phase 1 CS-A** (CVE, ~15 min, verify access) — do first.
2. **Phase 2** quick wins — 2.1 (davistroy usermod), 2.3 (builder prune) anytime; 2.2 (swap) at idle or fold into 3.6.
3. **Phase 3 CS-B** — supervised idle window (multi-hour; prod down ~2–4 h, restored): 3.0→3.6. **Needs user-approved window.**
4. **Phase 4** curated image prune — after CS-B settles.

## Generated ADRs
- `docs/adr/ADR-0001-nvfp4-sm121-quantization.md` — Proposed (resolved by Phase 3.5).
