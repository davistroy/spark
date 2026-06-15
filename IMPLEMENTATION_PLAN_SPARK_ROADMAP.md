# Implementation Plan — DGX Spark Roadmap (P0–P4)

**Generated:** 2026-06-11
**Completed:**
**Based On:** spark-recon Entry 075 + live-state investigation Entry 076 (2026-06-11); ultra-plan Phases 0–5 (this session).

> **Execute with:** `/implement-plan --input IMPLEMENTATION_PLAN_SPARK_ROADMAP.md` — but **only Phases 1–2 are autonomous-agent-executable.** Phases 3–6 are operational/human-gated (guided SSH, physical-console reboot, production downtime, approval gates). See the Execution Mode legend; do not point autonomous orchestration at the guided phases.

---

## Scope

Covers the five roadmap items from the 2026-06-11 recon through the eval-adoption *decision*:
- **P0** MTP/#37754 stability triage — **already resolved CLEAN** (Entry 076); residual = add Xid visibility.
- **P1** Kernel/driver maintenance (6.17.0-1021 + driver 580.159.03) + re-baseline.
- **P2** vLLM 0.22.x / DFlash evaluation under gated adoption.
- **P3** Contingent backlog (Qwen3.7, AutoRound, Atlas, NVFP4) — trigger-gated only.
- **P4** Tooling (spark-recon skill fixes) + observability.

## Out of Scope
- Model switching to Qwen3.7 (trigger-gated; planned when open weights land).
- Autonomous execution of Atlas/AutoRound experiments (entry criteria only).
- Multi-node / cluster topologies.
- Homeserver work beyond creating new Grafana objects.
- Editing the stale `IMPLEMENTATION_PLAN.md` (May 6) or completed `IMPLEMENTATION_PLAN_MODEL_EVAL.md`.

---

## Pre-Plan Gates (Phase 0 Constitution — every work item must comply)

| Gate | Rule |
|------|------|
| **Reboot class** | Kernel/driver reboot = "unrecoverable without physical access" → STOP, require explicit physical-console confirmation; never evenings/weekends; check MOK/SecureBoot; `dpkg --audit` + `dkms status` clean first. |
| **Production lock** | qwen35 (Qwen3.6-35B-A3B-FP8 pre-quant, BF16 KV, batched-tokens 32768, MTP=2) untouched until eval gates pass AND user approves. Never add `--kv-cache-dtype fp8`/`--quantization fp8`. Rollback = `cp /home/claude/docker-compose.yml.pre-fp8prequant …`. |
| **Container ops** | Read `spark-device.md` first; change ONLY one parameter vs documented command; show diff before running; absolute mount paths; post-restart watch GPU mem >3 GB in 60s + `/health` 200. |
| **Driver series** | Stay on 580.x. 590 = GB10 UMA leak regression, unsupported on Spark. |
| **External state** | Grafana: never modify/delete existing dashboards (`spark-monitor`, `spark-inference`) — new UIDs only. Homeserver HTTP via `wget`, not curl. Back up non-git state before changing. |
| **Caches** | Separate Triton cache per build; never mix. HF cache absolute paths only. |
| **Access** | Remote as `claude@spark`. NOPASSWD covers docker/systemctl/modprobe/reboot/apt/dpkg/cp/mv/etc. `journalctl`/`dmesg`/`fwupdmgr` are NOT NOPASSWD (use `sudo cp` kern.log workaround for logs). |
| **Debugging** | Diagnose before changing; one variable at a time; STOP after 2 failed attempts and present analysis. |
| **Definition of done** | Every step logged to LAB_NOTEBOOK.md immediately; learnings → CLAUDE.md + memory; failures documented in detail. Script memory ≤1.5 GB RSS (stream logs). |

**Compliance status:** All phases below comply. One flagged item carried into Phase 1 (the `Current Config` corrections require user confirmation — the constitution reserves that section to the user).

---

## Execution Mode legend

| Mode | Meaning | implement-plan? |
|------|---------|-----------------|
| **AUTO** | Code/doc edits in a git repo; subagent-implementable, testable, committable. | Yes |
| **GUIDED-SSH** | Live read/standard ops on the Spark over SSH; reversible; user should be watching. | No — guided session |
| **PHYSICAL** | Requires user at the physical console (reboot). | No — scheduled with user |
| **GATED** | Production downtime + benchmark + explicit user adoption approval. | No — gated session |

## Execution Hints

```
default_model: sonnet
phase_overrides:
  "Phase 1: Documentation Truth-Up": haiku
  "Phase 2: spark-recon Tooling Fixes": sonnet
# Phases 3-6 are non-AUTO; model tiers N/A (human-driven).
```

---

# Phase 1: Documentation Truth-Up (CS0)

**Objective:** Reconcile project docs with the live-state ground truth from Entry 076. **Execution Mode: AUTO.**
**Note:** Several CS0 items already landed this session (Entry 076 written; CLAUDE.md spec-flag + FLASH_ATTN bullets corrected; #37754 watch item re-tagged CLEAN; baseline tracking values updated). This phase finishes the remainder.

### 1.1 Correct SPARK_BASELINE.md `Current Config` section ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15 — user confirmed (U-7); Current Config rewritten to live docker-inspect state (pre-quant FP8 model, FLASH_ATTN backend, BF16 KV, `--speculative-config` JSON, Entry 073 throughput, 0-restart stability).
**Model Tier:** haiku
**Depends On:** none
**Execution Mode:** AUTO — but **requires user confirmation** (constitution reserves `Current Config` to the user).
**Tasks:**
- Replace stale rows with live `docker inspect` values (Entry 076): `model` → `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quant, Entry 073); `attention_backend` → `FLASH_ATTN` (auto-selected; FlashInfer MoE-only via `VLLM_FLASHINFER_MOE_BACKEND=latency`); `kv_cache` → BF16/auto, 504,912 tokens @131K; spec flags → `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`; remove on-the-fly `--quantization fp8` references.
**Acceptance Criteria:** `Current Config` matches `docker inspect qwen35` output; no on-the-fly-FP8 or FlashInfer-attention claims remain.

### 1.2 Fix EVAL_STUDY_STATUS.md stale line ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15
**Model Tier:** haiku
**Depends On:** none
**Execution Mode:** AUTO
**Tasks:** Update the "Production NOT changed — awaiting user approval" line to reflect the 2026-05-18 switch (Entry 073, applied).
**Acceptance Criteria:** No line claims production is unchanged; cross-references Entry 073.

### 1.3 Commit the session's doc changes to a feature branch ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15
**Model Tier:** sonnet
**Depends On:** 1.1, 1.2
**Execution Mode:** AUTO
**Tasks:** Branch off main (`feature/spark-roadmap-2026-06`); stage CLAUDE.md, LAB_NOTEBOOK.md, SPARK_BASELINE.md, EVAL_STUDY_STATUS.md + untracked recon docs (IMPLEMENTATION_PLAN_MODEL_EVAL.md, MODEL_EVALUATION_2026_05.md, this plan); commit with a descriptive message; do NOT commit on main.
**Acceptance Criteria:** Clean tree on a feature branch; `git log` shows the doc-truth-up commit; main untouched.

<!-- BEGIN DOD -->
### Definition of Done (Runnable)
- **tree-clean**: `git status --short` — pass: empty after commit
- **branch-not-main**: `git rev-parse --abbrev-ref HEAD` — pass: not `main`/`master`
- **no-stale-onthefly**: `grep -c "on-the-fly" SPARK_BASELINE.md` Current Config block — pass: 0 in that section
<!-- END DOD -->

### Phase 1 Completion Checklist
- [ ] `Current Config` reflects live `docker inspect` (user-confirmed)
- [ ] EVAL_STUDY_STATUS.md no longer claims production unchanged
- [ ] Feature branch created; doc changes committed off main

### Testing Requirements
- Doc-only; verification = grep checks above + visual diff review.

---

# Phase 2: spark-recon Tooling Fixes (CS4)

**Objective:** Fix the drifted spark-recon skill config in the plugin source repo. **Execution Mode: AUTO.**
**Repo:** `~/.claude/plugins/marketplaces/troys-plugins/plugins/personal-plugin/` (git, `github:davistroy/claude-marketplace`, v9.2.0). **Separate repo from the spark project.**

### 2.1 Correct stale Machine Config + endpoints in spark-recon SKILL.md ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15
**Model Tier:** sonnet
**Depends On:** none
**Execution Mode:** AUTO
**Tasks (confirmed line numbers from source):**
- L57 `current_model: "Qwen/Qwen3.5-35B-A3B"` → `"Qwen/Qwen3.6-35B-A3B-FP8"`.
- L58 `quantization: "FP8 on-the-fly"` → `"FP8 pre-quantized (native)"`.
- L46 + L139–141: drop forum category **720** (permanently removed; merged into 719/721). Update `check5_source` and the Check 5 endpoint list to 719 + 721 only; note 719 aggregates everything.
- Check 1: document the Firestore `benchmarks`-collection REST path as the preferred access method (App-Check gate bypassed; tracking unfrozen 2026-06-11).
- L264 `qwen_current_model` (and any other Qwen3.5 references): update to Qwen3.6-35B-A3B-FP8.
**Acceptance Criteria:** `grep -nE 'Qwen3.5-35B-A3B|on-the-fly|720\.json' SKILL.md` returns 0 stale hits in Machine Config / endpoint sections.

### 2.2 Grep plugin for the same stale values (spark-audit likely shares them) ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15 — found + fixed spark-audit (pre-quant anti-pattern + FLASHINFER expectation); no other siblings affected.
**Model Tier:** sonnet
**Depends On:** 2.1
**Execution Mode:** AUTO
**Tasks:** `grep -rnE 'Qwen3.5-35B-A3B|FP8 on-the-fly|720\.json' ~/.claude/plugins/marketplaces/troys-plugins/plugins/personal-plugin/`; apply the same corrections to spark-audit and any other skill that shares them. Do NOT alter unrelated machine configs (jetson, etc.).
**Acceptance Criteria:** No stale Qwen3.5/on-the-fly/720 references remain in the personal-plugin spark skills.

### 2.3 Validate + version bump + publish ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15 — 9.2.0→9.3.0 (plugin.json + marketplace.json + CHANGELOG); focused validation passed; PR #95 on davistroy/claude-marketplace (not merged — no --auto-merge). Cache refresh happens post-merge.
**Model Tier:** sonnet
**Depends On:** 2.1, 2.2
**Execution Mode:** AUTO
**Tasks:** Run `/personal-plugin:validate-plugin`; `/personal-plugin:bump-version` to 9.3.0 with CHANGELOG entry; commit + push the marketplace repo; refresh the installed cache.
**Acceptance Criteria:** validate-plugin passes; plugin.json at 9.3.0; CHANGELOG notes the spark-recon config fix; cache reflects 9.3.0.

<!-- BEGIN DOD -->
### Definition of Done (Runnable)
- **no-stale-model**: `grep -rcE 'Qwen3.5-35B-A3B|720\.json' <plugin spark skills>` — pass: 0
- **validate**: `/personal-plugin:validate-plugin` — pass: no errors
- **version**: `grep '"version"' <plugin.json>` — pass: `9.3.0`
<!-- END DOD -->

### Phase 2 Completion Checklist
- [ ] spark-recon SKILL.md model/quant/endpoints corrected
- [ ] spark-audit + siblings checked and fixed
- [ ] validate-plugin passes; version bumped to 9.3.0; pushed; cache refreshed

### Testing Requirements
- `/personal-plugin:validate-plugin` clean; a spark-recon dry-run hits only 719/721 (no 720 404).

---

# Phase 3: Observability — Xid/NVRM Alerting (CS1)

**Objective:** Close the monitoring blind spot (nothing on the box surfaces Xid today). **Execution Mode: GUIDED-SSH.** No reboot, no production-container stop.
**Pre-req finding (Entry 076):** `/opt/gpu-exporter/gpu_exporter.py` (systemd, :9400) emits nvidia-smi-style gauges only — zero Xid metrics. node-exporter on :9100. Grafana on homeserver:3050 (12.4.2), Prometheus datasource UID `PBFA97CFB590B2093`. No alert rules exist.

### 3.1 Back up and extend the GPU exporter ✅ Completed 2026-06-15
**Status:** COMPLETE 2026-06-15 — `/opt/gpu-exporter/gpu_exporter.py` extended with `gpu_xid_events_total`, `nvrm_alloc_failures_total`, `spark_qwen35_restart_count`, `spark_qwen35_running`; original backed up to `gpu_exporter.py.bak-20260615`; service active; verified end-to-end via Prometheus (job `spark-gpu`, `gpu_xid_events_total=0`). nvrm count shows 37 historical (May eval-window; ages out on rotation).
**Execution Mode:** GUIDED-SSH
**Depends On:** none
**Tasks:** `sudo cp /opt/gpu-exporter/gpu_exporter.py{,.bak-20260611}`; add two counters parsed from `/var/log/kern.log` (stream, no slurp; rotation-tolerant): `gpu_xid_events_total`, `nvrm_alloc_failures_total`; `sudo systemctl restart gpu-exporter`.
**Acceptance Criteria:** `curl -s localhost:9400/metrics | grep -E 'gpu_xid_events_total|nvrm_alloc_failures_total'` returns both; existing gauges intact; service active.
**Rollback:** restore `.bak-20260611`, restart.

### 3.2 Add Grafana dashboard (dashboard-only per U-2; new objects only) ⏸ BLOCKED — Grafana 13 credentials
**Status:** BLOCKED 2026-06-15 — dashboard **authored + committed** (`grafana/spark-reliability-dashboard.json`, uid `spark-reliability`, 8 panels, dashboard-only per U-2: Xid/NVRM/restart stat tiles + power/clock/temp/fault timeseries). **Cannot deploy:** Grafana upgraded 12.4.2→**13.0.1**; stored `admin:Spark2026!` returns 401; no Bitwarden entry found. Prometheus already scrapes the metrics, so import is one step once creds are provided.
**Execution Mode:** GUIDED-SSH (deploy step blocked on creds)
**Depends On:** 3.1 (done) + Prometheus scrape confirmed (done) + **current Grafana 13 credentials or a service-account token (NEW BLOCKER U-8)**
**Deploy when unblocked (datasource UID `PBFA97CFB590B2093`, never touch existing dashboards):** UI Import the committed JSON, OR `wget`/curl POST `{"dashboard": <json>, "folderUid": "dfhdwahqbii9sc", "overwrite": false}` to `/api/dashboards/db`. uid `spark-reliability` confirmed not pre-existing — safe with `overwrite:false`.
**Acceptance Criteria:** dashboard live with unique uid; panels render against the new metrics; no existing dashboard modified.

### Phase 3 Completion Checklist
- [ ] Exporter emits Xid + NVRM counters; existing gauges intact; backup saved
- [ ] Prometheus scrapes the new metrics
- [ ] 3 alert rules live (new UIDs); test-fire confirmed; existing dashboards untouched

### Testing Requirements
- `curl localhost:9400/metrics | grep -E 'xid|nvrm'`; Prometheus query returns series; Grafana rule test-fire.

---

# Phase 4: Kernel/Driver Maintenance + Re-Baseline (CS2)

**Objective:** 6.17.0-1014 → 6.17.0-1021 + driver 580.142 → 580.159.03 (Copyfail/Dirtyfrag patch + GB10 OOM improvements), then re-baseline. **Execution Mode: PHYSICAL — requires user at the console.** Weekday daytime only.
**Pre-req findings (Entry 076):** dkms empty, `dpkg --audit` clean, SecureBoot on, **no pending firmware** (apt-only window), 230 packages upgradable, 2.0 T disk free, 1014 nvidia modules retained (GRUB fallback intact). Depends On: Phase 3 (alerting live before reboot).

### 4.1 Day-of pre-flight ✅ Completed 2026-06-15 (read-only)
**Status:** COMPLETE 2026-06-15 — READY with one critical caveat. Idle 0/0; dkms empty; `dpkg --audit` clean; SecureBoot on; compose backups present; all 8 containers healthy. Targets available: kernel **6.17.0-1021.21**, driver **580.159.03-0ubuntu0.24.04.1** (stays on 580.x; module 1021 candidate present, Installed:none). Sim: 240 upgraded, 10 new, 4 removed. **CRITICAL FINDING** → see 4.2 hazard.
**Execution Mode:** GUIDED-SSH
**Tasks (done):** confirmed idle; dkms/dpkg clean; `apt -s dist-upgrade` reviewed (removals flagged below); compose backups present. **Still required before 4.2:** `dpkg -l > /home/claude/pre-upgrade-dpkg.txt` snapshot + **explicit physical-console confirmation from user.**
**Acceptance Criteria:** all green; user confirms physical presence. (Green except the documented module-removal hazard.)

### 4.2 Execute upgrade + reboot
**Status:** PENDING (PHYSICAL — user at console)
**Execution Mode:** PHYSICAL
**Depends On:** 4.1
**⚠ HAZARD (discovered by 4.1 sim):** `dist-upgrade` will **REMOVE the running kernel's nvidia module** (`linux-modules-nvidia-580-open-6.17.0-1014-nvidia`) AND does **not** install the new kernel's module (1021 shows Installed:none). A naive `dist-upgrade && reboot` leaves BOTH kernels without a working nvidia module → GPU dead on boot, GRUB fallback to 1014 also broken → physical recovery required.
**Tasks (revised, hazard-aware order):**
1. `sudo apt update && sudo apt dist-upgrade`
2. **Before any reboot — install the new kernel's module:** `sudo apt install linux-modules-nvidia-580-open-6.17.0-1021-nvidia` (candidate 6.17.0-1021.21 confirmed; signed prebuilt, Secure-Boot safe).
3. **Preserve the 1014 fallback:** if step 1 removed it, `sudo apt install linux-modules-nvidia-580-open-6.17.0-1014-nvidia` so GRUB fallback stays bootable until 1021 is proven.
4. `fwupdmgr refresh && fwupdmgr get-updates` (expect none — no firmware pending as of 2026-06-15).
5. Confirm BOTH 1021 and 1014 have an nvidia module on disk, THEN reboot with user watching.
**Acceptance Criteria:** clean reboot into 6.17.0-1021 with a working nvidia module loaded; 1014 fallback still bootable.

### 4.3 Post-reboot verification
**Status:** PENDING
**Execution Mode:** GUIDED-SSH
**Depends On:** 4.2
**Tasks:** `uname -r`=6.17.0-1021; `nvidia-smi` driver=580.159.03 + sane clocks/power under a test load (14W-throttle check); `mokutil --sb-state` enabled; `ip route` metrics 700/600 intact; all 8 containers healthy; qwen35 `/health` 200 + GPU mem growth; Phase 3 alerts green.
**Acceptance Criteria:** all checks pass; no Xid post-boot.
**Rollback:** GRUB previous kernel (1014 modules on disk); driver downgrade from `pre-upgrade-dpkg.txt`.

### 4.4 Re-baseline (supersedes suspect Entry 052)
**Status:** PENDING
**Execution Mode:** GATED (production stop for bench)
**Depends On:** 4.3
**Tasks:** `run_full_suite.sh current_baseline` (bench stages: throughput c1/4/8/16 ×3 + AR 30); write Entry 078; `restore_production.sh`.
**Acceptance Criteria:** fresh baseline recorded; AR ≥28/30; production restored + healthy. This becomes the Phase 5 comparison anchor.

### Phase 4 Completion Checklist
- [ ] Pre-flight green + physical-console confirmed
- [ ] Kernel 6.17.0-1021 + driver 580.159.03 + matching nvidia module
- [ ] Post-reboot checks pass (incl. no 14W throttle, routing intact, alerts green)
- [ ] Re-baseline recorded (Entry 078); production restored

### Testing Requirements
- `uname -r`, `nvidia-smi`, `mokutil --sb-state`, `dpkg --audit`, `/health` 200, `run_full_suite.sh current_baseline`.

---

# Phase 5: vLLM 0.22.x + DFlash Eval Campaign (CS3)

**Objective:** Evaluate the +20% DFlash recipe and the v0.22.x build under hard gates; adopt only on user approval. **Execution Mode: GATED.** Sandbox; production untouched until adoption. Each arm = separate session; 12h soaks run unattended under Phase 3 alerting.
**Pre-req findings (Entry 076):** DFlash supported in the current image (no build upgrade needed for Arm B); drafter `z-lab/Qwen3.6-35B-A3B-DFlash` NOT cached (~1.2 GB download); harness + profiles + `restore_production.sh` intact; eugr 0.19x/0.20x images present, 0.22 needs adapting to single-node TP=1. Depends On: Phase 4 (evals run on post-update kernel, anchored to Entry 078).

### 5.1 Arm B — DFlash n8 on current build (one variable)
**Status:** PENDING · **Execution Mode:** GATED · **Depends On:** 4.4
**Tasks:** new profile `qwen36_fp8_dflash_n8.env`, sole diff vs `current_baseline.env`: `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":8}'` (revision-pinned). smoke → throughput → AR 30 → 1h soak; promising → 12h soak.
**Acceptance Criteria:** throughput vs Entry 078; AR ≥28/30; DFlash acceptance recorded; zero Xid/restart in soak.

### 5.2 Arm B3 — Arm B + prefix caching (resolve the ON/OFF discrepancy)
**Status:** PENDING · **Execution Mode:** GATED · **Depends On:** 5.1
**Tasks:** add `--enable-prefix-caching`; re-run AR + a ≥200-token shared-prefix accuracy probe (Arena recipe ON vs eugr-removed-for-accuracy).
**Acceptance Criteria:** prefix-caching accuracy impact quantified; keep ON only if AR holds.

### 5.3 Arm C — eugr 0.22.1 build (build variable)
**Status:** PENDING · **Execution Mode:** GATED · **Depends On:** 5.1
**Tasks:** adapt eugr 0.22.1 stack to single-node TP=1 with our exact prod flags; fresh Triton cache `/home/claude/.cache/triton-eugr022`; prod-equivalent config to isolate build effect; re-validate pre-quant FP8 under #41215. Watch for the ~12h cuDNN graph-corruption bug (fixed only in v0.23.0 — if it reproduces, stop and wait for 0.23).
**Acceptance Criteria:** build effect measured vs Entry 078; pre-quant-vs-on-the-fly verdict re-confirmed for 0.22; soak stability.

### 5.4 Arm D — winning spec method on winning build (+ optional Arm E NVFP4)
**Status:** PENDING · **Execution Mode:** GATED · **Depends On:** 5.2, 5.3
**Tasks:** combine best spec method + best build; optional Arm E: NVFP4 modelopt (RedHatAI, ~22 GB) on the 0.22 stack.
**Acceptance Criteria:** single best config identified with full c1/4/8/16 + AR + 12h soak.

### 5.5 Gated adoption decision
**Status:** PENDING · **Execution Mode:** GATED (user approval) · **Depends On:** 5.4
**Adoption gates (ALL):** ≥+5% c8 vs Entry 078 re-baseline; AR ≥28/30; 12h soak zero Xid/restart. Then present to user. On approval: show one-param compose diff first; `.pre-<change>` backup; staged restart per pre-flight; Entry + CLAUDE.md updates. `restore_production.sh` after every session regardless.
**Acceptance Criteria:** explicit user adoption decision recorded; production either switched (gates passed + approved) or unchanged with rationale.

### Phase 5 Completion Checklist
- [ ] Arms B / B3 / C / D (/E) run with full suites under alerting
- [ ] All three adoption gates evaluated against Entry 078
- [ ] User adoption decision recorded; production restored/healthy either way

### Testing Requirements
- Per arm: `run_full_suite.sh <profile>`, `run_ar_tasks.py` ≥28/30, `soak_test.py` 12h zero-alert, `restore_production.sh` exit 0 + `/health` 200.

---

# Phase 6: Contingent Backlog (CS5)

**Objective:** Trigger-gated experiments — **no scheduled work; do nothing until entry criteria fire.** **Execution Mode: GATED.**

### 6.1 Qwen3.7 27B/35B open weights → benchmark day
**Status:** PENDING (trigger) · **Trigger:** official Qwen HF org publishes Qwen3.7 27B or 35B weights (check weekly through mid-July; ignore `RscriptSQwen` squat). **Action:** new harness profile → full throughput + AR suite vs production.

### 6.2 AutoRound W4A16 INT4
**Status:** PENDING (trigger) · **Trigger:** Phase 5 complete AND (published 35B W4A16 checkpoint OR free 6h GPU window). **Action:** verify CUDA-graph capture vs Entry 068 gs=32 hang first (gs=128 may differ); bench vs FP8.

### 6.3 Atlas runtime
**Status:** PENDING (trigger) · **Trigger:** single-stream latency becomes a real workload need. **Action:** AGPLv3 license review; post-2026-06-02 image; sandbox; note long-context/c≥2 degradation + tool-call corruption history.

### 6.4 NVFP4 re-bench
**Status:** PENDING (trigger) · **Trigger:** rides on Phase 5 Arm E if run. **Action:** modelopt + marlin MoE; compare to FP8 prod.

---

## Risk Mitigation

| Risk | Phase | Severity | Mitigation | Status |
|------|-------|----------|------------|--------|
| Boot failure without console | 4 | High | Physical presence required; GRUB fallback (1014 modules retained, verified); signed prebuilt modules; no DKMS/MOK exposure | Open |
| 230-package collateral in dist-upgrade | 4 | Med | `apt -s` simulate + `dpkg -l` snapshot day-of; review removals/holds before commit | Open |
| 14W/513MHz throttle post-reboot | 4 | Med | Test-load GPU check in 4.3; fix = wall power-cycle (not reboot) | Open |
| Kernel-selection verdict flip on 0.22 (Entry 054→070 precedent) | 5 | Med | Re-validate pre-quant vs on-the-fly in Arm C; don't assume | Open |
| ~12h cuDNN graph-corruption bug on 0.22 | 5 | Med | 12h soak gate; if reproduced, halt and target v0.23.0 | Open |
| Production downtime per eval arm | 5 | Med | Sandbox + `restore_production.sh` every session; user idle-window scheduling | Open |
| Exporter breakage | 3 | Low | `.bak-20260615` + immediate `curl` verify | Mitigated (deployed clean; backup saved; Prometheus confirms) |
| dist-upgrade removes the running kernel's nvidia module → both kernels left without GPU module | 4 | **High** | 4.1 sim caught it; 4.2 now installs the 1021 module AND retains the 1014 module BEFORE reboot; physical console + GRUB fallback | Open (mitigation documented) |
| Grafana 13 upgrade invalidated stored creds → cannot deploy dashboard | 3 | Low | Dashboard JSON committed for one-click import; deploy on creds (U-8) | Open |
| Touching a user Grafana dashboard | 3 | High | New UIDs only; never PUT existing dashboards | Open |
| Arena +20% not portable to shared-GPU stack (we run gpu_util 0.70 / 3 models vs Arena solo 0.85) | 5 | Low | Treat +20% as ceiling; gate on +5% c8 floor vs Entry 078 | Open |

## Unknowns Register

| ID | Unknown | Severity | Affects | Resolution |
|----|---------|----------|---------|------------|
| U-1 | DFlash runtime behavior on SM121/v0.19.1 (code present ≠ tested) | Med | 5.1 | Arm B smoke test |
| U-2 | Alert-notification routing (email/push vs dashboard-only) | Low | 3.2 | RESOLVED 2026-06-15 — dashboard-only (no notification routing); dashboard authored accordingly |
| U-8 | Current Grafana 13 credentials / service-account token | Low | 3.2 | **User input needed** — Grafana 12.4.2→13.0.1; stored creds 401; blocks dashboard deploy |
| U-3 | eugr 0.22 single-node TP=1 adaptation effort | Med | 5.3 | Timebox 3h; precedent images exist |
| U-4 | dist-upgrade collateral specifics | Med | 4.1 | `apt -s` review day-of |
| U-5 | User idle windows for eval downtime + current spark-llm consumers | Low | 5.x | Confirm before scheduling arms |
| U-6 | Physical-access scheduling for the reboot | Low | 4.2 | Weekday daytime slot with user |
| U-7 | `Current Config` correction approval | Low | 1.1 | RESOLVED 2026-06-15 — user approved; applied |

## Implementation Sequence

`Phase 1 (AUTO)` → `Phase 2 (AUTO, parallelizable with 1)` → `Phase 3 (GUIDED-SSH)` → `Phase 4 (PHYSICAL, scheduled)` → `Phase 4.4 re-baseline` → `Phase 5 arms B→B3→C→D(→E)` → `Phase 5.5 gated adoption` → `Phase 6 dormant (triggers)`.

Phases 1–2 are independent of each other (different repos) and can run concurrently. Phase 3 should land before Phase 4 (alerting before reboot) and before Phase 5 (free soak instrumentation). Phase 5 hard-depends on Phase 4.4's re-baseline as its comparison anchor.

## Verification Command Index

- **P1:** `git status --short`; `git rev-parse --abbrev-ref HEAD`; grep stale-term checks.
- **P2:** `/personal-plugin:validate-plugin`; `grep -rcE 'Qwen3.5-35B-A3B|720\.json'`; version grep.
- **P3:** `curl -s localhost:9400/metrics | grep -E 'xid|nvrm'`; Prometheus series query; Grafana rule test-fire.
- **P4:** `uname -r`; `nvidia-smi --query-gpu=driver_version,power.draw,clocks.sm --format=csv`; `mokutil --sb-state`; `dpkg --audit`; `curl localhost:8000/health`; `run_full_suite.sh current_baseline`.
- **P5:** `run_full_suite.sh <profile>`; `run_ar_tasks.py` (≥28/30); `soak_test.py` 12h; `restore_production.sh`.

## Generated ADRs
None — L2–L3 operational scope; durable decisions live in LAB_NOTEBOOK + CLAUDE.md Verified Rules per Entry-073 precedent.
