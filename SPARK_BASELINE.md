# Spark Performance Baseline

Last updated: 2026-06-11 (Entry 075 — spark-recon)
Last recon: 2026-06-11 (Entry 075)
Last audit: 2026-05-09 (Entry 062)
Last benchmark: 2026-05-18 (Entries 070-072 — three-model comparative eval, 50h study)

## Production Switch APPLIED (2026-05-18, Entry 073)

**Switched from on-the-fly FP8 to pre-quant FP8.** Measured gains exceeded predictions.

| Metric | Pre-switch (2026-04-30) | Post-switch (2026-05-18) | Delta |
|--------|-------------------------|--------------------------|-------|
| model | `Qwen/Qwen3.6-35B-A3B` (BF16 + `--quantization fp8`) | `Qwen/Qwen3.6-35B-A3B-FP8` (native FP8) | pre-quantized |
| kv_cache_dtype | fp8 | auto (BF16) | gotcha #1 eliminated |
| max_num_batched_tokens | 4096 | 32768 | +8x batch budget |
| c=1 tok/s | 59.2 | **66.9** | **+13.0%** |
| c=4 aggregate | 159.0 | **198.9** | **+25.1%** |
| c=8 aggregate | 373.8 | **427.7** | **+14.4%** |
| c=16 aggregate | 525.0 | **678.7** | **+29.3%** |
| KV cache tokens @ 131K | 1,123,584 | 504,912 | **-55%** (BF16 KV uses 2× memory/token) |
| Max concurrency @ 131K | 8.57x | 3.85x | -55% |
| AR pass rate | 28/30 | 28/30 (Phase B sub-run 3) | equivalent |
| MTP acceptance | 80% (pos0 89%, pos1 71%) | 80% (same MTP=2 config) | equivalent |
| 4h stability | (parity 100% / 0 errors) | 15,160 reqs / 100% / 0 errors / 0 drift | equivalent |
| Cold start | ~6 min | 7.25 min (fresh Triton cache) | slightly slower one-time |

**Rollback:** `cp /home/claude/docker-compose.yml.pre-fp8prequant /home/claude/docker-compose.yml && docker compose stop qwen35 && docker compose up -d qwen35` (~6 min). Backup file preserves the exact prior config.

**Contradicts Entry 054-055** (2026-04-30 pre-quant rejection). Difference is vLLM version: this build is v0.19.1rc1.dev219+g72ff142c3.d20260412 (cu132+MTP). Kernel selection paths have evolved.

See `MODEL_EVALUATION_2026_05.md` for full comparison matrix; `LAB_NOTEBOOK.md` Entry 073 for switch details.

## Rejected models (2026-05-18 study)

| Model | c=1 vs Qwen3.6 best (66.7) | c=16 agg vs Qwen3.6 best (603) | Reject reason |
|-------|---------------------------|--------------------------------|---------------|
| `zai-org/GLM-4.7-Flash` | 38.5 (-42%) | 210.2 (-65%) | Slower at every concurrency, equivalent quality |
| `Qwen/Qwen3-Coder-Next-FP8` | 21.6 (-68%) | 249.8 (-59%) | Slower + 0% MTP acceptance (vllm#37554 q_scale fallback) |


## Current Config
| Field | Value |
|-------|-------|
<!-- Corrected 2026-06-15 (user-confirmed, U-7) to live `docker inspect` state: Entry 073 pre-quant switch + Entry 076 verified backend. -->
| image | vllm-cu132-test:latest (v0.19.1rc1.dev219+cu132) — adopted 2026-04-23 |
| model | Qwen/Qwen3.6-35B-A3B-FP8 (native pre-quantized FP8) — adopted 2026-05-18 Entry 073 (was Qwen/Qwen3.6-35B-A3B + on-the-fly `--quantization fp8`, 2026-04-23→05-18) |
| served_model_name | spark-llm (renamed from qwen3.5-35b on 2026-04-24, Phase 4) |
| vllm_version | v0.19.1rc1.dev219+cu132 |
| speculative_decoding | MTP=2 via `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (acceptance ~80%; verified 0 Xid / 0 restarts, Entry 076) |
| fp8 | native pre-quant FP8 (block-scaled). Do NOT add `--quantization fp8` or `--kv-cache-dtype fp8` (Entry 073) |
| kv_cache_dtype | BF16 (auto) — Entry 073 |
| moe_backend | TRITON (auto-selected); FlashInfer used for MoE kernels via `VLLM_FLASHINFER_MOE_BACKEND=latency` |
| attention_backend | FLASH_ATTN (auto-selected on SM121, verified 2026-06-11 Entry 076) — NOT FlashInfer |
| async_scheduling | Enabled |
| chunked_prefill | Enabled |
| max_num_batched_tokens | 32768 (was 4096 pre-2026-05-18; bumped on pre-quant switch, Entry 073) |
| gpu_memory_utilization | 0.70 (increased from 0.65 on 2026-04-24) |
| max_model_len | 131072 (128K, bumped from 32K on 2026-05-10 Entry 065; model native 262144) |
| kv_cache_memory | BF16 KV: 504,912 tokens @131K, max concurrency 3.85x (Entry 073). BF16 KV uses ~2× memory/token vs the prior FP8 KV (1,123,584 tokens) |
| single_request_tok_s | 66.9 (post-switch, 2026-05-18 Entry 073) — prev on-the-fly: 59.2 |
| c4_aggregate_tok_s | 198.9 — prev: 159.0 |
| c8_aggregate_tok_s | 427.7 — prev: 373.8 |
| c16_aggregate_tok_s | 678.7 — prev: 525.0 |
| startup_time | ~435s cold (fresh FP8 Triton JIT; first ~20 reqs warm to full speed) — Entry 073 |
| triton_cache | /home/claude/.cache/triton-cu132 (separate from cu130 cache) |
| stability | container started 2026-05-18T18:21Z, 0 restarts / 0 OOM / 0 Xid as of 2026-06-11 (Entry 076); host up 41 days |

## Soak Test Results (2026-05-10, Entry 066)
**Duration:** 30 minutes sustained load | **Requests:** 245 | **Success rate:** 100%

| Metric | Value | Assessment |
|--------|-------|-----------|
| Average throughput | 11.95 tok/sec | Stable, consistent |
| Peak throughput | 15.71 tok/sec | Good |
| Average latency | 43.3 sec (128K context + 512 tok gen) | Expected, predictable |
| p99 latency | 62.3 sec | Within acceptable range |
| Latency std dev | 11.2% | Low, stable (good for LLM) |
| Throughput trend | +1.0% over 30 min | Stable (not degrading) |
| Context scaling | Linear (64K:12.68, 96K:12.00, 128K:11.47 tok/s) | No cliff at 128K |
| GPU memory | 4.6–5.8% of budget | Healthy headroom |
| Speculative acceptance (MTP) | 70% average, pos0 78–82%, pos1 56–73% | Excellent robustness |
| Container uptime | 100% (28.9 min window) | Zero restarts/errors |
| Swap accumulation | Zero | Clean per-process |
| **Verdict** | **PRODUCTION READY** | ✓ cu132+MTP stable at 128K |

## Previous Config (rollback target)
| Field | Value |
|-------|-------|
| image | vllm-custom:sm121-inject |
| model | Qwen/Qwen3.5-35B-A3B |
| single_request_tok_s | 48.6 (clean, Entry 009) |
| c16_aggregate_tok_s | 311.7 |
| vllm_version | v0.17.0rc1 |

## v0.19.0 Upgrade Benchmark (2026-04-11, Entry 017)
| Config | E2E tok/s (c~4) | Server Aggregate | MoE Backend | FP8 Kernel |
|--------|-----------------|-----------------|-------------|------------|
| v0.19.0 auto-select | 29.0 | 115.4 tok/s | TRITON | CUTLASS |
| v0.19.0 forced Marlin | 30.1 | 118.8 tok/s | MARLIN | MARLIN |
| v0.19.0 pre-quant FP8 | HUNG | — | — | — |
| v0.17 sm121-inject (old) | ~30* | ~90 tok/s | MARLIN | MARLIN |

*All pre-power-cycle tests had ~3 persistent ghost requests inflating background load. Post-power-cycle clean numbers below.

## Post-Power-Cycle Clean Benchmark (2026-04-11, Entry 022)
| Concurrency | v0.19.0 tok/s | v0.17 sm121-inject | Delta |
|-------------|--------------|-------------------|-------|
| c1 | **53.5** | 48.6 | **+10%** |
| c4 aggregate | 140.4 | 133.9 | +5% |
| c8 aggregate | 216.0 | 210.4 | +3% |
| c16 aggregate | 303.1 | 311.7 | -3% |

Ghost requests: **zero** after power cycle (were 3 persistent before). Power cycle confirmed to clear stale vLLM state.

## Arena Tracking
| Field | Value |
|-------|-------|
| arena_top_fp8_qwen35_tok_s | 80.27 on vLLM (Stojanovic, recipe by eugr: DFlash n8 + flash_attn + fastsafetensors; +32% vs prior baseline 60.70, +20% vs our live 66.9). Absolute top FP8 incl. non-vLLM runtimes: 172.03 on Atlas (Szymon Walczak). Captured 2026-06-11 |
| arena_top_fp8_qwen35_entry | Qwen3.6-35B-A3B-FP8 (Stojanovic, eugr DFlash-n8 recipe, vLLM, container vllm-node-tf5) |
| arena_top_hybrid_tok_s | 108-125 synthetic, ~80 sustained (INT4+FP8 hybrid + MTP=2) — stale 2026-04-30 capture |
| arena_top_overall_tok_s | 218.85 (large-model; tiny LFM2.5-350M BF16 at 222.77 excluded as not comparable) |
| arena_top_overall_entry | Qwen3.6-35B-A3B-NVFP4 on Atlas runtime (RedHatAI checkpoint, NVFP4 KV cache, Rajendra Rawat). Prior baseline PrismaQuant 95.11 now rank 8 |
| arena_top_overall_multinode | gpt-oss-120b (MXFP4, 2-node) — 75.96 tok/s (informational only, stale 2026-04-30) |
| arena_access_method | Firestore REST — `benchmarks` collection world-readable (project `spark-arena`, public client key in JS bundle); `entries`/`leaderboard`/`recipes` App-Check-gated (403). 122 approved docs with embedded recipes, current to 2026-06-10. Unfrozen 2026-06-11 |

## Version Tracking
| Field | Value |
|-------|-------|
| vllm_last_checked_version | v0.23.0 (stable 2026-06-15) |
| vllm_latest_observed | v0.23.0 stable (2026-06-15); eugr tracking **0.23.1rc1.dev207** (2026-06-20, +78 commits vs dev129; no new commits/wheel June 21). v0.23.0 highlights: DeepSeek-V4 hardening, Model Runner V2 default for Llama/Mistral dense, Gemma 4 Unified encoder-free, multi-tier KV cache offloading, "breakable CUDA graphs," prefix-cache corruption remedies, Rust frontend expansion. **FP4 GEMM for SM120/121 landed in v0.22.0** (FlashInfer b12x MoE + CUTLASS-FP4 MoE backend). **PR #45277 (CUDA arch build coverage cleanup) MERGED 2026-06-14 (in v0.23.0):** removed false-positive SM12x CUTLASS FP8 MoE support gate (CUTLASS grouped GEMM never had SM12x kernel; FlashInfer b12x MoE is the correct SM12x path); added SM12x to `cuda_archs_sm90plus()` + DSV3 router paths. Relevant context for Arm C eval — CUTLASS MoE dispatch path more explicit in v0.23.0. No explicit SM121/GB10 text in v0.23.0 release notes; cuDNN graph-corruption fix unconfirmed. DeepGEMM SM12x still blocked (#41063 open). Gemma4 PRs #39138 AND #40099 BOTH still open. **Target for Arm C eval: eugr dev207 wheel (June 20).** |
| qwen_current_model | Qwen/Qwen3.6-35B-A3B (adopted 2026-04-23) |
| gemma4_pr_status | #39138 OPEN (needs-rebase; last activity 2026-06-16 automated project assignment only — no code progress; re-confirmed 2026-06-21) / #40099 OPEN (awaiting code-owner review, last substantial activity 2026-04-22; re-confirmed 2026-06-21). Both still required; unmerged in v0.23.0. |

## spark-vllm-docker Tracking
| Field | Value |
|-------|-------|
| svd_last_checked_date | 2026-06-21. No new commits or wheel since June 20; latest remains **0.23.1rc1.dev207+gdced29076.d20260620** (June 20 wheel build), FlashInfer **0.6.13-9c5ed7c1-d20260618**. Last repo commit: June 19 — "Rolled back DeepSeek PR inclusion" (PR #41834 SM12x Triton fallbacks added and reverted same day — still unstable). Prior changes still apply: gpu_memory_utilization default 0.7→0.8; CUDA base `nvidia/cuda:13.0.2-devel-ubuntu24.04`; MiniMax IPC fused path disabled (PR #43410). **PR #279** (DFlash + FlashInfer FP8 KV Cache) still OPEN. Do not combine instanttensor with DFlash (issue #211). |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-06-21 (WebSearch fallback; 719.json returns 403 in remote execution env) |
| forum_endpoints | **Category 720 PERMANENTLY REMOVED** (404 re-confirmed; /c/721/show.json shows 719 parent + 721 only child; former projects topics merged in). Scan 719.json only — it aggregates everything; 721 is a near-duplicate. **Category 722 (`announcements`) confirmed** — /t/371965 and /t/372623 are under this sub-category. **Category 723 (`dgx-spark-gb10-projects`)** unverified; may be projects sub-category replacement for removed 720. Endpoint list fix needed in spark-recon skill source. |
| forum_posts_since_084 | New: **/t/372748 "Optimizing DGX for Openclaw Brain"** — community user post about running NemoClaw/OpenClaw agentic stack on single DGX Spark; no perf/driver/firmware relevance. No new driver/firmware/crash/OOM reports since 2026-06-20. |
| forum_posts_since_083 | **RTX Spark consumer notebook line** announced at Computex 2026 (/t/371812 "Next version of DGX Spark is here: It is a notebook") — starting $2,899, Fall 2026, consumer-grade product distinct from DGX Spark professional workstation; NOT a GB10 replacement. No new driver/firmware/crash/OOM reports since 2026-06-19. |
| forum_posts_since_082 | Driver **595.45.04** + **CUDA 13.2** confirmed in cuda-compute-repo (community-flagged beta — not for production DGX OS). `gpu_memory_utilization >0.80` causes system hangs on DGX Spark (/t/366060 — corroborates eugr's 0.80 recipe cap). No new driver/firmware/crash/OOM reports since 2026-06-18. |
| forum_posts_since_081 | **DGX Spark Software Updates — June 2026 Release** (/t/371965): NVIDIA co-developed NVFP4 checkpoint for Qwen3.6-35B + MTP; claims 2.6× vs FP8 (baseline = FP8 without MTP; vs our FP8+MTP2 the real delta is +33% c1, −21% c8). Multi-node Cluster Assistant (2–4 nodes) now in NVIDIA Sync. **"What is actually new in the June Software Release?"** (/t/372623): community scrutiny of claims. No new driver/firmware/crash/OOM reports since 2026-06-16. 14W throttle bug still unfixed (from prior entries). |
| forum_posts_since_079 | New NVIDIA announcement: "DGX Spark Software Updates - June 2026 Release" (/t/371965) — content inaccessible (403 in this run); likely documents the June software update. Atlas thread active (page 7+, 403). No new driver/firmware/crash findings accessible via WebSearch for period 2026-06-11→2026-06-16. |
| forum_posts_since_074 | ~110 new topics 2026-05-27→2026-06-11. Headlines: **vLLM #37754 FlashInfer+MTP crash on SM121 (MTP=2 ≈ 9h MTBF — our exact prod config, /t/366822)**; Copyfail-patched kernel shipped (6.17.0-1018+, current line 6.17.0-1021 + driver 580.159.03 with better GB10 OOM handling, /t/373018); AutoRound W4A16 INT4 viable via Marlin gs=128 ("2x FP8" claim unverified, /t/372466); self-built vLLM v0.22.0 works on GB10 (/t/371853; ~12h cuDNN bug fixed in v0.23.0); Qwen3.6-NVFP4 249-268 agg tok/s on nightly (/t/371810); Atlas 75.6-93 tok/s c1 independent measure with failure modes (/t/369263); Gemma4-31B official QAT W4A16 from Google (/t/372444); 14W throttle bug still unfixed. |

## Gemma 4 Reference Numbers (2026-04-11, Entries 020-021; updated 2026-04-30, Entry 061)
| Model | Quant | c1 tok/s | c8 agg | c16 agg | Notes |
|-------|-------|---------|--------|---------|-------|
| 26B-A4B (MoE) | FP8 | 38.9 | 257.6 | 387.5 | Our April 11 benchmark. Guided JSON blocked (#39130, #40080). |
| 26B-A4B (MoE) | BF16 | 23.6 | 158.7 | 206.7 | Day-1 floor. TRITON_ATTN forced (hetero heads). |
| 31B (Dense) | NVFP4 | 6.8 | 54.0 | — | Bandwidth-bound. Not viable for interactive on single-node. |
| 31B (Dense) | BF16 | 3.7 | 28.2 | — | Bandwidth-bound. Matches community exactly. |

## Gemma 4 Community Numbers (as of 2026-04-30, Entry 061)
| Model | Quant | c1 tok/s | c4 agg | Source | Notes |
|-------|-------|---------|--------|--------|-------|
| 26B-A4B NVFP4 (bg-digitalservices) | W4A4 modelopt | 52 | ~114 | ai-muninn.com Apr 13 | 16.5 GB; `VLLM_NVFP4_GEMM_BACKEND=marlin`; 97.6% quality vs BF16 |
| 26B-A4B FP8 on-the-fly (eugr recipe) | FP8 | ~45-50 | ~140 | forum Apr 3-5 | eugr v0.20.1rc1; InstantTensor breaks init — use safetensors mode |
| **Structured output status** | — | — | — | PRs #39138 + #40099 | **NOT fixed. Both PRs open, unmerged as of 2026-04-30.** |

## Recon Triggers
<!-- spark-recon reads these to prioritize findings. Format: source | pattern (keywords, OR/AND logic) | action level + what to do | date added -->
<!-- Remove rows when resolved. Add rows after each experiment/research session. -->

| Source | Pattern | Action | Added |
|--------|---------|--------|-------|
| vllm_release | gemma4 AND (guided OR grammar OR xgrammar) | ACTION: confirm PRs #39138 AND #40099 merged — both required before Gemma 4 experiment (Entry 061) | 2026-04-11 |
| vllm_release | DeepGEMM AND (SM12 OR SM121 OR Blackwell OR GB10) | ACTION: benchmark Qwen3.5 FP8 with DeepGEMM | 2026-04-11 |
| vllm_release | FlashInfer AND (heterogeneous OR mixed head) | INFO: could boost Gemma 4 single-request past 50 tok/s | 2026-04-11 |
| vllm_release | MXFP4 AND (online OR on-the-fly OR Qwen) | INFO: test MXFP4 quantization path on Qwen3.5 | 2026-04-11 |
| vllm_release | speculative AND (Qwen OR MoE) | INFO: test spec decode with Qwen3-0.6B draft model | 2026-04-11 |
| arena | fp8 AND {model_family} AND single-node > baseline_tok_s * 1.10 | ACTION: investigate config difference vs current baseline | 2026-04-11 |
| huggingface | Qwen3.7 (27B OR 35B) OR Qwen3.6-Plus OR Qwen4 model weights | ACTION: benchmark day — full throughput + quality suite. Qwen3.7 27B/35B announced forthcoming; release window open mid-June — check weekly. Ignore `RscriptSQwen` squat. | 2026-04-11 (upd 2026-06-11) |
| forum | gemma4 AND (guided JSON OR grammar OR structured output) fix | ACTION: verify both #39138 (xgrammar bypass) and #40099 (repetition loop) merged before scheduling experiment | 2026-04-11 |
| vllm_release | #37754 OR (FlashInfer AND MTP AND (crash OR Xid)) | ACTION: production MTP=2 stability — if upstream fix lands, note version and re-test; until then monitor dmesg Xid 13 | 2026-06-11 |

## Watch Items
- **[VERIFIED CLEAN 2026-06-11 — Entry 076]** **vLLM #37754: FlashInfer + MTP crashes on SM121 (MTP=2 ≈ 9-hour MTBF, Xid 13).** Checked: zero Xid events in kernel logs back to ~2026-03-06, zero qwen35 restarts/OOM since 2026-05-18, 41-day host uptime. **Key correction: production attention backend is FLASH_ATTN (auto-selected), not FlashInfer** — #37754's FlashInfer-attention path does not directly apply; residual exposure is FlashInfer MoE only (`VLLM_FLASHINFER_MOE_BACKEND=latency`), 3+ weeks clean. MTP=2 retained. Xid alerting still planned (no visibility on the box today). Trigger row stays armed for upstream fix tracking. (Current Config row `attention_backend | FLASHINFER` is stale — user to correct.)
- **[ACTION 2026-06-11]** **DFlash eval elevated to top eval candidate (perf + stability).** eugr Arena recipe hits 80.27 tok/s c1 (+20% vs our live 66.9), same model/image family: DFlash n8 (`z-lab/Qwen3.6-35B-A3B-DFlash`) + `--attention-backend flash_attn` + `--load-format fastsafetensors`, gpu_mem_util 0.85, prefix caching ON. Resolve prefix-caching discrepancy during eval (Arena recipe ON vs svd recipe removed-for-accuracy 2026-05-14). Also mitigates the #37754 MTP risk. Sandbox only — do NOT touch production qwen35.
- **[DONE 2026-06-15 — Entry 078]** **Kernel/driver maintenance:** applied 6.17.0-1014→**6.17.0-1021** + driver 580.142→**580.159.03** (Copyfail/Dirtyfrag CVE patched; GB10 OOM-handling improvements). Reboot CLEAN, GPU functional under SecureBoot, all containers healthy, no perf regression (c1 65.4 vs 66.9). **Gotcha discovered & resolved:** dist-upgrade flipped nvidia to DKMS signed with an unenrolled MOK key → would have killed the GPU; recovered by restoring prebuilt Canonical-signed modules (see CLAUDE.md dist-upgrade rule). No firmware was pending (fwupdmgr clean), so the firmware-HOLD question was moot. 590 still unsupported.
- **[ESCALATED 2026-06-18 UPDATE — NVFP4 added as Arm D]** **vLLM 0.23.x upgrade eval: all gating conditions met; now scope includes NVFP4.** v0.23.0 released 2026-06-15; eugr at v0.23.1rc1.dev129 (June 17, +76 commits). v0.23.0 adds FP4 GEMM for SM120/121 (FlashInfer b12x MoE + CUTLASS-FP4) — the enabling kernel for NVFP4 on single-node DGX Spark. **Arm C:** eugr 0.23.x build + current FP8 model (pure build upgrade). **Arm D (NEW):** NVFP4 (`RedHatAI/Qwen3.6-35B-A3B-NVFP4` or NVIDIA-co-developed checkpoint) + MTP-3; expected +33% c1, −21% c8 vs current FP8+MTP2. DFlash+FP8-KV (PR #279) still OPEN — add as Arm B3 once merged. No explicit SM121/cuDNN graph-corruption text in v0.23.0 release notes; verify via soak test. Migration notes: `--speculative-config` JSON, gpu_mem_util→0.80, validate UMA. Caution: sparkrun issue #164 (FlashInfer in eugr :latest). Sandbox only.
- **[UPDATED 2026-06-18]** **Qwen3.7 27B/35B open weights NOT released — check Qwen HF org WEEKLY through mid-July.** The predicted June window (based on 3.6 API→weights lag) passed without release. Qwen3.7-Max (May 19, API-only) and 3.7-Plus (June 1, API-only) both remain closed. No HF repo under official Qwen org as of June 18. If no release by 2026-07-16, reconsider whether Qwen is shifting to closed-weight-first for new generations. A Qwen3.7-35B-A3B-FP8 would still be a near-drop-in successor when released. Beware HF squat `RscriptSQwen/Qwen3.7-plus` (fake, 2026-06-04).
- **[NEW 2026-06-11]** **AutoRound W4A16 INT4 viable on SM121** — narrows Entry 068 "no viable INT4 path" (that rejection was AWQ/compressed-tensors gs=32-specific): whpthomas claims "2x FP8, similar quality" (unverified WIP, /t/372466; Intel AutoRound, Marlin backend, gs=128); Qwen3.5-122B AutoRound Arena recipe scores 92/100 (/t/370834); jwarner: Marlin W4A16 = fastest 4-bit path on GB10, W4A4 nonexistent (/t/372559). Experiment candidate after DFlash eval.
- **[UPDATED 2026-06-18]** **NVFP4 is now first-class on GB10 AND officially NVIDIA-backed.** NVIDIA June 2026 DGX Spark update (/t/371965) ships co-developed NVFP4 checkpoint for Qwen3.6-35B + MTP. Community numbers on vLLM v0.23.x (technigmaai recipe, FlashInfer attn, CUTLASS-FP4 MoE, MTP-3): **97 tok/s c1** (+32.7% vs our FP8+MTP2 73.1), **322 tok/s c8 agg** (−20.9% vs our 406.9). v0.23.0 FP4 GEMM for SM120/121 is the enabling kernel. Shape: same c1-win/c8-loss tradeoff as DFlash. Arena overall top = Qwen3.6-NVFP4 on Atlas (~120 tok/s c1). **Add as Arm D alongside Arm C (vLLM 0.23.x upgrade) eval.** Gate criterion: ≥+5% c8 (unlikely to pass) OR documented pure-c1 workload shift. NVIDIA's 2.6× claim uses FP8-without-MTP as baseline, not our current config.
- **[NEW 2026-06-11]** **Arena tracking UNFROZEN:** Firestore `benchmarks` collection world-readable via REST (project `spark-arena`, public client key in JS bundle); `entries`/`leaderboard`/`recipes` still App-Check-gated (403). Use this path for future Arena checks (122 approved docs with embedded recipes).
- **[NEW 2026-06-11]** **#37554 watch was miscalibrated:** issue closed-as-completed 2026-03-20 — the closure IS the q_scale=1.0 fallback that causes Coder-Next 0% MTP acceptance. Real watch: a proper KV-scale calibration fix for hybrid GDN+attention (none in v0.22.x) + whether #39949 changes Coder-Next MTP behavior. CLAUDE.md bullet corrected 2026-06-11.
- **[NEW 2026-06-11]** Spark-recon Check 5 endpoint fix needed in personal-plugin skill source: drop category 720 (permanently removed; 719.json sufficient).
- **[SUPERSEDED 2026-06-11 — eugr now 0.22.1rc1.dev330; see vLLM 0.22.x eval item above]** **eugr base jumped to vLLM 0.21.1rc1.dev292+cu132 + FlashInfer 0.6.12** (rebuilt 2026-05-26, ~2 minor versions ahead of our prod v0.19.1rc1.dev219). Reinforces the pending eugr eval. **ACTION:** bench `dgx-vllm-eugr-nightly:latest` vs current `vllm-cu132-test:latest`; keep current unless ≥+5% c8 AND quality holds. Sandbox only — do NOT touch production qwen35. Pre-quant FP8 verdict already flipped once across builds (Entry 054 vs 070), so kernel-selection paths may have shifted again.
- **[SUPERSEDED 2026-06-11 by elevated DFlash eval item above]** **DFlash speculative decoding recipe matured (eugr).** `qwen3.6-35b-a3b-fp8-dflash.yaml` uses draft `z-lab/Qwen3.6-35B-A3B-DFlash`, num_speculative_tokens=15, flash_attn backend. **Caveat:** prefix caching removed 2026-05-14 for accuracy. Candidate to eval vs our MTP=2 when a slot opens.
- **[SUPERSEDED 2026-06-11 by Qwen3.7 27B/35B weekly watch above]** **Qwen3.7-Max announced 2026-05-20** (Alibaba Cloud Summit) — flagship, 1M ctx, native extended-thinking, **API-only on DashScope, no open weights.** Monitor for open-weights release (next ACTION trigger candidate, alongside Qwen4/Qwen3.6-Plus).
- **[NEW 2026-05-27]** **Poolside Laguna XS.2** — 33B MoE / 3B active, NVFP4+INT4, 256K ctx; jwarner: "similar to Qwen3.6-35B-A3B, less verbose," may beat Gemma4-26B-A4B. No Spark tok/s yet. Watch for benchmarks; potential future eval comparator. **[UPDATE 2026-06-11]** Official `poolside/Laguna-XS.2-FP8` + `Laguna-XS.2-speculator.dflash` confirmed on HF (Apache 2.0, SWE-bench Verified 68.2%) — most credible coding-specialist alternative with a sound SM121 path.
- **[SUPERSEDED 2026-06-11 by NVFP4 first-class item above]** **NVFP4 reportedly working on single Spark via marlin GEMM** — two independent sources: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (~55.9 tok/s c1, MTP 83-93%, GB10-validated) and Nemotron-3-Super-120B-A12B-NVFP4 (23.45 tok/s, `VLLM_NVFP4_GEMM_BACKEND=marlin`, /t/370070). **Partially contradicts standing "NVFP4 broken on SM121" fact.** Both below FP8 prod throughput, so low priority — value is the enablement recipe, not perf. Verify next deep-dive.
- **[NEW 2026-05-27]** **vLLM v0.21.0 breaking changes for any future custom rebuild:** C++20 build requirement (#40380), Transformers v4→v5 default (#40389). Our GLM-4.7 eval already needed transformers 5.0.0, so v5 default is aligned. Qwen3.5/3.6 Gated DeltaNet attention (#41025) is an upstream arch-path change to validate if/when we move off the bespoke cu132 build.
- **[RESOLVED 2026-06-11 — confirmed permanent; 719.json sufficient]** **Forum Category 720 (gb10-projects) returns HTTP 404** — merged/removed; former projects topics now under 719/721.
- **[CONFIRMED 2026-05-27]** **FP8 is the only sound quant path on SM121.** Forum: INT8 AWQ (W8A16) completely broken (illegal mem access in conch-triton, /t/371315), AWQ INT4 1.8-4.9 tok/s (/t/371529); jwarner: "FP8 essentially replaced INT8." Validates pre-quant FP8 production choice; consistent with Entry 068 Marlin-WNA16 hang.
- **[SUPERSEDED 2026-06-11]** **eugr v0.20.2rc1.dev299+cu132** (up from dev173, +126 upstream commits, rebuilt 2026-05-13). eugr now full-time NVIDIA Spark Team staff. Official Qwen3.6-35B-A3B-FP8 recipe + dflash variant. New: InstantTensor loader for 397B recipes. No 35B recipe changes since 2026-05-08. **ACTION:** Bench `dgx-vllm-eugr-nightly:latest` + recipe vs current `vllm-cu132-test:latest` (v0.19.1rc1.dev219). Decision criterion: keep current unless ≥+5% c8 AND quality holds.
- **[NEW 2026-05-09]** **Atlas inference engine** (Avarok Cybersecurity, `avarok/atlas-gb10:latest`, AGPLv3) — Rust+CUDA, no PyTorch, MTP K=2. Claims **121–140 tok/s c1** on Qwen3.6-35B-A3B-FP8 (~130 sustained). Vendor-published, not yet leaderboard-verified. **ACTION:** Sandboxed eval (do NOT touch production qwen35); license check first. **[UPDATE 2026-06-11]** Now corroborated by two independent sources: Arena top-5 domination (172.03 FP8 / 218.85 NVFP4 tg128 c1) + forum measure 75.6-93 tok/s c1 (azampatti, /t/369263). Known failure modes: long-context + c≥2 slowdown (-35-40%), tool-call corruption; fixes merged 2026-06-02. Real but immature — still sandbox-only + AGPLv3 review.
- **[REJECTED 2026-05-13]** **AWQ INT4 `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (Entry 068):** -10.6% c1, -28.4% c8, -22.1% c16 vs FP8 production. CUDA graph capture hangs with Marlin WNA16 MoE on SM121 (forces enforce-eager). BF16 KV cache negates weight savings. FP8 on-the-fly remains optimal.
- **[NEW 2026-05-09]** `z-lab/Qwen3.6-35B-A3B-DFlash` drafter — 0.5B BF16 block-diffusion, vLLM-compatible via `--speculative-config '{"method":"dflash",...}'`. Different mechanism than MTP; claims up to 2.9× on B200. Future experiment slot.
- **[NEW 2026-05-09]** **SM121 cutlass-dsl PTX gotcha:** `nvidia-cutlass-dsl 4.5.x` emits invalid PTX for GB10 sm_121 `_mma`. eugr pins 4.4.2. Document for any future custom kernel build.
- **[NEW 2026-05-09]** **HOLD on UEFI firmware advancement** beyond 2026-04-30. Multiple users (holger.pandel /t/369572) report UEFI install failures. **[UPDATE 2026-06-11]** No new failure wave since; HOLD liftable via the June-release full `fwupdmgr` sequence (see kernel/driver maintenance item above).
- **[NEW 2026-05-09]** New A3B-class comparators worth a future quality+speed bench: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`, MiMo-V2.5.
- **[NEW 2026-05-09]** New NVFP4 Qwen3.6-35B-A3B variants: `unsloth/Qwen3.6-35B-A3B-NVFP4` (3 days, 17.2k dl), `Ex0bit/Qwen3.6-35B-A3B-PRISM-NVFP4` (6 days, 75.5k dl). Alternatives to RedHatAI variant.
- **[UPDATE 2026-05-13]** v0.20.x prefix-caching + spec-decode regression confirmed multi-source (arena search + forum). Reinforces v0.20.x HOLD.
- **[RESOLVED 2026-06-11 — Firestore `benchmarks` REST path found; tracking unfrozen]** Spark Arena leaderboard table is JS-rendered behind a Firestore App Check ACL — anonymous reads denied. Numeric `arena_top_*` values re-captured 2026-06-11; see `arena_access_method`.
- **[RESOLVED 2026-05-09]** qwen35 + bge-m3 swap pressure cleared via planned restart (Entry 064). qwen35 EC: 2.16 GiB → 264 MB (-88%); bge-m3 EC: 1.54 GiB → 0 kB. Restart pattern repeatable for future runs.
- **[NEW 2026-05-09]** **Entry 052 post-firmware baseline (65.9/174.7/394.3/634.0 c1/c4/c8/c16) is suspect.** Post-rollback re-bench (Entry 064, no functional changes) measured 58.3/161.5/374.1/552.7 — within ±3% of pre-firmware Entry 050 baseline (59.9/166.2/373.8/564.0). Schedule controlled re-bench to determine whether Entry 052 "+10%" firmware gain is reproducible. If not, downgrade `single_request_tok_s` to ~60 tok/s.
- **[NEW 2026-05-09]** **Prefix caching re-test deferred.** First attempt rolled back (Entry 064): synthetic bench prompts (~15-20 tokens) shorter than 16-token cache block boundary → 0 hits across 3151 queries; -7-9% throughput from pure overhead. Future test must use realistic ≥200-token shared system prompt + multiple varying user messages.
- **[NEW 2026-05-09]** qwen3-embed EngineCore (1.26 GiB swap) and gliner (1.74 GiB swap) NOT addressed in Entry 064. Defer to next maintenance cycle if they grow.
- **[RESOLVED 2026-04-24]** MTP=2 on Qwen3.6 — ablation benchmark (Entry 043-044): MTP degrades c4 by 14.9% but improves c8 by 24.7% and c16 by 19.6%. c1 tied (~51 tok/s). Decision: KEEP MTP. Primary workload is pipeline at c8-c16 where MTP wins.
- **[RESOLVED 2026-04-30]** Firmware update applied (Entry 050). Post-firmware benchmark (Entry 052): c1 +10.0%, c4 +5.1%, c8 +5.5%, c16 +12.4%. c16 634.0 tok/s is new project record.
- **[SUPERSEDED 2026-05-13]** eugr v0.20.1rc1.dev96 → now at v0.20.2rc1.dev299. See updated entry above.
- **[ACTION 2026-04-30]** Pre-quant FP8 hang rule INVALIDATED for Qwen3.6: Seth Hobson's Arena entry + community forum reports + model availability all confirm Qwen3.6-35B-A3B-FP8 works on v0.20.0. Re-test on our cu132 image.
- **[NEW 2026-04-30]** vLLM-Tune (serapis): kernel tuning CLI for Triton FP8/MoE. +58% prefill, +9.5% decode on Qwen3.6-35B-A3B-FP8. Test compatibility with cu132+MTP config.
- **[NEW 2026-04-30]** FlashQLA: linear attention kernels, 2x speedup claimed vs FlashInfer. SM90+ but jwarner says GB10 works. No vLLM integration yet. Immature — monitor.
- **[NEW 2026-04-30]** DFlash speculative decoding: 91-97 tok/s with NVFP4, 80+ with AWQ. Alternative to MTP. Not in mainline vLLM. joshua.dale.warner leading.
- **[CLOSED 2026-05-13]** NVFP4/INT4 quantization path (Entry 060→068). AWQ INT4 tested and REJECTED (Entry 068): -10 to -28% throughput, CUDA graph hang with Marlin on SM121. NVFP4 broken (no SM121 hw instruction). PrismaQuant/DFlash both require unmerged code. **No viable INT4 path exists for SM121.** FP8 on-the-fly confirmed optimal. Close investigation.
- **[SCOPED 2026-04-30]** PrismaQuant: now #1 Arena at 95.11 tok/s (Sean Williams, INT4 4.75-bit + DFlash). Quality 88/100 vs FP8 91/100 (3.3% gap on opaque internal score). Run after AWQ INT4 experiment if AWQ shows >10% c1 gain and quality holds.
- **[UPDATE 2026-05-13]** vLLM v0.20.2 (2026-05-10): LOW bugfix on v0.20.0. DeepGEMM SM12x blocked (#41063, no timeline). Gemma4 PRs still open. MTP IMA fix in v0.20.0 could help acceptance rate. HOLD remains — MoE refactor regression risk + prefix caching regression.
- **[NEW 2026-05-13]** **antirez/ds4 custom SM121 CUDA engine** (/t/369791): purpose-built for DS4 Flash on single Spark, 29.2 tok/s tg128, 95% of memory bandwidth ceiling (~215-227 GB/s of 273 GB/s peak). Custom sm_121 kernels achieving near-roofline. MTP draft support in progress. Signals untapped hardware potential beyond vLLM.
- **[RESOLVED 2026-06-11 — patched kernel shipped; update planned, see kernel/driver maintenance item]** **CVE-2026-31431 (Copyfail) + Dirtyfrag LPE** (/t/369489): `linux-image-6.17.0-1018-nvidia 6.17.0-1018.18` in noble updates since 2026-05-15; current line 6.17.0-1021.
- GPU power-draw throttle bug: after crash/sleep, GPU enters 14W/513 MHz cap → throughput halves. Fix: wall power cycle (unplug 1 min), not reboot. Systemic across OEM variants.
- Sparkview (github.com/parallelArchitect/sparkview): GB10-aware GPU monitor with PSI pressure, clock state, unified memory handling. Consider installing.
- GB10 bandwidth baseline (parallelArchitect, Apr 25): GPU read bandwidth drops 44% under inference load (161→90 GB/s). Confirms memory bandwidth as primary bottleneck, not coherence.
- Qwen3.6-27B: dense 27B, Gated DeltaNet hybrid. Bandwidth-limited ~7.8 tok/s on GB10. Not primary model candidate.
- Tool Eval Bench CLI (SerraphimSerapis): Qwen3.6 scores 100/100 on ToolCall-15.
- Thermal shutdown issue systemic across DGX Spark OEMs. Throttle GPU clocks to 2100-2400 MHz if experiencing shutdowns.
- Qwen3.6-Plus: still API-only (May 13), no HF weights. Monitor for open-weight release.
- Qwen4 monitor: no announcement as of 2026-05-13, prediction markets suggest before July 2026
- Hybrid INT4+FP8 checkpoint achieves 108-125 tok/s synthetic, ~80 sustained. Requires custom checkpoint build.
- DanTup/spark-evals GitHub repo — systematic Inspect AI quality evals across quant formats.
- **[SCOPED 2026-04-30]** Gemma 4 status researched (Entry 061). Community 26B NVFP4: 52 tok/s c1 (up from 38.9); still 21% below Qwen3.6. Structured output NOT fixed — PRs #39138 (xgrammar bypass) and #40099 (repetition loops) both unmerged. eugr "recipe fixes" were Python env fixes, not throughput/correctness fixes. InstantTensor broke Gemma 4 init; safetensors workaround degrades perf 75%. **Decision: DO NOT schedule experiment until both PRs merge.** bg-digitalservices NVFP4 checkpoint (16.5 GB, 97.6% quality) is the best 26B option when experiment is viable.
- InstantTensor in eugr/spark-vllm-docker — operator fusion library. Confirmed broke Gemma 4 compatibility (vincenzoa, Apr 29-30). Workaround: safetensors mode but 75% perf regression.
- v0.20.0 stability: one tester reverted to v0.19.2 on Qwen3.6-27B (forum, Apr 30). Not yet validated for Qwen3.6-35B-A3B.
- **[RESOLVED 2026-04-24]** eugr 0.19.2rc1 — REJECTED (Entry 045-046). Superseded by v0.20.1rc1 evaluation.
- **[RESOLVED 2026-04-24]** Tool calling parser confirmed correct. `qwen3_coder` parses Qwen3.6 XML format.
- **[RESOLVED]** vLLM v0.19.1 released 2026-04-18. TurboQuant landed in v0.20.0.
