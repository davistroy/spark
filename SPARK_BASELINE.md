# Spark Performance Baseline

Last updated: 2026-05-10
Last recon: 2026-05-09 (Entry 063)
Last audit: 2026-05-09 (Entry 062)
Last benchmark: 2026-05-10 (Entry 066 — 30-min soak test, 128K context stability)

## Current Config
| Field | Value |
|-------|-------|
| image | vllm-cu132-test:latest (v0.19.1rc1.dev219+cu132) — adopted 2026-04-23 |
| model | Qwen/Qwen3.6-35B-A3B (on-the-fly FP8) — adopted 2026-04-23, snapshot 53c43178507d69762986fbfa314f6e8d4d859409 |
| served_model_name | spark-llm (renamed from qwen3.5-35b on 2026-04-24, Phase 4) |
| vllm_version | v0.19.1rc1.dev219+cu132 |
| speculative_decoding | MTP=2 (method: mtp, num_speculative_tokens: 2, acceptance rate 80.7%) |
| mtp_drafter | Qwen3_5MoeMTP, 34.16 GiB total model load |
| moe_backend | TRITON (auto-selected) |
| fp8_kernel | CutlassFP8ScaledMMLinearKernel |
| attention_backend | FLASHINFER |
| async_scheduling | Enabled |
| chunked_prefill | Enabled |
| gpu_memory_utilization | 0.70 (increased from 0.65 on 2026-04-24) |
| max_model_len | 131072 (128K, bumped from 32K on 2026-05-10 Entry 065; model native 262144) |
| kv_cache_memory | 47.18 GiB (1,123,584 tokens, max concurrency 29.76x at 131K full context) |
| single_request_tok_s | 65.9 (post-firmware, 2026-04-30 Entry 052) — prev: 59.9 (2026-04-24) |
| c4_aggregate_tok_s | 174.7 — prev: 166.2 |
| c8_aggregate_tok_s | 394.3 — prev: 373.8 |
| c16_aggregate_tok_s | 634.0 — prev: 564.0 |
| firmware_gain | c1 +10.0%, c4 +5.1%, c8 +5.5%, c16 +12.4% (all levels improved) |
| startup_time | ~364s (warm Triton cache, cu132-cu132 dir) |
| triton_cache | /home/claude/.cache/triton-cu132 (separate from cu130 cache) |

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
| arena_top_fp8_qwen35_tok_s | 60.70 (Seth Hobson / traderaegis, v0.20.0, pre-quant FP8, MTP=1) |
| arena_top_fp8_qwen35_entry | Qwen3.6-35B-A3B-FP8 (Seth Hobson / traderaegis) |
| arena_top_hybrid_tok_s | 108-125 synthetic, ~80 sustained (INT4+FP8 hybrid + MTP=2) |
| arena_top_overall_tok_s | 95.11 |
| arena_top_overall_entry | Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm (INT4, Sean Williams, DFlash spec decode) |
| arena_top_overall_multinode | gpt-oss-120b (MXFP4, 2-node) — 75.96 tok/s (informational only) |

## Version Tracking
| Field | Value |
|-------|-------|
| vllm_last_checked_version | v0.20.1 (stable 2026-05-04) |
| vllm_latest_observed | v0.20.1 (2026-05-04, MEDIUM — DeepSeek V4 patch; FlashInfer one-sided BF16/MXFP8, PTX FP32→FP4, multi-stream GEMM; #41199 reasoning-parser kwargs to structured output; CUDA-graph + num_gpu_blocks_override fixes. **No SM121/GB10/MoE/spec-decode items.** HOLD remains.) |
| qwen_current_model | Qwen/Qwen3.6-35B-A3B (adopted 2026-04-23) |
| gemma4_pr_status | #39138 OPEN (active 2026-05-08) / #40099 OPEN (stale, last update 2026-04-22). Both still required. |

## spark-vllm-docker Tracking
| Field | Value |
|-------|-------|
| svd_last_checked_date | 2026-05-09 (eugr/spark-vllm-docker — v0.20.2rc1.dev173+cu132, FlashInfer 0.6.11, official `qwen3.6-35b-a3b-fp8.yaml` + `qwen3.6-35b-a3b-fp8-dflash.yaml` recipes added 2026-05-06, dedicated chat-template fix mod, two perf-regression Dockerfile fixes 2026-05-08, SM121 cutlass-dsl 4.4.2 pin in b12x mod) |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-05-09 |
| forum_posts_since_063 | ~31 active topics since 2026-04-30 (Atlas engine claim 100-130 tok/s Qwen3.6-FP8, eugr joins NVIDIA Spark Team 2026-05-04, UEFI firmware install failures, MiMo-V2.5, Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8, DeepSeek V4 Flash MXFP4 PoL on single GB10, Albond Qwen3.5-122B-A10B 51 tok/s on single Spark) |

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
| huggingface | Qwen3.6-Plus OR Qwen4 model weights | ACTION: benchmark day — full throughput + quality suite | 2026-04-11 |
| forum | gemma4 AND (guided JSON OR grammar OR structured output) fix | ACTION: verify both #39138 (xgrammar bypass) and #40099 (repetition loop) merged before scheduling experiment | 2026-04-11 |

## Watch Items
- **[NEW 2026-05-09]** **eugr v0.20.2rc1.dev173+cu132 + official Qwen3.6-35B-A3B-FP8 recipe** (Entry 063 cross-correlated 3 sources). eugr is now NVIDIA Spark Team staff (2026-05-04). Recipe ships dedicated chat-template fix and a `dflash` variant. Spark Arena's official MTP recipe pins this exact image. **ACTION:** Bench `dgx-vllm-eugr-nightly:latest` + recipe vs current `vllm-cu132-test:latest` (v0.19.1rc1.dev219). Decision criterion: keep current unless ≥+5% c8 AND quality holds.
- **[NEW 2026-05-09]** **Atlas inference engine** (Avarok Cybersecurity, `avarok/atlas-gb10:latest`, AGPLv3) — Rust+CUDA, no PyTorch, MTP K=2. Claims **121–140 tok/s c1** on Qwen3.6-35B-A3B-FP8 (~130 sustained). Vendor-published, not yet leaderboard-verified. **ACTION:** Sandboxed eval (do NOT touch production qwen35); license check first.
- **[NEW 2026-05-09]** **AWQ INT4 candidate for Entry 060 minimum-viable experiment:** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` — Apache 2.0, 438k dl/month, ~9–10 GB, vLLM ≥0.19.0, supports `--reasoning-parser qwen3` + `--tool-call-parser qwen3_coder`. Works on our current cu132 image. **ACTION:** Run 45-min test next maintenance window.
- **[NEW 2026-05-09]** `z-lab/Qwen3.6-35B-A3B-DFlash` drafter — 0.5B BF16 block-diffusion, vLLM-compatible via `--speculative-config '{"method":"dflash",...}'`. Different mechanism than MTP; claims up to 2.9× on B200. Future experiment slot.
- **[NEW 2026-05-09]** **SM121 cutlass-dsl PTX gotcha:** `nvidia-cutlass-dsl 4.5.x` emits invalid PTX for GB10 sm_121 `_mma`. eugr pins 4.4.2. Document for any future custom kernel build.
- **[NEW 2026-05-09]** **HOLD on UEFI firmware advancement** beyond 2026-04-30. Multiple users (holger.pandel /t/369572) report UEFI install failures.
- **[NEW 2026-05-09]** New A3B-class comparators worth a future quality+speed bench: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`, MiMo-V2.5.
- **[NEW 2026-05-09]** New NVFP4 Qwen3.6-35B-A3B variants: `unsloth/Qwen3.6-35B-A3B-NVFP4` (3 days, 17.2k dl), `Ex0bit/Qwen3.6-35B-A3B-PRISM-NVFP4` (6 days, 75.5k dl). Alternatives to RedHatAI variant.
- **[NEW 2026-05-09]** v0.20.x prefix-caching + spec-decode regression reported by arctic.gus (forum, May 7-8). Reinforces v0.20.x HOLD.
- **[NEW 2026-05-09]** Spark Arena leaderboard table is JS-rendered behind a Firestore App Check ACL — anonymous reads denied. Numeric `arena_top_*` tracking values frozen at last manual capture (2026-04-30) until an authenticated path or alternative ranking source is wired in.
- **[RESOLVED 2026-05-09]** qwen35 + bge-m3 swap pressure cleared via planned restart (Entry 064). qwen35 EC: 2.16 GiB → 264 MB (-88%); bge-m3 EC: 1.54 GiB → 0 kB. Restart pattern repeatable for future runs.
- **[NEW 2026-05-09]** **Entry 052 post-firmware baseline (65.9/174.7/394.3/634.0 c1/c4/c8/c16) is suspect.** Post-rollback re-bench (Entry 064, no functional changes) measured 58.3/161.5/374.1/552.7 — within ±3% of pre-firmware Entry 050 baseline (59.9/166.2/373.8/564.0). Schedule controlled re-bench to determine whether Entry 052 "+10%" firmware gain is reproducible. If not, downgrade `single_request_tok_s` to ~60 tok/s.
- **[NEW 2026-05-09]** **Prefix caching re-test deferred.** First attempt rolled back (Entry 064): synthetic bench prompts (~15-20 tokens) shorter than 16-token cache block boundary → 0 hits across 3151 queries; -7-9% throughput from pure overhead. Future test must use realistic ≥200-token shared system prompt + multiple varying user messages.
- **[NEW 2026-05-09]** qwen3-embed EngineCore (1.26 GiB swap) and gliner (1.74 GiB swap) NOT addressed in Entry 064. Defer to next maintenance cycle if they grow.
- **[RESOLVED 2026-04-24]** MTP=2 on Qwen3.6 — ablation benchmark (Entry 043-044): MTP degrades c4 by 14.9% but improves c8 by 24.7% and c16 by 19.6%. c1 tied (~51 tok/s). Decision: KEEP MTP. Primary workload is pipeline at c8-c16 where MTP wins.
- **[RESOLVED 2026-04-30]** Firmware update applied (Entry 050). Post-firmware benchmark (Entry 052): c1 +10.0%, c4 +5.1%, c8 +5.5%, c16 +12.4%. c16 634.0 tok/s is new project record.
- **[ACTION 2026-04-30]** eugr v0.20.1rc1.dev96+cu132 + FlashInfer 0.6.9: published Apr 30. 2 minor versions ahead of us. Previous 0.19.2rc1 rejection doesn't apply — fresh evaluation needed.
- **[ACTION 2026-04-30]** Pre-quant FP8 hang rule INVALIDATED for Qwen3.6: Seth Hobson's Arena entry + community forum reports + model availability all confirm Qwen3.6-35B-A3B-FP8 works on v0.20.0. Re-test on our cu132 image.
- **[NEW 2026-04-30]** vLLM-Tune (serapis): kernel tuning CLI for Triton FP8/MoE. +58% prefill, +9.5% decode on Qwen3.6-35B-A3B-FP8. Test compatibility with cu132+MTP config.
- **[NEW 2026-04-30]** FlashQLA: linear attention kernels, 2x speedup claimed vs FlashInfer. SM90+ but jwarner says GB10 works. No vLLM integration yet. Immature — monitor.
- **[NEW 2026-04-30]** DFlash speculative decoding: 91-97 tok/s with NVFP4, 80+ with AWQ. Alternative to MTP. Not in mainline vLLM. joshua.dale.warner leading.
- **[SCOPED 2026-04-30]** NVFP4/INT4 quantization path scoped (Entry 060). Four paths: AWQ INT4 (works today on cu132, ~70-85 tok/s estimate), PrismaQuant 4.75-bit (~75-90 tok/s, quality 88/100), NVFP4 without DFlash (~85-100 tok/s, needs flashinfer_cutlass build), NVFP4+DFlash (~95-127 tok/s, unmerged). **Immediate action:** Run AWQ INT4 minimum viable experiment (45 min, works on current image) — data point needed before any further INT4 investment. Defer NVFP4+DFlash until DFlash merges to mainline. See Entry 060 for full decision matrix.
- **[SCOPED 2026-04-30]** PrismaQuant: now #1 Arena at 95.11 tok/s (Sean Williams, INT4 4.75-bit + DFlash). Quality 88/100 vs FP8 91/100 (3.3% gap on opaque internal score). Run after AWQ INT4 experiment if AWQ shows >10% c1 gain and quality holds.
- **[UPDATE]** vLLM v0.20.0 GA (Apr 27): CUDA 13.0 default, PyTorch 2.11, DeepGEMM integrated, TurboQuant 2-bit KV. No SM121-specific improvements. HOLD for now — MoE refactor regression risk.
- GPU power-draw throttle bug: after crash/sleep, GPU enters 14W/513 MHz cap → throughput halves. Fix: wall power cycle (unplug 1 min), not reboot. Systemic across OEM variants.
- Sparkview (github.com/parallelArchitect/sparkview): GB10-aware GPU monitor with PSI pressure, clock state, unified memory handling. Consider installing.
- GB10 bandwidth baseline (parallelArchitect, Apr 25): GPU read bandwidth drops 44% under inference load (161→90 GB/s). Confirms memory bandwidth as primary bottleneck, not coherence.
- Qwen3.6-27B: dense 27B, Gated DeltaNet hybrid. Bandwidth-limited ~7.8 tok/s on GB10. Not primary model candidate.
- Tool Eval Bench CLI (SerraphimSerapis): Qwen3.6 scores 100/100 on ToolCall-15.
- Thermal shutdown issue systemic across DGX Spark OEMs. Throttle GPU clocks to 2100-2400 MHz if experiencing shutdowns.
- Qwen3.6-Plus: still API-only (Apr 30), no HF weights. Monitor for open-weight release.
- Qwen4 monitor: no announcement as of 2026-04-30, prediction markets suggest before July 2026
- Hybrid INT4+FP8 checkpoint achieves 108-125 tok/s synthetic, ~80 sustained. Requires custom checkpoint build.
- DanTup/spark-evals GitHub repo — systematic Inspect AI quality evals across quant formats.
- **[SCOPED 2026-04-30]** Gemma 4 status researched (Entry 061). Community 26B NVFP4: 52 tok/s c1 (up from 38.9); still 21% below Qwen3.6. Structured output NOT fixed — PRs #39138 (xgrammar bypass) and #40099 (repetition loops) both unmerged. eugr "recipe fixes" were Python env fixes, not throughput/correctness fixes. InstantTensor broke Gemma 4 init; safetensors workaround degrades perf 75%. **Decision: DO NOT schedule experiment until both PRs merge.** bg-digitalservices NVFP4 checkpoint (16.5 GB, 97.6% quality) is the best 26B option when experiment is viable.
- InstantTensor in eugr/spark-vllm-docker — operator fusion library. Confirmed broke Gemma 4 compatibility (vincenzoa, Apr 29-30). Workaround: safetensors mode but 75% perf regression.
- v0.20.0 stability: one tester reverted to v0.19.2 on Qwen3.6-27B (forum, Apr 30). Not yet validated for Qwen3.6-35B-A3B.
- **[RESOLVED 2026-04-24]** eugr 0.19.2rc1 — REJECTED (Entry 045-046). Superseded by v0.20.1rc1 evaluation.
- **[RESOLVED 2026-04-24]** Tool calling parser confirmed correct. `qwen3_coder` parses Qwen3.6 XML format.
- **[RESOLVED]** vLLM v0.19.1 released 2026-04-18. TurboQuant landed in v0.20.0.
