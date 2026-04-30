# Spark Performance Baseline

Last updated: 2026-04-30
Last recon: 2026-04-30 (Entry 049)
Last benchmark: 2026-04-30 (Entry 052 — post-firmware)

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
| kv_cache_memory | 47.95 GiB (1,142,736 tokens, max concurrency 85.92x) |
| single_request_tok_s | 65.9 (post-firmware, 2026-04-30 Entry 052) — prev: 59.9 (2026-04-24) |
| c4_aggregate_tok_s | 174.7 — prev: 166.2 |
| c8_aggregate_tok_s | 394.3 — prev: 373.8 |
| c16_aggregate_tok_s | 634.0 — prev: 564.0 |
| firmware_gain | c1 +10.0%, c4 +5.1%, c8 +5.5%, c16 +12.4% (all levels improved) |
| startup_time | ~364s (warm Triton cache, cu132-cu132 dir) |
| triton_cache | /home/claude/.cache/triton-cu132 (separate from cu130 cache) |

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
| vllm_last_checked_version | v0.20.0 (prerelease 2026-04-23) / v0.19.1 (stable 2026-04-18) |
| vllm_latest_observed | v0.20.0 GA (2026-04-27, MEDIUM — CUDA 13.0 default, PyTorch 2.11, FlashAttention 4, MoE refactor, DeepGEMM integrated) |
| qwen_current_model | Qwen/Qwen3.6-35B-A3B (adopted 2026-04-23) |

## spark-vllm-docker Tracking
| Field | Value |
|-------|-------|
| svd_last_checked_date | 2026-04-30 (eugr/spark-vllm-docker — v0.20.1rc1.dev96+cu132, FlashInfer 0.6.9, experimental b12x support, Gemma 4 recipe fixes) |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-04-30 |
| forum_posts_since_049 | ~30 active topics since Apr 24 (firmware update +6%, vLLM-Tune, FlashQLA, DFlash 91-97 tok/s, v0.20.0 early reports, GB10 bandwidth baseline) |

## Gemma 4 Reference Numbers (2026-04-11, Entries 020-021)
| Model | Quant | c1 tok/s | c8 agg | c16 agg | Notes |
|-------|-------|---------|--------|---------|-------|
| 26B-A4B (MoE) | FP8 | 38.9 | 257.6 | 387.5 | Best Gemma config. Guided JSON broken. |
| 26B-A4B (MoE) | BF16 | 23.6 | 158.7 | 206.7 | TRITON_ATTN forced (hetero heads) |
| 31B (Dense) | NVFP4 | 6.8 | 54.0 | — | Bandwidth-bound. FLASHINFER_CUTLASS GEMM. |
| 31B (Dense) | BF16 | 3.7 | 28.2 | — | Bandwidth-bound. Matches community exactly. |

## Recon Triggers
<!-- spark-recon reads these to prioritize findings. Format: source | pattern (keywords, OR/AND logic) | action level + what to do | date added -->
<!-- Remove rows when resolved. Add rows after each experiment/research session. -->

| Source | Pattern | Action | Added |
|--------|---------|--------|-------|
| vllm_release | gemma4 AND (guided OR grammar OR xgrammar) | ACTION: test Gemma 4 guided JSON for pipeline | 2026-04-11 |
| vllm_release | DeepGEMM AND (SM12 OR SM121 OR Blackwell OR GB10) | ACTION: benchmark Qwen3.5 FP8 with DeepGEMM | 2026-04-11 |
| vllm_release | FlashInfer AND (heterogeneous OR mixed head) | INFO: could boost Gemma 4 single-request past 50 tok/s | 2026-04-11 |
| vllm_release | MXFP4 AND (online OR on-the-fly OR Qwen) | INFO: test MXFP4 quantization path on Qwen3.5 | 2026-04-11 |
| vllm_release | speculative AND (Qwen OR MoE) | INFO: test spec decode with Qwen3-0.6B draft model | 2026-04-11 |
| arena | fp8 AND {model_family} AND single-node > baseline_tok_s * 1.10 | ACTION: investigate config difference vs current baseline | 2026-04-11 |
| huggingface | Qwen3.6-Plus OR Qwen4 model weights | ACTION: benchmark day — full throughput + quality suite | 2026-04-11 |
| forum | gemma4 AND (guided JSON OR grammar OR structured output) fix | INFO: community confirmation of #39130 fix | 2026-04-11 |

## Watch Items
- **[RESOLVED 2026-04-24]** MTP=2 on Qwen3.6 — ablation benchmark (Entry 043-044): MTP degrades c4 by 14.9% but improves c8 by 24.7% and c16 by 19.6%. c1 tied (~51 tok/s). Decision: KEEP MTP. Primary workload is pipeline at c8-c16 where MTP wins.
- **[RESOLVED 2026-04-30]** Firmware update applied (Entry 050). Post-firmware benchmark (Entry 052): c1 +10.0%, c4 +5.1%, c8 +5.5%, c16 +12.4%. c16 634.0 tok/s is new project record.
- **[ACTION 2026-04-30]** eugr v0.20.1rc1.dev96+cu132 + FlashInfer 0.6.9: published Apr 30. 2 minor versions ahead of us. Previous 0.19.2rc1 rejection doesn't apply — fresh evaluation needed.
- **[ACTION 2026-04-30]** Pre-quant FP8 hang rule INVALIDATED for Qwen3.6: Seth Hobson's Arena entry + community forum reports + model availability all confirm Qwen3.6-35B-A3B-FP8 works on v0.20.0. Re-test on our cu132 image.
- **[NEW 2026-04-30]** vLLM-Tune (serapis): kernel tuning CLI for Triton FP8/MoE. +58% prefill, +9.5% decode on Qwen3.6-35B-A3B-FP8. Test compatibility with cu132+MTP config.
- **[NEW 2026-04-30]** FlashQLA: linear attention kernels, 2x speedup claimed vs FlashInfer. SM90+ but jwarner says GB10 works. No vLLM integration yet. Immature — monitor.
- **[NEW 2026-04-30]** DFlash speculative decoding: 91-97 tok/s with NVFP4, 80+ with AWQ. Alternative to MTP. Not in mainline vLLM. joshua.dale.warner leading.
- **[NEW 2026-04-30]** RedHatAI/Qwen3.6-35B-A3B-NVFP4: 127 tok/s with DFlash reported. Requires nightly cu130 + flashinfer_cutlass backend. Quality tradeoff unknown.
- **[UPDATE]** PrismaQuant: now #1 Arena at 95.11 tok/s (Sean Williams, INT4 4.75-bit + DFlash). Up from "monitor" to confirmed leader. Quality 88/100 vs FP8 91/100 (marginal penalty).
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
- Gemma 4 NVFP4 at 45 tok/s (vs our 38.9 FP8): delta explained by NVFP4 quantization. Guided JSON still untested.
- InstantTensor in eugr/spark-vllm-docker — operator fusion library. Note: broke Gemma 4 compatibility (vincenzoa, Apr 29).
- v0.20.0 stability: one tester reverted to v0.19.2 on Qwen3.6-27B (forum, Apr 30). Not yet validated for Qwen3.6-35B-A3B.
- **[RESOLVED 2026-04-24]** eugr 0.19.2rc1 — REJECTED (Entry 045-046). Superseded by v0.20.1rc1 evaluation.
- **[RESOLVED 2026-04-24]** Tool calling parser confirmed correct. `qwen3_coder` parses Qwen3.6 XML format.
- **[RESOLVED]** vLLM v0.19.1 released 2026-04-18. TurboQuant landed in v0.20.0.
