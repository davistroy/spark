# Spark Performance Baseline

Last updated: 2026-04-23
Last recon: 2026-04-24 (Entry 042)

## Current Config
| Field | Value |
|-------|-------|
| image | vllm-cu132-test:latest (v0.19.1rc1.dev219+cu132) — adopted 2026-04-23 |
| model | Qwen/Qwen3.6-35B-A3B (on-the-fly FP8) — adopted 2026-04-23, snapshot 53c43178507d69762986fbfa314f6e8d4d859409 |
| vllm_version | v0.19.1rc1.dev219+cu132 |
| speculative_decoding | MTP=2 (method: mtp, num_speculative_tokens: 2, acceptance rate 80.7%) |
| mtp_drafter | Qwen3_5MoeMTP, 34.16 GiB total model load |
| moe_backend | TRITON (auto-selected) |
| fp8_kernel | CutlassFP8ScaledMMLinearKernel |
| attention_backend | FLASHINFER |
| async_scheduling | Enabled |
| chunked_prefill | Enabled |
| single_request_tok_s | 51.2 (cu132+MTP, 2026-04-23 benchmark) |
| c4_aggregate_tok_s | 160.8 |
| c8_aggregate_tok_s | 384.4 |
| c16_aggregate_tok_s | 576.0 |
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
| arena_top_fp8_qwen35_tok_s | 70-81 (FP8+MTP=2, joshua.dale.warner optimizations thread) |
| arena_top_fp8_qwen35_entry | Qwen3.5-35B-A3B FP8+MTP=2 (joshua.dale.warner) |
| arena_top_hybrid_tok_s | 108-125 synthetic, ~80 sustained (INT4+FP8 hybrid + MTP=2) |
| arena_top_overall_tok_s | 73.33 |
| arena_top_overall_entry | Qwen3-Coder-Next-int4-AutoRound (single-node, INT4) |
| arena_top_overall_multinode | gpt-oss-120b (MXFP4, 2-node) — 75.96 tok/s (informational only) |

## Version Tracking
| Field | Value |
|-------|-------|
| vllm_last_checked_version | v0.20.0 (prerelease 2026-04-23) / v0.19.1 (stable 2026-04-18) |
| vllm_latest_observed | v0.20.0 (2026-04-23, prerelease, MEDIUM — CUDA 13.0 default, PyTorch 2.11, FlashAttention 4, MoE refactor) |
| qwen_current_model | Qwen/Qwen3.6-35B-A3B (adopted 2026-04-23) |

## spark-vllm-docker Tracking
| Field | Value |
|-------|-------|
| svd_last_checked_date | 2026-04-24 (eugr/spark-vllm-docker — v0.19.2rc1.dev154+cu132, FlashInfer 0.6.8, flashinfer_cutlass re-enabled, PR #40191 torch fix) |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-04-24 |
| forum_posts_since_042 | ~30+ active topics since Apr 15 (Qwen3.6 launch, PrismaQuant, GPU power-draw bug, Sparkview, DeepSeek V4, Tool Eval Bench, thermal shutdown) |

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
- **[RESOLVED 2026-04-24]** MTP=2 on Qwen3.6 — ablation benchmark (Entry 043-044): MTP degrades c4 by 14.9% but improves c8 by 24.7% and c16 by 19.6%. c1 tied (~51 tok/s). Decision: KEEP MTP. Primary workload is pipeline at c8-c16 where MTP wins. Forum reports of degradation validated at c1/c4 only.
- **[NEW]** PrismaQuant: mixed-precision quant framework (jwarner, Apr 22). 22 GB model at 87.8 tok/s, near-BF16 quality. Monitor for vLLM integration and community adoption.
- **[NEW]** GPU power-draw throttle bug: after crash/sleep, GPU enters 14W/513 MHz cap → throughput halves. Fix: wall power cycle (unplug 1 min), not reboot. Systemic across OEM variants.
- **[NEW]** Sparkview (github.com/parallelArchitect/sparkview): GB10-aware GPU monitor with PSI pressure, clock state, unified memory handling. Consider installing.
- **[NEW]** eugr 0.19.2rc1.dev154+cu132 build (Apr 23): flashinfer_cutlass re-enabled, FlashInfer 0.6.8, PR #40191 torch fix. Worth benchmarking.
- **[NEW]** vLLM v0.20.0 prerelease (Apr 23): CUDA 13.0 default, PyTorch 2.11, FlashAttention 4, TurboQuant 2-bit KV, MoE refactor. Monitor for stable release.
- **[NEW]** Qwen3.6-27B (Apr 22): dense 27B, Gated DeltaNet hybrid, FP8 on HF. Bandwidth-limited ~7.8 tok/s on GB10 (15.2 with MTP=3). Not primary model candidate but could serve as routing/fast model.
- **[NEW]** Qwen3.6-35B-A3B-FP8 official pre-quant on HF. Pre-quant hang rule (from Qwen3.5 experience) needs re-test on 3.6.
- **[NEW]** Tool Eval Bench CLI (SerraphimSerapis): Qwen3.6 scores 100/100 on ToolCall-15. Install: `uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git`
- **[NEW]** Thermal shutdown issue systemic across DGX Spark OEMs. Throttle GPU clocks to 2100-2400 MHz if experiencing shutdowns.
- Qwen3.6-Plus: still API-only (Apr 24), no HF weights. Monitor for open-weight release.
- **[RESOLVED]** vLLM v0.19.1 released 2026-04-18. TurboQuant landed in v0.20.0, not v0.19.1.
- Hybrid INT4+FP8 checkpoint achieves 108-125 tok/s synthetic, ~80 sustained. Requires custom checkpoint build. Future investigation.
- DanTup/spark-evals GitHub repo — systematic Inspect AI quality evals across quant formats.
- Qwen4 monitor: no announcement as of 2026-04-24, prediction markets suggest before July 2026
- Forum stream loading + gather-free Triton decode techniques: may be relevant for multi-request scenarios (c16, c32)
- Tool calling fix: `--tool-call-parser qwen3_xml` + enhanced jinja template. Untested on Qwen3.6. Test at next maintenance window.
- Gemma 4 NVFP4 at 45 tok/s (vs our 38.9 FP8): delta explained by NVFP4 quantization. Guided JSON still untested. Revisit when vLLM fixes structured output.
- InstantTensor in eugr/spark-vllm-docker — operator fusion library, monitor for perf claims
