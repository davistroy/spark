# Spark Performance Baseline

Last updated: 2026-03-31
Last recon: 2026-03-31 (Entry 012)

## Current Config
| Field | Value |
|-------|-------|
| image | vllm-custom:sm121-inject |
| model | Qwen/Qwen3.5-35B-A3B |
| single_request_tok_s | 48.6 |
| c16_aggregate_tok_s | 311.7 |
| vllm_version | v0.17.0rc1 |

## Arena Tracking
| Field | Value |
|-------|-------|
| arena_top_fp8_qwen35_tok_s | 52.32 |
| arena_top_fp8_qwen35_entry | Huihui-Qwen3.5-35B-A3B-abliterated (Artyom) |
| arena_top_overall_tok_s | 75.96 |
| arena_top_overall_entry | gpt-oss-120b (MXFP4, 2-node) |

## Version Tracking
| Field | Value |
|-------|-------|
| vllm_last_checked_version | v0.17.1 |
| vllm_latest_observed | v0.18.1 (2026-03-31, not actionable — #38126 not included, Qwen3.5 regression) |
| qwen_current_model | Qwen/Qwen3.5-35B-A3B |

## spark-vllm-docker Tracking
| Field | Value |
|-------|-------|
| svd_last_checked_date | 2026-03-31 |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-03-31 |

## Watch Items
- #38126 (SM121 CMake fix) merged to vLLM main 2026-03-27, awaiting release — will eliminate need for custom kernel builds
- Test pre-quantized `Qwen/Qwen3.5-35B-A3B-FP8` model next config change (sus's Arena entry uses it at 50.75 tok/s)
- coolthor's MXFP4 analysis: 57-59 tok/s on Qwen3.5 — potential 20% improvement path
- FlashInfer PR #2913 GDC fix: include in next vLLM upgrade to prevent latent SM121 crashes
