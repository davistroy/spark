# ADR-0001 — NVFP4 quantization on SM121 (GB10): eval-gated adoption decision

**Status:** Proposed (eval-gated — resolved by Phase 3 of `IMPLEMENTATION_PLAN_2026-06-30.md`)
**Date:** 2026-06-30
**Deciders:** Troy Davis (owner), Claude Code (analysis)
**Related:** LAB_NOTEBOOK Entry 081 (recon), Entry 082 (audit); SPARK_ROADMAP §5.3 (Arm C, absorbed); CLAUDE.md verified-rule "no viable INT4 path on SM121"

## Context

Production runs `Qwen/Qwen3.6-35B-A3B-FP8` (native pre-quant FP8, BF16 KV, MTP=2) on the custom
`vllm-cu132-test:latest` build (vLLM v0.19.1rc1.dev219+cu132). Authoritative throughput baseline
(kernel 1021, Entry 080): **c1 73.1 / c4 186.7 / c8 406.9 / c16 730.5 tok/s.**

A standing verified rule (CLAUDE.md, Entry 068) states: *"Marlin WNA16 MoE CUDA graph capture hangs
on SM121 → must use `--enforce-eager`; no viable INT4/4-bit quantization path exists for SM121."*
**That rule was derived specifically from AWQ-INT4** (compressed-tensors, group_size=32, num_bits=4).

The 2026-06-30 recon (Entry 081) surfaced a cross-correlated contradiction from **three independent
sources**:
1. **Arena:** Luis Poveda — `nvidia/Qwen3.6-35B-A3B-NVFP4` on **portable vLLM** (not Atlas),
   **118.91 tok/s c1** (+78% vs prod 66.9, +48% vs best FP8-vLLM 80.27), submitted 2026-06-30,
   running **Marlin MoE with FULL CUDA graphs (no `--enforce-eager`)** + FlashInfer attn + MTP=3 +
   FP8 KV. Corroborated by Hon Lam Gabriel Leung (76.81, 06-27).
2. **eugr/spark-vllm-docker:** new `qwen3.6-35b-a3b-nvfp4.yaml` recipe (Marlin MoE + FlashInfer +
   MTP=3 + FP8 KV, `VLLM_MARLIN_USE_ATOMIC_ADD=1`).
3. **NVIDIA forum:** iromu reports 249–268 tok/s decode @ 92.2% spec-accept on vLLM nightly.

**NVFP4 is a distinct, native-Blackwell FP4 format** (hardware FP4 tensor cores on sm_121), NOT the
AWQ-INT4/Marlin-WNA16 path the rejection rule covers. The current build already registers
`modelopt_fp4` in its quantization methods. This is the first credible challenge to the "no 4-bit
path" rule and the single largest potential single-node throughput gain observed for our exact model
on a deployable runtime.

## Decision

**Run a gated, sandboxed NVFP4 evaluation arm before any adoption decision** (do not adopt on the
strength of external leaderboard numbers). The eval is staged to separate the two confounded
variables — *quantization* (NVFP4 vs FP8) and *build* (v0.23.x vs current v0.19.1) — via a
cheap probe on the current build plus a faithful run on the v0.23.x build with an FP8/MTP=2 build
control.

**Adoption gate (all required, per Phase 5 standard):**
- Throughput: **≥ +5% c8 aggregate vs MTP baseline 406.9** (i.e. ≥ **427.2** tok/s)
- Quality: **AR ≥ 28/30** (same fixtures as Entry 080; 4-bit quality scrutiny)
- Stability: clean 30-min soak to qualify; **12-hour soak + explicit user approval** before any
  production switch (Entry 073 precedent — eval ≠ adoption)

## Alternatives considered

| Alternative | Why not (now) |
|---|---|
| **Keep FP8 pre-quant + MTP=2 (status quo)** | Default if the eval fails the gate. Proven, stable, 0-restart. No action = this. |
| **DFlash spec-decode** | REJECTED Entry 080 — single-stream optimizer, −16.8% c8. Wrong for shared endpoint. |
| **Atlas runtime NVFP4** (218.85 tok/s) | Non-portable (proprietary, AGPLv3), single-stream-skewed at concurrency, immature. Not deployable on our vLLM stack. |
| **AWQ-INT4 / Marlin-WNA16** | REJECTED Entry 068 — CUDA-graph hang on SM121, −10 to −28% vs FP8. This is the rule NVFP4 is distinct from. |

## Consequences

- **If adopted:** potential large c1/c8 throughput gain + ~half the weight memory (more KV headroom);
  amends the CLAUDE.md verified rule to "no viable *INT4/AWQ-WNA16* path; NVFP4 (native FP4) is
  viable" — a material correction to project knowledge.
- **If rejected:** confirms FP8 pre-quant remains optimal on SM121 for this model; the rule stands
  with the AWQ-specific clarification already noted.
- **Risks during eval:** flashinfer #2776 (NVFP4-MoE CUDA-graph-capture crash — build-dependent;
  enforce-eager fallback); v0.23.x ~12h cuDNN graph-corruption bug (soak-gated); 4-bit quality
  regression (AR-gated). All sandboxed — production untouched until gate + approval.
