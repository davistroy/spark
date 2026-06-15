# Autoresearch LLM Eval Study — FINAL STATUS

**Started:** 2026-05-16 16:24 UTC
**Completed:** 2026-05-18 ~17:30 UTC
**Wall-clock duration:** ~49h
**Mode:** Autonomous (user requested "complete as much as possible without me")

## All phases COMPLETE

| Phase | Status | Outcome |
|-------|--------|---------|
| A — Harness + parity gate | ✅ COMPLETE | Harness validated within ±10% of 2026-04-24 baseline |
| B — Qwen3.6-FP8 pre-quant (3 sub-runs) | ✅ COMPLETE | **Wins vs current production** at most concurrency (+10-15%) |
| C — GLM-4.7-Flash | ✅ COMPLETE | REJECT (42-65% slower than Qwen3.6) |
| D — Qwen3-Coder-Next-FP8 (u=0.80 only, u=0.70 skipped) | ✅ COMPLETE | REJECT (58-69% slower; 0% MTP acceptance) |
| E — Synthesis | ✅ COMPLETE | `MODEL_EVALUATION_2026_05.md` + LAB_NOTEBOOK Entries 070-072 |

## Recommendation summary

**Switch production from `Qwen/Qwen3.6-35B-A3B` (BF16+on-the-fly FP8+FP8 KV+MTP=2) to `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quant, BF16 KV, MTP=2).**

Expected gains:
- c=1: +12.7% (59.2 → 66.7 tok/s)
- c=4 agg: +11.8% (159.0 → 177.8)
- c=16 agg: +14.9% (525.0 → 603.1)
- AR quality: equivalent
- 4h stability: 100% success, 0 errors, 0 GPU drift (equivalent)

This contradicts Entry 054-055; vLLM version progression changed kernel selection.

## Production state

> **UPDATE 2026-05-18 (Entry 073):** The recommended switch was **APPLIED**. Production now runs `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quant, BF16 KV, MTP=2). Measured gains exceeded predictions (c=1 +13.0%, c=4 agg +25.1%, c=8 +14.4%, c=16 +29.3%). Rollback path documented in SPARK_BASELINE.md. The "deferred for user approval" note below is historical (true only 2026-05-16→18).

**Production qwen35 was restored after each eval phase** during the study. Post-study state (before the 2026-05-18 switch):
- `qwen35` (Qwen3.6-35B-A3B BF16+on-the-fly FP8) — running, healthy
- `qwen3-embed`, `gliner`, `chromadb`, `neo4j`, `ce-service`, `bge-m3`, `node-exporter` — all healthy

~~No production config change was made. The recommended switch is **deferred for user approval**.~~ (Superseded — switch applied 2026-05-18, Entry 073.)

## Files produced

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_PLAN_MODEL_EVAL.md` | The plan (created 2026-05-16) |
| `MODEL_EVALUATION_2026_05.md` | **Final comparative report + migration plan** (start here for review) |
| `LAB_NOTEBOOK.md` Entries 069-072 | Detailed per-phase logs |
| `SPARK_BASELINE.md` (updated) | Pending production switch + rejected models documented |
| `memory/MEMORY.md` (updated) | Long-form summary indexed |
| `/home/claude/llm-eval/` on Spark | Persistent harness for future studies |
| `/home/claude/llm-eval/results/<run_id>/` on Spark | Raw measurement data per run |

## Outstanding action items for user

1. **Review `MODEL_EVALUATION_2026_05.md`.** It contains the full comparison matrix, deep-dive per model, and step-by-step migration plan.
2. **Decide on production switch.** If approved, follow the migration plan in the report (~10 min downtime + 30 min validation).
3. **Consider committing**: there are new files (`IMPLEMENTATION_PLAN_MODEL_EVAL.md`, `MODEL_EVALUATION_2026_05.md`, `EVAL_STUDY_STATUS.md`) and modified files (`LAB_NOTEBOOK.md`, `SPARK_BASELINE.md`, `memory/MEMORY.md`) ready for git review.

## Issues encountered + fixed (for posterity)

1. **GLM-4.7 `glm4_moe_lite` arch unrecognized** by transformers 4.57.6. Built derived image `vllm-cu132-test:glm47` with transformers 5.0.0 (vLLM's `<5` pin is pip warning only — runtime accepts 5.0.0).
2. **`--attention-backend triton` rejected** in current vLLM (now namespaced as `triton_mla`/`triton_attn`/etc.). Fixed.
3. **Chain script trap syntax broken** by sed escaping. Replaced with `cleanup()` function.
4. **Pre-flight `[ -d PATH ]`** failed against root-owned HF cache. Replaced with `docker run` existence check.
5. **`docker compose up -d bge-m3 gliner` triggered qwen35 dependency** when eval container still present → name conflict, chain aborted. Phase D u=0.70 sub-run skipped (not load-bearing since Coder-Next already rejected).
6. **Coder-Next `q_scale=1.0` fallback** confirmed vllm#37554 manifestation; 0% MTP acceptance.
7. **Smoke probe `None` content** (Qwen reasoning_parser); patched with `enable_thinking=false`.

## What would beat the recommendation

Future Qwen3.6/3.7-family sibling at <=80B/3B-A active params, with native MTP layer compatible with the existing FP8 KV cache. Same-family upgrades likely high-confidence.

Re-evaluate this report when:
- vllm-project/vllm#37554 closes (would unlock Coder-Next and remove FP8 KV gotcha for Qwen3.6 too)
- vLLM ships 0.20.x stable (Entry 067 HOLD)
- Qwen3.7 / Qwen3.6-Coder-A3B release
- Nemotron-3-Nano-A3B (rumored long-context sibling)
