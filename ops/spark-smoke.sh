#!/usr/bin/env bash
# spark-smoke.sh — weekly NON-DISRUPTIVE inference smoke + perf-drift probe.
# Hits the LIVE qwen35 endpoint (does NOT stop it). Catches generation-path
# failures and throughput regression that /health 200 can miss.
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh
ROUTINE=smoke

# Baseline c1 (kernel 1021, Entry 080) = 73.1 tok/s. Warn well below to avoid false alarms.
TOKS_WARN="${SPARK_SMOKE_TOKS_WARN:-45}"

OUT="$(spark_ssh 'bash -s' <<'REMOTE'
body='{"model":"spark-llm","messages":[{"role":"user","content":"List three practical uses for a Raspberry Pi, one sentence each."}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
# Warm-up (discarded): absorbs one-time Triton JIT after a restart so the measured
# request reflects steady-state decode, not cold-start compile.
curl -s --max-time 120 -H 'Content-Type: application/json' \
  -d '{"model":"spark-llm","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  http://localhost:8000/v1/chat/completions >/dev/null 2>&1
t0=$(date +%s.%N)
resp=$(curl -s --max-time 120 -H 'Content-Type: application/json' -d "$body" http://localhost:8000/v1/chat/completions 2>/dev/null)
t1=$(date +%s.%N)
el=$(awk "BEGIN{print $t1-$t0}")
ct=$(echo "$resp" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin); print(d["usage"]["completion_tokens"])
except Exception: print(0)' 2>/dev/null)
echo "ELAPSED=$el"; echo "COMP_TOKENS=$ct"
# MTP acceptance from /metrics (best-effort)
m=$(curl -s --max-time 8 http://localhost:8000/metrics 2>/dev/null)
acc=$(echo "$m" | awk '/^vllm:spec_decode_num_accepted_tokens_total/{a=$2} /^vllm:spec_decode_num_draft_tokens_total/{d=$2} END{if(d>0)printf "%.1f",100*a/d; else print "NA"}')
echo "MTP_ACCEPT_PCT=$acc"
echo "DAYS_UP=$(( ( $(date +%s) - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' qwen35)" +%s) ) / 86400 ))"
echo "SWAP_MB=$(free -m | awk '/Swap/{print $3}')"
REMOTE
)"
RC=$?
declare -A V; while IFS='=' read -r k v; do [ -n "$k" ] && V[$k]="$v"; done <<<"$OUT"

if [ $RC -ne 0 ] || [ -z "${V[COMP_TOKENS]:-}" ]; then
  notify "$ROUTINE" CRIT "SSH/probe failed (rc=$RC)"; exit 1; fi
CT="${V[COMP_TOKENS]:-0}"
if [ "$CT" -lt 1 ] 2>/dev/null; then
  notify "$ROUTINE" CRIT "inference returned 0 tokens (generation path broken though /health may be 200)"; exit 1; fi
TPS=$(awk "BEGIN{if(${V[ELAPSED]:-0}>0)printf \"%.1f\", $CT/${V[ELAPSED]}; else print 0}")
SUMMARY="tok/s=$TPS (c1; ${CT} tok in ${V[ELAPSED]}s) MTP_accept=${V[MTP_ACCEPT_PCT]:-NA}% up=${V[DAYS_UP]:-?}d swap=${V[SWAP_MB]:-?}MB"
if awk "BEGIN{exit !($TPS < $TOKS_WARN)}"; then
  notify "$ROUTINE" ANOMALY "throughput regression: $SUMMARY (baseline c1~73)"; exit 1; fi
log_ok "$ROUTINE" "$SUMMARY"; echo "OK: $SUMMARY"
