#!/usr/bin/env bash
# spark-chromadb-memwatch.sh — frequent (10-min) chromadb memory-growth sampler.
# TEMPORARY DIAGNOSTIC (LAB_NOTEBOOK Entry 116, 2026-08-03): instruments the working
# theory that chromadb's memory footprint grows against its 10GiB cgroup cap until
# host RAM saturates and swap-thrashes badly enough to starve SSH/network — then
# something restarts chromadb and the cycle resets. Logs EVERY sample (not
# quiet-on-healthy like the other routines) so the growth curve is visible; also
# alerts if chromadb restarts again or host memory gets dangerously tight. Read-only.
# Remove from crontab (ops/spark-ops.cron) once the growth pattern is confirmed/refuted.
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh
ROUTINE=chromadb-memwatch
STATE="$SPARK_OPS_LOG_DIR/chromadb-memwatch.state"

RAW="$(spark_ssh 'bash -s' <<'REMOTE'
echo "CHROMA_RESTARTS=$(docker inspect -f '{{.RestartCount}}' chromadb 2>/dev/null)"
echo "CHROMA_STARTED=$(docker inspect -f '{{.State.StartedAt}}' chromadb 2>/dev/null)"
echo "CHROMA_MEMRAW=$(docker stats --no-stream --format '{{.MemUsage}}' chromadb 2>/dev/null)"
echo "SWAP_USED_MB=$(free -m | awk '/^Swap:/{print $3}')"
echo "MEM_AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')"
echo "LOAD1=$(cut -d' ' -f1 /proc/loadavg)"
REMOTE
)"
SSH_RC=$?
declare -A V; while IFS='=' read -r k v; do [ -n "$k" ] && V[$k]="$v"; done <<<"$RAW"

if [ $SSH_RC -ne 0 ] || [ -z "${V[CHROMA_MEMRAW]:-}" ]; then
  notify "$ROUTINE" CRIT "SSH/gather failed (rc=$SSH_RC) — cannot sample chromadb memory (box may be unreachable again)"
  exit 1
fi

PREV_RESTARTS=""
[ -f "$STATE" ] && PREV_RESTARTS="$(cut -d' ' -f1 "$STATE" 2>/dev/null)"
echo "${V[CHROMA_RESTARTS]} ${V[CHROMA_STARTED]}" > "$STATE"

ANOM=()
if [ -n "$PREV_RESTARTS" ] && [ "${V[CHROMA_RESTARTS]:-0}" -gt "$PREV_RESTARTS" ] 2>/dev/null; then
  ANOM+=("chromadb restarted again ($PREV_RESTARTS->${V[CHROMA_RESTARTS]}) at ${V[CHROMA_STARTED]}")
fi
AVAIL="${V[MEM_AVAIL_MB]:-99999}"
[ "$AVAIL" -lt 500 ] 2>/dev/null && ANOM+=("host available mem ${AVAIL}MB <500 (thrash risk)")
SW="${V[SWAP_USED_MB]:-0}"
[ "$SW" -ge 12000 ] 2>/dev/null && ANOM+=("swap ${SW}MB >=12G")

SUMMARY="chroma_mem=${V[CHROMA_MEMRAW]:-NA} restarts=${V[CHROMA_RESTARTS]:-NA} avail=${AVAIL}MB swap=${SW}MB load1=${V[LOAD1]:-NA}"
if [ ${#ANOM[@]} -gt 0 ]; then
  notify "$ROUTINE" ANOMALY "$(IFS='; '; echo "${ANOM[*]}") | $SUMMARY"
  exit 1
fi
log_ok "$ROUTINE" "$SUMMARY"
echo "OK: $SUMMARY"
