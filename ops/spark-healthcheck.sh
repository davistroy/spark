#!/usr/bin/env bash
# spark-healthcheck.sh — daily liveness/error check for the DGX Spark.
# Quiet on healthy (exit 0, log only); on anomaly -> notify + exit 1.
# Run from an SSH-capable host (this VM) via cron. Reads the box; never modifies it.
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh

ROUTINE=healthcheck
STATE="$SPARK_OPS_LOG_DIR/healthcheck.state"

# ---- gather (single SSH; structured KEY=VALUE) ----
RAW="$(spark_ssh 'bash -s' <<'REMOTE'
echo "CONTAINERS_RUNNING=$(docker ps -q | wc -l)"
unh=""; for c in qwen35 qwen3-embed gliner chromadb neo4j bge-m3 ce-service; do
  s=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$c" 2>/dev/null)
  [ "$s" = "healthy" ] || [ "$s" = "running" ] || unh="$unh $c:$s"
done
echo "UNHEALTHY=${unh# }"
for p in 8000 8001 8002 8004; do
  echo "HEALTH_$p=$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://localhost:$p/health 2>/dev/null)"
done
read -r t u pw <<<"$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ')"
echo "GPU_TEMP=${t:-NA}"; echo "GPU_UTIL=${u:-NA}"; echo "GPU_POWER=${pw:-NA}"
echo "SWAP_USED_MB=$(free -m | awk '/Swap/{print $3}')"
echo "DISK_PCT=$(df --output=pcent / | tail -1 | tr -dc '0-9')"
r=""; for c in qwen35 qwen3-embed gliner chromadb neo4j bge-m3 ce-service; do
  r="$r $c:$(docker inspect -f '{{.RestartCount}}' "$c" 2>/dev/null)"
done
echo "RESTARTS=${r# }"
errs=$(for c in qwen35 qwen3-embed gliner bge-m3; do docker logs --tail 300 "$c" 2>&1; done \
       | grep -icE 'CUDA error|illegal memory|Xid|OOM|out of memory|Traceback|EngineCore.*(died|fail)|core dumped'); echo "ERRORS=$errs"
REMOTE
)"
SSH_RC=$?

declare -A V
while IFS='=' read -r k v; do [ -n "$k" ] && V[$k]="$v"; done <<<"$RAW"

ANOM=()
if [ $SSH_RC -ne 0 ] || [ -z "${V[CONTAINERS_RUNNING]:-}" ]; then
  notify "$ROUTINE" CRIT "SSH/gather failed (rc=$SSH_RC) — box unreachable or docker down"; exit 1
fi
[ "${V[CONTAINERS_RUNNING]}" -lt 8 ] 2>/dev/null && ANOM+=("only ${V[CONTAINERS_RUNNING]}/8 containers running")
[ -n "${V[UNHEALTHY]}" ] && ANOM+=("unhealthy:${V[UNHEALTHY]}")
for p in 8000 8001 8002 8004; do
  [ "${V[HEALTH_$p]:-000}" != "200" ] && ANOM+=("port $p /health=${V[HEALTH_$p]:-000}")
done
T="${V[GPU_TEMP]:-0}"; [ "${T%.*}" -ge 85 ] 2>/dev/null && ANOM+=("GPU temp ${T}C >=85") || { [ "${T%.*}" -ge 80 ] 2>/dev/null && ANOM+=("GPU temp ${T}C >=80 (warn)"); }
SW="${V[SWAP_USED_MB]:-0}"; [ "$SW" -ge 15000 ] 2>/dev/null && ANOM+=("swap ${SW}MB >=15G (near full)") || { [ "$SW" -ge 12000 ] 2>/dev/null && ANOM+=("swap ${SW}MB >=12G (warn)"); }
DK="${V[DISK_PCT]:-0}"; [ "$DK" -ge 90 ] 2>/dev/null && ANOM+=("disk ${DK}% >=90") || { [ "$DK" -ge 80 ] 2>/dev/null && ANOM+=("disk ${DK}% >=80 (warn)"); }
[ "${V[ERRORS]:-0}" -gt 0 ] 2>/dev/null && ANOM+=("${V[ERRORS]} fatal log matches (CUDA/Xid/OOM/Traceback) in last 300 lines")

# restart-delta vs last run
if [ -f "$STATE" ]; then
  prev="$(cat "$STATE")"
  for kv in ${V[RESTARTS]}; do
    n="${kv%%:*}"; c="${kv##*:}"; pc=$(echo "$prev" | tr ' ' '\n' | grep "^$n:" | cut -d: -f2)
    [ -n "$pc" ] && [ "$c" -gt "$pc" ] 2>/dev/null && ANOM+=("$n restarted ($pc->$c)")
  done
fi
echo "${V[RESTARTS]}" > "$STATE"

SUMMARY="temp=${V[GPU_TEMP]}C util=${V[GPU_UTIL]}% swap=${V[SWAP_USED_MB]}MB disk=${V[DISK_PCT]}% errs=${V[ERRORS]}"
if [ ${#ANOM[@]} -gt 0 ]; then
  notify "$ROUTINE" ANOMALY "$(IFS='; '; echo "${ANOM[*]}") | $SUMMARY"
  exit 1
fi
log_ok "$ROUTINE" "$SUMMARY"
echo "OK: $SUMMARY"
