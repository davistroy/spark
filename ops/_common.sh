#!/usr/bin/env bash
# _common.sh — shared helpers for Spark ops routines (sourced, not executed).
# Runs ON an SSH-capable host (this VM / obvm). Reaches the Spark over SSH.

SPARK_SSH_KEY="${SPARK_SSH_KEY:-$HOME/.ssh/id_claude_code}"
SPARK_HOST="${SPARK_HOST:-claude@spark.k4jda.net}"
SPARK_OPS_LOG_DIR="${SPARK_OPS_LOG_DIR:-$HOME/spark-ops-logs}"
# Optional: set SPARK_ALERT_WEBHOOK to a URL that accepts a POST body (ntfy/Slack/Discord/etc).
SPARK_ALERT_WEBHOOK="${SPARK_ALERT_WEBHOOK:-}"

mkdir -p "$SPARK_OPS_LOG_DIR" 2>/dev/null || true

# spark_ssh <remote-command> — run a command on the Spark, fail fast.
spark_ssh() {
  ssh -i "$SPARK_SSH_KEY" -o ConnectTimeout=15 -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "$SPARK_HOST" "$@"
}

# notify <routine> <severity> <summary>  — deliver an alert (stdout + log + optional webhook).
notify() {
  local routine="$1" sev="$2" summary="$3"
  local line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [$routine] $sev: $summary"
  echo "$line" >&2
  echo "$line" >> "$SPARK_OPS_LOG_DIR/${routine}.log"
  if [ -n "$SPARK_ALERT_WEBHOOK" ]; then
    curl -fsS --max-time 10 -H "Title: spark-${routine} ${sev}" \
         -d "$summary" "$SPARK_ALERT_WEBHOOK" >/dev/null 2>&1 || true
  fi
}

# log_ok <routine> <summary> — record a clean run (no alert).
log_ok() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [$1] OK: $2" >> "$SPARK_OPS_LOG_DIR/$1.log"
}
