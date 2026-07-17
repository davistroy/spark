#!/usr/bin/env bash
# _common.sh — shared helpers for Spark ops routines (sourced, not executed).
# Runs ON an SSH-capable host (this VM / obvm). Reaches the Spark over SSH.

# Optional local, gitignored config (alert endpoint / overrides). Kept out of the
# public repo. Use `:=` form inside these files (e.g. `: "${SPARK_ALERT_WEBHOOK:=URL}"`)
# so a value already exported in the environment still wins. See ops/spark-ops.env.example.
for _envf in "$HOME/.spark-ops.env" "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/spark-ops.env"; do
  [ -r "$_envf" ] && . "$_envf"
done
unset _envf

SPARK_SSH_KEY="${SPARK_SSH_KEY:-$HOME/.ssh/id_claude_code}"
SPARK_HOST="${SPARK_HOST:-claude@spark.k4jda.net}"
SPARK_OPS_LOG_DIR="${SPARK_OPS_LOG_DIR:-$HOME/spark-ops-logs}"
# Alert delivery. Preferred: Pushover (SPARK_PUSHOVER_TOKEN + SPARK_PUSHOVER_USER).
# If those are empty, notify() resolves them from Bitwarden at runtime (see
# _spark_resolve_pushover). Generic fallback: SPARK_ALERT_WEBHOOK (raw-body webhook,
# e.g. ntfy/Slack/Discord). All optional; empty = log-only.
SPARK_PUSHOVER_TOKEN="${SPARK_PUSHOVER_TOKEN:-}"
SPARK_PUSHOVER_USER="${SPARK_PUSHOVER_USER:-}"
SPARK_ALERT_WEBHOOK="${SPARK_ALERT_WEBHOOK:-}"
# Bitwarden Secrets Manager item names (per the "tokens in Bitwarden" rule).
SPARK_BWS_PUSHOVER_TOKEN_KEY="${SPARK_BWS_PUSHOVER_TOKEN_KEY:-PUSHOVER_APP_TOKEN}"
SPARK_BWS_PUSHOVER_USER_KEY="${SPARK_BWS_PUSHOVER_USER_KEY:-PUSHOVER_USER_KEY}"

mkdir -p "$SPARK_OPS_LOG_DIR" 2>/dev/null || true

# spark_ssh <remote-command> — run a command on the Spark, fail fast.
spark_ssh() {
  ssh -i "$SPARK_SSH_KEY" -o ConnectTimeout=15 -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "$SPARK_HOST" "$@"
}

# _spark_resolve_pushover — populate SPARK_PUSHOVER_{TOKEN,USER} from Bitwarden if not
# already set in the environment. Keeps the actual tokens in Bitwarden (never on disk).
# Returns 0 only if both creds are available afterward.
_spark_resolve_pushover() {
  [ -n "$SPARK_PUSHOVER_TOKEN" ] && [ -n "$SPARK_PUSHOVER_USER" ] && return 0
  local bws_bin; bws_bin="$(command -v bws 2>/dev/null || echo "$HOME/bin/bws")"
  [ -x "$bws_bin" ] || return 1
  # Prefer the canonical BWS token from claude-env.sh: cron has no interactive shell,
  # and an inherited env token may be stale. Fall back to an exported token if absent.
  if [ -r "$HOME/.config/claude-env.sh" ]; then
    eval "$(grep -m1 '^export BWS_ACCESS_TOKEN=' "$HOME/.config/claude-env.sh")" 2>/dev/null
  fi
  [ -n "${BWS_ACCESS_TOKEN:-}" ] || return 1
  local json; json="$("$bws_bin" secret list 2>/dev/null)" || return 1
  [ -n "$json" ] || return 1
  SPARK_PUSHOVER_TOKEN="$(printf '%s' "$json" | _spark_json_val "$SPARK_BWS_PUSHOVER_TOKEN_KEY")"
  SPARK_PUSHOVER_USER="$(printf '%s' "$json" | _spark_json_val "$SPARK_BWS_PUSHOVER_USER_KEY")"
  [ -n "$SPARK_PUSHOVER_TOKEN" ] && [ -n "$SPARK_PUSHOVER_USER" ]
}

# _spark_json_val <key> — read a bws `secret list` JSON array on stdin, print the
# `value` whose `key` matches. Uses python3 (already present on the ops host).
_spark_json_val() {
  python3 -c 'import sys,json
k=sys.argv[1]
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
print(next((s.get("value","") for s in d if s.get("key")==k),""))' "$1" 2>/dev/null
}

# notify <routine> <severity> <summary>  — deliver an alert (stdout + log + Pushover/webhook).
# Quiet-on-healthy design: only called on anomaly. Logs a breadcrumb if delivery fails.
notify() {
  local routine="$1" sev="$2" summary="$3"
  local line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [$routine] $sev: $summary"
  echo "$line" >&2
  echo "$line" >> "$SPARK_OPS_LOG_DIR/${routine}.log"

  _spark_resolve_pushover || true
  local rc=0 via=""
  if [ -n "$SPARK_PUSHOVER_TOKEN" ] && [ -n "$SPARK_PUSHOVER_USER" ]; then
    local prio=0
    case "$sev" in CRIT*) prio=1 ;; WARN*) prio=0 ;; *) prio=-1 ;; esac
    curl -fsS --max-time 10 \
      --data-urlencode "token=${SPARK_PUSHOVER_TOKEN}" \
      --data-urlencode "user=${SPARK_PUSHOVER_USER}" \
      --data-urlencode "title=spark-${routine} ${sev}" \
      --data-urlencode "message=${summary}" \
      --data-urlencode "priority=${prio}" \
      https://api.pushover.net/1/messages.json >/dev/null 2>&1
    rc=$?; via="pushover"
  elif [ -n "$SPARK_ALERT_WEBHOOK" ]; then
    curl -fsS --max-time 10 -H "Title: spark-${routine} ${sev}" \
         -d "$summary" "$SPARK_ALERT_WEBHOOK" >/dev/null 2>&1
    rc=$?; via="webhook"
  fi
  if [ -n "$via" ] && [ "$rc" -ne 0 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [$routine] ALERT-DELIVERY-FAILED via $via (curl rc=$rc)" \
      >> "$SPARK_OPS_LOG_DIR/${routine}.log"
  fi
}

# log_ok <routine> <summary> — record a clean run (no alert).
log_ok() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [$1] OK: $2" >> "$SPARK_OPS_LOG_DIR/$1.log"
}
