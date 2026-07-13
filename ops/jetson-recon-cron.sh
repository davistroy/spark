#!/usr/bin/env bash
# jetson-recon-cron.sh — weekly run of the LLM-driven jetson-recon skill, headless.
#
# PLACEMENT NOTE: this script lives in spark/ops/ (not a new jetson/ops/ tree) even
# though it operates entirely against ~/dev/personal/jetson/. Rationale: jetson has
# no ops/ directory, no crontab block, and no _common.sh of its own — scaffolding a
# full parallel ops system (new _common.sh, install-cron.sh, jetson-ops.cron,
# README, crontab marker block) for a single weekly job would be more moving parts
# than the job warrants. Reusing spark's ops/_common.sh (notify/log_ok helpers,
# SPARK_OPS_LOG_DIR) and its existing install-cron.sh mechanism is the
# least-invasive consistent home. The script simply `cd`s into the jetson repo
# for the git pull + claude invocation instead of the spark repo; logs land in
# spark's log dir under a distinctly-named `jetson-recon*.log` so they're never
# confused with spark's own recon logs. If jetson ops ever grows a second job,
# promote this into a real jetson/ops/ tree at that point.
#
# Same headless pattern as spark-recon-cron.sh / spark-audit-cron.sh. Recon is
# report+recommend only — never touches the live Jetson — but DOES edit
# LAB_NOTEBOOK.md (and, rarely, JETSON_BASELINE.md) in the jetson repo, so the
# prompt below asks the model to commit that directly to main and push (same
# convention as spark-recon-cron.sh).
#
# Requirements: `claude` CLI authenticated on this VM; SSH key to the Jetson
# (used internally by the jetson-recon skill's Check 5 live-health SSH).
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh
ROUTINE=jetson-recon
JETSON_REPO="${JETSON_REPO:-$HOME/dev/personal/jetson}"

command -v claude >/dev/null 2>&1 || { notify "$ROUTINE" CRIT "claude CLI not found on PATH"; exit 1; }
cd "$JETSON_REPO" || { notify "$ROUTINE" CRIT "repo not found: $JETSON_REPO"; exit 1; }
git pull --rebase --autostash --quiet origin main 2>/dev/null || true

LOG="$SPARK_OPS_LOG_DIR/jetson-recon-$(date -u +%Y%m%dT%H%M%SZ).log"

claude -p "/personal-plugin:jetson-recon

After the skill produces its report and appends the LAB_NOTEBOOK.md entry, commit
that change directly to main (message format: 'Weekly jetson recon <YYYY-MM-DD>
(Entry <N>): <overall classification> — <one-line summary>') and push to origin.
Do not update JETSON_BASELINE.md tracking values without explicit user confirmation
— since this is a headless run with no user present, skip that step and just
report what would change. Do not modify the running Jetson device." \
  --permission-mode bypassPermissions --max-turns 30 2>&1 | tee "$LOG"

# Safety net: if the model didn't commit as instructed, the tree is left dirty —
# flag it instead of letting it fail silently (breaks this script's own next
# `git pull --rebase`, and any manual jetson work done between runs).
if ! git diff --quiet -- LAB_NOTEBOOK.md JETSON_BASELINE.md 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  notify "$ROUTINE" WARN "weekly jetson recon left uncommitted changes in $JETSON_REPO — check LAB_NOTEBOOK.md/JETSON_BASELINE.md and commit manually (log: $LOG)"
  exit 1
fi

log_ok "$ROUTINE" "weekly jetson recon invoked (log: $LOG)"
