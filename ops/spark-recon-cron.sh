#!/usr/bin/env bash
# spark-recon-cron.sh — weekly run of the LLM-driven spark-recon skill, headless.
# Mirrors spark-audit-cron.sh's headless pattern. Recon is report+recommend only —
# it never touches the live Spark system — but it DOES edit LAB_NOTEBOOK.md (and,
# rarely, SPARK_BASELINE.md) in THIS repo. Unlike the audit skill (which commits
# to a branch + opens a PR), this repo's established recon convention is a direct
# commit to main (see `git log --oneline | grep "spark recon"` — daily CCR-driven
# recon entries already follow this pattern), so the prompt below asks the model
# to do the same after appending its entry.
#
# NOTE: a cloud CCR trigger (`spark-recon-daily`, see ops/README.md "Relationship
# to the cloud recon") already runs this skill daily from an anthropic_cloud
# sandbox (web-only, no live-box checks). This VM-cron job is a separate,
# lower-frequency (weekly) path that runs locally and does not depend on CCR
# availability — intentionally redundant, not a duplicate to remove.
#
# Requirements: `claude` CLI authenticated on this VM.
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh
ROUTINE=recon
REPO="${SPARK_REPO:-$HOME/dev/personal/spark}"

command -v claude >/dev/null 2>&1 || { notify "$ROUTINE" CRIT "claude CLI not found on PATH"; exit 1; }
cd "$REPO" || { notify "$ROUTINE" CRIT "repo not found: $REPO"; exit 1; }
git pull --rebase --autostash --quiet origin main 2>/dev/null || true

LOG="$SPARK_OPS_LOG_DIR/recon-$(date -u +%Y%m%dT%H%M%SZ).log"

claude -p "/personal-plugin:spark-recon

After the skill produces its report and appends the LAB_NOTEBOOK.md entry, commit
that change directly to main (message format: 'Weekly spark recon <YYYY-MM-DD>
(Entry <N>): <overall classification> — <one-line summary>') and push to origin.
Do not update SPARK_BASELINE.md tracking values without explicit user confirmation
— since this is a headless run with no user present, skip that step and just
report what would change. Do not modify the running Spark system." \
  --permission-mode bypassPermissions --max-turns 30 2>&1 | tee "$LOG"

# Safety net: if the model didn't commit as instructed, the tree is left dirty,
# which would break the next `git pull --rebase` (this script's own next run, or
# spark-audit-cron.sh on Tuesdays). Flag it instead of letting it fail silently.
if ! git diff --quiet -- LAB_NOTEBOOK.md SPARK_BASELINE.md 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  notify "$ROUTINE" WARN "weekly recon left uncommitted changes in $REPO — check LAB_NOTEBOOK.md/SPARK_BASELINE.md and commit manually (log: $LOG)"
  exit 1
fi

log_ok "$ROUTINE" "weekly recon invoked (log: $LOG)"
