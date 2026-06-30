#!/usr/bin/env bash
# spark-audit-cron.sh — weekly run of the LLM-driven spark-audit skill, headless.
# Unlike health/smoke/security (deterministic bash), the audit's value is nuanced
# best-practices comparison, so it runs the actual skill via the claude CLI.
#
# Requirements: `claude` CLI authenticated on this VM; SSH key to the Spark.
# Tune the permission posture to your setup. This wrapper is intentionally NOT
# auto-run during setup (it would spawn a nested Claude session).
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh
ROUTINE=audit
REPO="${SPARK_REPO:-$HOME/dev/personal/spark}"

command -v claude >/dev/null 2>&1 || { notify "$ROUTINE" CRIT "claude CLI not found on PATH"; exit 1; }
cd "$REPO" || { notify "$ROUTINE" CRIT "repo not found: $REPO"; exit 1; }
git pull --rebase --quiet origin main 2>/dev/null || true

# Runs the audit skill; it SSHes to the box, appends an audit Entry to LAB_NOTEBOOK,
# and (per skill) can open a PR. acceptEdits lets it write the entry; SSH/docker reads
# should be pre-allowlisted in your project settings.json to avoid prompts under cron.
claude -p "Invoke the personal-plugin:spark-audit skill against the live DGX Spark (SSH claude@spark.k4jda.net via ~/.ssh/id_claude_code). Run the read-only config audit, append a concise audit Entry to LAB_NOTEBOOK.md with the next Entry number, and commit to a new branch + open a PR. Do NOT modify the running system." \
  --permission-mode acceptEdits 2>&1 | tail -30

log_ok "$ROUTINE" "weekly audit invoked"
