#!/usr/bin/env bash
# install-cron.sh — append the spark-ops entries to the current crontab (idempotent).
# Re-running replaces the spark-ops block; never touches your other cron lines.
set -euo pipefail
cd "$(dirname "$0")"
MARK="# >>> spark-ops >>>"; ENDMARK="# <<< spark-ops <<<"
current="$(crontab -l 2>/dev/null || true)"
cleaned="$(printf '%s\n' "$current" | sed "/$(printf '%s' "$MARK" | sed 's/[][\.*^$/]/\\&/g')/,/$(printf '%s' "$ENDMARK" | sed 's/[][\.*^$/]/\\&/g')/d")"
{ printf '%s\n' "$cleaned" | sed '/^$/d'; echo "$MARK"; cat spark-ops.cron; echo "$ENDMARK"; } | crontab -
echo "Installed. spark-ops block now in crontab:"; crontab -l | sed -n "/$MARK/,/$ENDMARK/p"
