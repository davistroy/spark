#!/usr/bin/env bash
# spark-security.sh — weekly security posture check for the DGX Spark.
# Host-key integrity (detect reversion to factory/shared keys), external-port
# diff, host-key backup presence, failed-auth scan, optional image CVE scan.
# Alert on anomaly. Reads the box; never modifies it.
set -uo pipefail
cd "$(dirname "$0")"; . ./_common.sh
ROUTINE=security

# Expected post-rotation ED25519 fingerprint (CVE-2026-24218, Entry 094). Override via env.
EXPECTED_FP="${SPARK_EXPECTED_ED25519_FP:-SHA256:xXNusbupisxTmURNJV5khmepwIoym0UldLz3020g14c}"
# Externally-listening TCP ports we expect (services + ssh + exporters). Space-separated.
ALLOW_PORTS="${SPARK_ALLOW_PORTS:-22 7473 7474 7687 8000 8001 8002 8003 8004 8005 9100 9400}"

OUT="$(spark_ssh 'bash -s' <<'REMOTE'
echo "ED25519_FP=$(ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub 2>/dev/null | awk '{print $2}')"
echo "BACKUP_PRESENT=$([ -d /home/claude/ssh-hostkey-backup-20260630 ] && echo yes || echo no)"
# distinct externally-listening tcp ports (0.0.0.0 / [::])
ports=$(ss -tlnH 2>/dev/null | awk '{print $4}' | grep -E '(^0\.0\.0\.0|^\*|^\[::\]):' | sed -E 's/.*:([0-9]+)$/\1/' | sort -un | tr '\n' ' ')
echo "LISTEN_PORTS=$ports"
# failed ssh auths in last 7 days (best-effort; needs log readability)
fa=$( { journalctl -u ssh --since "7 days ago" 2>/dev/null || sudo -n cat /var/log/auth.log 2>/dev/null; } | grep -ic "failed password\|invalid user" )
echo "FAILED_AUTH=$fa"
# optional image CVE scan (only if a scanner is installed; time-boxed)
if command -v trivy >/dev/null 2>&1; then
  crit=$(timeout 240 trivy image --quiet --severity CRITICAL --format json vllm-cu132-test:latest 2>/dev/null | grep -c '"Severity":"CRITICAL"')
  echo "IMG_SCAN=trivy CRIT=$crit (vllm-cu132-test:latest)"
elif command -v grype >/dev/null 2>&1; then
  crit=$(timeout 240 grype vllm-cu132-test:latest -o json 2>/dev/null | grep -c '"severity": "Critical"')
  echo "IMG_SCAN=grype CRIT=$crit (vllm-cu132-test:latest)"
else
  echo "IMG_SCAN=none (no trivy/grype installed)"
fi
REMOTE
)"
RC=$?
declare -A V; while IFS='=' read -r k v; do [ -n "$k" ] && V[$k]="$v"; done <<<"$OUT"
[ $RC -ne 0 ] && { notify "$ROUTINE" CRIT "SSH/gather failed (rc=$RC)"; exit 1; }

ANOM=()
# host-key integrity — the CVE regression guard
if [ "${V[ED25519_FP]:-}" != "$EXPECTED_FP" ]; then
  ANOM+=("HOST KEY CHANGED: got ${V[ED25519_FP]:-none}, expected $EXPECTED_FP (reversion/tamper? CVE-2026-24218)")
fi
[ "${V[BACKUP_PRESENT]:-no}" != "yes" ] && ANOM+=("host-key backup dir missing (rollback unavailable)")
# unexpected external ports
for p in ${V[LISTEN_PORTS]:-}; do
  case " $ALLOW_PORTS " in *" $p "*) :;; *) ANOM+=("unexpected external port $p listening");; esac
done
# failed auth surge
[ "${V[FAILED_AUTH]:-0}" -ge 50 ] 2>/dev/null && ANOM+=("${V[FAILED_AUTH]} failed SSH auths in 7d (>=50 — possible brute force)")
# image CVEs
case "${V[IMG_SCAN]:-}" in *CRIT=0*|*none*) :;; *CRIT=*) ANOM+=("CRITICAL image CVEs: ${V[IMG_SCAN]}");; esac

SUMMARY="hostkey=ok ports=[${V[LISTEN_PORTS]}] failed_auth=${V[FAILED_AUTH]:-?} ${V[IMG_SCAN]:-}"
if [ ${#ANOM[@]} -gt 0 ]; then
  notify "$ROUTINE" ANOMALY "$(IFS='; '; echo "${ANOM[*]}")"; exit 1; fi
log_ok "$ROUTINE" "$SUMMARY"; echo "OK: $SUMMARY"
