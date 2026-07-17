# Spark Ops — scheduled health/security/stability/perf routines

These run **from an SSH-capable host** (this VM, `obvm`) and reach the Spark over SSH
(`claude@spark.k4jda.net`, key `~/.ssh/id_claude_code`). They **read** the box; none
modify it. Each is **quiet on healthy** and, on anomaly, prints + logs + exits non-zero
(so a `MAILTO` cron mails it) + POSTs to `$SPARK_ALERT_WEBHOOK` if set.

| Script | Cadence | Goal | Disruptive | Checks |
|--------|---------|------|------------|--------|
| `spark-healthcheck.sh` | daily | health/stability | no | 8/8 containers up+healthy, `/health` 200 (8000/1/2/4), GPU temp/util/mem, swap, disk, restart-delta, fatal-log scan (CUDA/Xid/OOM/Traceback) |
| `spark-smoke.sh` | weekly | perf/stability | **no** | live c1 inference (warm-up first), tok/s vs baseline (~73), MTP acceptance, days-up, swap |
| `spark-security.sh` | weekly | security | no | SSH host-key integrity (vs rotated fp — CVE-2026-24218 reversion guard), external-port diff, key-backup presence, failed-auth surge, optional trivy/grype image CVE scan |
| `spark-audit-cron.sh` | weekly | health/perf | no | runs the LLM `personal-plugin:spark-audit` skill (config drift, versions, memory, best-practices). **Needs headless `claude` auth — disabled in cron until verified.** |

## Setup
```bash
ops/install-cron.sh          # append the spark-ops block (idempotent)
crontab -l                   # verify
```
> **Status: installed and running on `ubuntu-vm` (crontab active; logs accruing in `~/spark-ops-logs/`).** healthcheck (daily) + smoke + security (weekly) tested green; audit cron enabled (commit `9be3a77`). **Alert channel WIRED 2026-07-17** (Entry 112 follow-up) — ntfy webhook, delivery verified (HTTP 200).
**Alerts:** the webhook lives in **`~/.spark-ops.env`** (gitignored — the repo is public), which `ops/_common.sh` sources at runtime; copy `ops/spark-ops.env.example` to set it up. Anomalies POST to that `SPARK_ALERT_WEBHOOK` (ntfy/Slack/Discord) and also append to `~/spark-ops-logs/<routine>.log`. Optional email: set `MAILTO=` in `ops/spark-ops.cron` (needs a mailer). To rotate the ntfy topic, edit `~/.spark-ops.env` and re-subscribe.

## Config (env overrides)
`SPARK_SSH_KEY`, `SPARK_HOST`, `SPARK_ALERT_WEBHOOK`, `SPARK_OPS_LOG_DIR`,
`SPARK_EXPECTED_ED25519_FP` (security), `SPARK_SMOKE_TOKS_WARN`, `SPARK_ALLOW_PORTS`.

## Not cron-managed (continuous observability)
The on-box `gpu-exporter` (Entry 077) already feeds Prometheus (`gpu_xid_events_total`,
restart counts, temp/clock/power). The `spark-reliability` Grafana dashboard
(`grafana/spark-reliability-dashboard.json`) visualizes it. The daily cron above already
**alerts** on the same critical conditions, so Grafana alert rules are an optional add-on.

## Relationship to the cloud recon
`spark-recon-daily` (CCR trigger, cloud, web-only) tracks the **external landscape**
(Arena/vLLM/models/forum). These VM routines track the **live box**. Complementary.
