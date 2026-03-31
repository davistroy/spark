# NOW_PLAN: Immediate DGX Spark Optimizations

**Date:** 2026-03-28
**Risk:** Low. All changes are reversible. None require container restarts except Step 1.
**Estimated downtime:** ~3 minutes (container restart in Step 1)

---

## Pre-Flight: Verify Baseline

Before making any changes, capture current state so you can compare after.

```bash
# Run on Spark — save baseline snapshot
ssh -i ~/.ssh/id_claude_code claude@spark.k4jda.net

# Memory baseline
free -h > /tmp/baseline_memory.txt
cat /proc/swaps >> /tmp/baseline_memory.txt

# vLLM metrics baseline
curl -s http://localhost:8000/metrics | grep -E "vllm:(kv_cache|num_preemptions|num_requests)" > /tmp/baseline_vllm.txt

# GPU baseline
nvidia-smi > /tmp/baseline_gpu.txt
```

---

## Step 1: Reduce LLM GPU Memory Utilization (0.72 -> 0.65)

**Why:** KV cache is at 0.54% utilization with zero preemptions across 422 requests. The current 0.72 allocates ~55 GB for KV cache. Reducing to 0.65 frees ~9 GB of host RAM, which should eliminate the 1.74 GB of swap usage entirely.

**Risk:** Low. Even at 0.65 you'll have ~46 GB of KV cache — over 80x the observed peak usage. Zero preemptions today means massive headroom.

**Rollback:** Change 0.65 back to 0.72 and restart the container.

### Procedure

```bash
# 1. Stop the LLM container
docker stop qwen35
docker rm qwen35

# 2. Restart with reduced memory utilization
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3.5-35B-A3B \
    --served-model-name qwen3.5-35b \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.65 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enforce-eager \
    --no-async-scheduling

# 3. Wait for health check (model loads in ~90 seconds)
echo "Waiting for qwen35 to start..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 5; echo "waiting..."; done
echo "qwen35 is healthy"

# 4. Verify the change took effect
curl -s http://localhost:8000/metrics | grep "vllm:cache_config_info"
# Confirm: gpu_memory_utilization="0.65"

# 5. Check memory improvement
free -h
cat /proc/swaps
# Expected: swap usage drops significantly or to zero
```

### Verification Checklist

- [ ] `curl http://localhost:8000/health` returns 200
- [ ] `vllm:cache_config_info` shows `gpu_memory_utilization="0.65"`
- [ ] `free -h` shows more available RAM than baseline (~4-5 GB improvement)
- [ ] Swap usage reduced (ideally < 500 MB)
- [ ] Run a test inference to confirm decode speed is unchanged:
  ```bash
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Count from 1 to 20"}],"max_tokens":100}' | python3 -m json.tool | head -20
  ```

---

## Step 2: Tune VM Parameters (swappiness, min_free_kbytes)

**Why:** Even after reducing GPU memory utilization, these settings prevent future swap pressure. `swappiness=1` tells Linux to avoid swapping almost entirely. `min_free_kbytes=262144` (256 MB) forces the kernel to keep a larger free memory reserve, preventing last-second swap spikes during model loading or burst traffic.

**Risk:** Very low. These are standard tuning parameters for memory-intensive workloads.

**Rollback:** `sysctl -w vm.swappiness=10` and `sysctl -w vm.min_free_kbytes=45155`

### Procedure

```bash
# Apply immediately (takes effect instantly, no restart needed)
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.min_free_kbytes=262144

# Make persistent across reboots
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf
echo "vm.min_free_kbytes=262144" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf

# Verify
sysctl vm.swappiness
# Expected: vm.swappiness = 1

sysctl vm.min_free_kbytes
# Expected: vm.min_free_kbytes = 262144
```

---

## Step 3: Tune TCP Buffer Sizes

**Why:** Default `rmem_max` and `wmem_max` are 208 KB. LLM API responses can be multi-megabyte (especially long streaming completions). Larger buffers improve network throughput for these responses, particularly when serving over WiFi where latency is higher.

**Risk:** None. Only affects max buffer size available to applications — doesn't consume memory until used.

**Rollback:** `sysctl -w net.core.rmem_max=212992` (restore defaults)

### Procedure

```bash
# Apply immediately
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sudo sysctl -w net.ipv4.tcp_wmem="4096 87380 16777216"

# Make persistent
echo "net.core.rmem_max=16777216" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf
echo "net.core.wmem_max=16777216" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf
echo "net.ipv4.tcp_rmem=4096 87380 16777216" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf
echo "net.ipv4.tcp_wmem=4096 87380 16777216" | sudo tee -a /etc/sysctl.d/99-spark-tuning.conf

# Verify
sysctl net.core.rmem_max net.core.wmem_max
```

---

## Step 4: Add vLLM Memory Metrics to Grafana Dashboard

**Why:** The GPU exporter can't read UMA memory stats (reports 0 for all memory fields). vLLM exposes its own memory metrics that actually work on GB10. Adding these to the dashboard gives you real visibility into GPU memory pressure and KV cache health.

**Risk:** None. Read-only dashboard change.

### Prometheus Already Scrapes These

The `vllm-llm` and `vllm-embed` scrape jobs already collect these metrics. You just need to add panels in Grafana.

### New Panels to Add

**Panel 1: KV Cache Blocks (in "LLM" row)**

```promql
# Gauge showing allocated vs max blocks
vllm:cache_config_info{job="vllm-llm"}
# Use label: num_gpu_blocks for display
```

**Panel 2: GPU Memory Utilization Trend (in "GPU & System Trends" row)**

```promql
# Process RSS memory per container (bytes -> GB)
process_resident_memory_bytes{job="vllm-llm"} / 1073741824
process_resident_memory_bytes{job="vllm-embed"} / 1073741824
```

**Panel 3: KV Cache Usage % Over Time (in "LLM" row)**

```promql
vllm:kv_cache_usage_perc{job="vllm-llm"}
```

This replaces the current gauge with a time-series so you can see KV cache usage patterns over time.

**Panel 4: Swap Usage (in "System Overview" row)**

```promql
# From node-exporter
(node_memory_SwapTotal_bytes{job="spark-node"} - node_memory_SwapFree_bytes{job="spark-node"}) / 1073741824
```

---

## Step 5: Plug In Ethernet (Optional, When Ready)

**Why:** Ethernet gives lower latency and higher throughput than WiFi for LLM API traffic. The Spark's WiFi metric is 600; ethernet will default to ~100, automatically becoming the preferred route.

**What happens:** NetworkManager activates "Wired connection 3" via DHCP. WiFi stays connected simultaneously. Default route switches to ethernet. Tailscale adapts automatically.

### Procedure

1. **Before plugging in:** Set a DHCP reservation on your router for MAC `fc:9d:05:13:27:f0` if you want a predictable IP
2. Plug in the ethernet cable
3. Verify both interfaces are up: `nmcli device status`
4. Verify routing: `ip route` — ethernet should have lower metric
5. Verify Tailscale: `tailscale status` — should show connected
6. Test API access from your workstation

### If Something Goes Wrong

WiFi remains connected. If ethernet causes issues:
```bash
# Disable ethernet auto-connect
nmcli connection modify "Wired connection 3" connection.autoconnect no
nmcli connection down "Wired connection 3"
# Back to WiFi-only
```

---

## Post-Implementation Verification

After completing all steps, run this comprehensive check:

```bash
# System health
free -h
cat /proc/swaps
nvidia-smi

# vLLM health
curl -sf http://localhost:8000/health && echo "LLM OK"
curl -sf http://localhost:8001/health && echo "Embed OK"
curl -sf http://localhost:8002/v1/ner -X POST \
  -H "Content-Type: application/json" \
  -d '{"texts":["test"],"labels":["PERSON"],"threshold":0.5}' && echo "GLiNER OK"

# Metrics flowing
curl -s http://localhost:8000/metrics | grep "vllm:kv_cache_usage_perc"
curl -s http://localhost:9100/metrics | grep "node_memory_SwapFree"

# VM tuning applied
sysctl vm.swappiness vm.min_free_kbytes net.core.rmem_max
```

### Success Criteria

- [ ] Swap usage < 500 MB (ideally 0)
- [ ] Available RAM > 8 GB (up from 4.6 GB)
- [ ] All three services healthy
- [ ] Inference speed unchanged (~23 tok/s decode)
- [ ] Grafana dashboard showing new panels
- [ ] sysctl values persisted in `/etc/sysctl.d/99-spark-tuning.conf`
