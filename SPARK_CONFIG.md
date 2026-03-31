# NVIDIA DGX Spark — Full System Configuration

> **Purpose:** Complete configuration reference to rebuild this system from scratch after a wipe.
> **Last verified:** 2026-03-19

---

## 1. Hardware & OS

| Property | Value |
|----------|-------|
| Hostname | `spark` |
| Device | NVIDIA DGX Spark (Jetson platform) |
| CPU Architecture | aarch64 (ARM64) |
| GPU | NVIDIA GB10 Blackwell, compute capability 12.1 (sm_121) |
| Memory | 128 GB unified (CPU/GPU shared), ~121.6 GiB visible to CUDA |
| Storage | 3.6 TB NVMe (`/dev/nvme0n1p2`), single partition for `/` |
| Swap | 15 GB configured |
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | `6.17.0-1008-nvidia` (NVIDIA custom, PREEMPT_DYNAMIC) |
| CUDA | 13.0+ (driver 590.48.01) |
| Docker | 29.1.3 |

## 2. Network

| Interface | Address | Notes |
|-----------|---------|-------|
| Wi-Fi (`wlP9s9`) | `<spark-lan-ip>/24` | Local LAN |
| Tailscale | `<spark-tailscale-ip>` | Tailnet: `<tailnet>` |
| Tailscale DNS | `<spark-tailscale-dns>` | Also reachable as `<spark-host>` |
| Docker bridge | `172.17.0.1/16` | Default bridge network |

### Tailscale setup
Tailscale is installed and running. The machine is registered as `spark` in the `<tailnet>` tailnet.

## 3. User Account

| Property | Value |
|----------|-------|
| Username | `<user>` |
| SSH key | ed25519, public key in `~/.ssh/authorized_keys` |
| Sudo | Requires password (no passwordless sudo) |
| SSH access | `ssh <user>@<spark-lan-ip>` |

## 4. Directory Layout

```
/home/<user>/
├── spark-vllm-docker/       # vLLM custom build system (Dockerfile, launch scripts, patches)
│   ├── Dockerfile            # Multi-stage build: FlashInfer + vLLM from source for sm_121
│   ├── launch-cluster.sh     # Multi-node vLLM launcher (used in solo mode here)
│   ├── run-recipe.sh         # Recipe runner wrapper
│   ├── wheels/               # Built wheels (FlashInfer, vLLM)
│   ├── mods/                 # Patches and modifications
│   └── recipes/              # Model serving recipes (YAML configs)
├── gliner-server/            # GLiNER NER service
│   ├── Dockerfile            # CUDA 13.0.1 + PyTorch nightly cu130 + GLiNER
│   └── server.py             # FastAPI server
├── gliner-env/
│   └── hf-cache/             # HuggingFace cache for GLiNER (user-writable)
├── hf_cache/
│   └── hub/                  # HuggingFace model cache for vLLM containers
├── litellm/
│   └── config.yaml           # LiteLLM proxy config
└── .cache/
    └── huggingface/          # Default HF cache (root-owned — see warnings below)
```

## 5. Docker Images

| Image | Tag | Size | Source |
|-------|-----|------|--------|
| `vllm-node` | `latest` | 25.5 GB | Custom build from `spark-vllm-docker/Dockerfile` |
| `vllm/vllm-openai` | `cu130-nightly` | 20.3 GB | Docker Hub (used for embedding model) |
| `gliner-ner` | `latest` | 7.71 GB | Custom build from `gliner-server/Dockerfile` |
| `ghcr.io/berriai/litellm` | `main-latest` | 1.87 GB | GitHub Container Registry |
| `nvidia/cuda` | `13.0.1-runtime-ubuntu24.04` | 2.55 GB | NVIDIA (GLiNER base) |

## 6. Container Configurations

### 6.1 qwen35 — LLM Inference (Port 8000)

The primary LLM serving the Qwen3.5-35B mixture-of-experts model with FP8 quantization.

**Key details:**
- **Image:** `vllm-node:latest` (custom-built vLLM with FlashInfer for sm_121)
- **Model:** `Qwen/Qwen3.5-35B-A3B-FP8` (pre-quantized FP8 checkpoint)
- **Served as:** `qwen3.5-35b`
- **Network mode:** `host` (binds directly to host port 8000, no port mapping needed)
- **Max context length:** 8192 tokens
- **GPU memory utilization:** 0.75
- **API endpoint:** `http://<spark-lan-ip>:8000/v1`

```bash
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 16g \
  -v /home/<user>/hf_cache/hub:/root/.cache/huggingface/hub:ro \
  vllm-node:latest \
  vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name qwen3.5-35b \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --max-num-batched-tokens 4192 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --structured-outputs-config '{"backend":"xgrammar"}' \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --load-format fastsafetensors \
    --disable-log-stats
```

**Notes:**
- Uses host networking (no `-p` port mapping)
- HF cache mounted read-only
- `--enforce-eager` NOT set in current config (was previously needed; may have been resolved in the custom vLLM build)
- Model load time: ~90 seconds
- Pre-quantized FP8 model used instead of on-the-fly quantization

### 6.2 qwen3-embed — Embedding Model (Port 8001)

Embedding model for vector search / RAG pipelines.

**Key details:**
- **Image:** `vllm/vllm-openai:cu130-nightly`
- **Model:** `Qwen/Qwen3-Embedding-4B`
- **Served as:** `qwen3-embedding-4b`
- **Network mode:** `bridge` (standard port mapping)
- **GPU memory utilization:** 0.08
- **Embedding dimension:** 2560
- **Max sequence length:** 40960 tokens (vLLM default for this model)
- **API endpoint:** `http://<spark-lan-ip>:8001/v1`

```bash
docker run -d \
  --name qwen3-embed \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8001:8001 \
  -v /home/<user>/.cache/huggingface:/root/.cache/huggingface:ro \
  vllm/vllm-openai:cu130-nightly \
    --model Qwen/Qwen3-Embedding-4B \
    --served-model-name qwen3-embedding-4b \
    --runner pooling \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.08 \
    --enforce-eager
```

**Notes:**
- `--enforce-eager` required — pooling models don't support cudagraphs
- `--runner pooling` sets vLLM to embedding/pooling mode
- Uses the default HF cache (root-owned, mounted read-only)

### 6.3 gliner — Named Entity Recognition (Port 8002)

Custom NER service using GLiNER for domain-specific entity extraction.

**Key details:**
- **Image:** `gliner-ner:latest` (custom build)
- **Model:** `urchade/gliner_large-v2.1` (~900M params, ~2 GB VRAM)
- **Network mode:** `bridge`
- **API endpoint:** `http://<spark-lan-ip>:8002/v1/ner`
- **Health check:** `GET /health`

```bash
docker run -d \
  --name gliner \
  --restart unless-stopped \
  --gpus all \
  -p 8002:8002 \
  -v /home/<user>/gliner-env/hf-cache:/root/.cache/huggingface \
  -e GLINER_MODEL=urchade/gliner_large-v2.1 \
  -e GLINER_DEVICE=cuda \
  gliner-ner:latest
```

**Notes:**
- First inference call takes ~10-15s (CUDA JIT kernel compilation), subsequent calls ~5-15ms
- Uses a separate user-writable HF cache (`gliner-env/hf-cache`), NOT the root-owned default cache
- Falls back to CPU automatically if CUDA fails

### 6.4 LiteLLM Proxy (Optional)

LiteLLM proxy configuration exists at `/home/<user>/litellm/config.yaml` but may not be running as a container currently. Config routes to the local qwen35 instance:

```yaml
model_list:
  - model_name: qwen3.5-35b
    litellm_params:
      model: openai/qwen3.5-35b
      api_base: http://localhost:8000/v1
      api_key: <your-api-key>

general_settings:
  master_key: <your-master-key>

litellm_settings:
  drop_params: true
  set_verbose: false
```

## 7. Container Startup Order

**This is critical.** Simultaneous startup causes CUDA memory allocation races.

```
1. Start qwen35        → wait for GET /health returns 200
2. Start qwen3-embed   → wait for GET /health returns 200
3. Start gliner        → wait for GET /health returns 200
```

The `--restart unless-stopped` policy does NOT enforce ordering on reboot. After a system reboot, you must manually stop all containers and restart them in order, or use a script.

**Health check commands:**
```bash
curl -s http://localhost:8000/health     # qwen35
curl -s http://localhost:8001/health     # qwen3-embed
curl -s http://localhost:8002/health     # gliner
```

## 8. Building the Custom vLLM Image

The `vllm-node:latest` image is built from source to support the GB10 Blackwell GPU (sm_121). Standard vLLM Docker images do not include sm_121 kernels.

**Build system:** `/home/<user>/spark-vllm-docker/`

The Dockerfile is a multi-stage build:
1. **Base:** `nvcr.io/nvidia/pytorch:26.01-py3`
2. **Stage 2:** Builds FlashInfer from source with `FLASHINFER_CUDA_ARCH_LIST=12.1a`
3. **Stage 4:** Builds vLLM from source with `TORCH_CUDA_ARCH_LIST=12.1a`, applies patches for Hopper-specific code
4. **Stage 6 (Runner):** Installs built wheels into a clean PyTorch base, adds Ray, fastsafetensors, nvidia-nvshmem

**To rebuild:**
```bash
cd /home/<user>/spark-vllm-docker
# Check build-and-copy.sh for the standard build command
docker build -t vllm-node:latest .
```

**Key build ARGs:**
- `TORCH_CUDA_ARCH_LIST=12.1a`
- `FLASHINFER_CUDA_ARCH_LIST=12.1a`
- `BUILD_JOBS=16`
- `VLLM_REF=main` (or specific tag/SHA)
- `FLASHINFER_REF=main`
- `VLLM_PRS=""` (optional PR diffs to apply)

**Patches applied at build time:**
- Reverts PR #34758 and #34302 (unguarded Hopper-only code that breaks on sm_121)
- FlashInfer cache patch to avoid re-downloading cubins

## 9. Building the GLiNER Image

```bash
cd /home/<user>/gliner-server
docker build -t gliner-ner:latest .
```

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    pip install --no-cache-dir gliner fastapi uvicorn

COPY server.py .

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
```

**Critical:** Must use PyTorch nightly with `cu130` index. Standard PyTorch cu126 only supports up to sm_90. cu128 detects the GPU but fails at inference with `invalid value for --gpu-architecture`.

## 10. GLiNER Server Code

The file `/home/<user>/gliner-server/server.py` implements:

- **POST `/v1/ner`** — Entity extraction
  - Input: `{"texts": [...], "labels": [...], "threshold": 0.5, "flat_ner": true}`
  - Default labels: PERSON, ORGANIZATION, EQUIPMENT, SOFTWARE, UI_ELEMENT, MENU_ITEM, PRODUCT, LOCATION, DOCUMENT, PROCESS, ERROR_CODE, ROLE, SETTING, FOOD_ITEM
  - Returns: `{"entities": [[{"text", "label", "score", "start", "end"}, ...], ...]}`

- **GET `/health`** — Returns model name, device, and CUDA availability

- Startup: Loads model, attempts CUDA with smoke test, falls back to CPU on failure

## 11. GPU Memory Budget

With the current 3-container configuration:

| Component | GPU Memory | Notes |
|-----------|-----------|-------|
| qwen35 (Qwen3.5-35B-A3B FP8) | ~91 GB | 0.75 × 121.6 GiB (weights + KV cache) |
| qwen3-embed (Qwen3-Embedding-4B) | ~10 GB | 0.08 × 121.6 GiB |
| gliner (gliner_large-v2.1) | ~2 GB | On CUDA |
| **Total allocated** | **~103 GB** | |
| **Remaining for OS/buffers** | **~19 GB** | |

## 12. Known Gotchas & Operational Rules

These are hard-won lessons. Do not ignore them.

1. **HF cache ownership:** `/home/<user>/.cache/huggingface` is root-owned (from vLLM docker volume mounts). Non-root processes must use a separate cache dir or set `HF_HOME` to a user-writable location. The GLiNER container uses `/home/<user>/gliner-env/hf-cache` for this reason.

2. **PyTorch CUDA on GB10:** Only cu130 nightly works. cu126 lacks sm_121 support entirely. cu128 detects the GPU but NVRTC JIT fails at inference.

3. **Docker GPU access:** Use `--gpus all` (not `--runtime nvidia`). The nvidia runtime is not configured on this system; GPU access is via device requests.

4. **vLLM GPU memory coordination:** Two vLLM containers cannot share GPU memory coordination. Each needs an explicit `--gpu-memory-utilization` value, and the sum must leave headroom for GLiNER and OS.

5. **FlashInfer MoE backend:** Set `VLLM_FLASHINFER_MOE_BACKEND=latency` if using environment-based configuration. The throughput backend has sm_121 kernel issues.

6. **NVFP4 is broken on GB10:** SM 12.1 lacks the hardware instruction support for NVFP4 quantization.

7. **Pre-quantized FP8 vs on-the-fly:** The pre-quantized checkpoint (`Qwen3.5-35B-A3B-FP8`) is currently in use. Previously, on-the-fly FP8 quantization from the BF16 model worked but the pre-quantized checkpoint was getting stuck during weight loading — this may have been resolved in newer vLLM builds.

8. **Startup order is critical:** See Section 7. Simultaneous container startup causes CUDA memory allocation races and transient hangs.

9. **No passwordless sudo:** OS-level changes (sysctl, systemctl) require interactive sudo password entry.

## 13. API Quick Reference

```bash
# LLM — Chat completions
curl http://<spark-lan-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"Hello"}]}'

# LLM — List models
curl http://<spark-lan-ip>:8000/v1/models

# Embeddings
curl http://<spark-lan-ip>:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding-4b","input":"Hello world"}'

# NER
curl -X POST http://<spark-lan-ip>:8002/v1/ner \
  -H "Content-Type: application/json" \
  -d '{"texts":["John Smith reported a fryer error code E24"],"labels":["PERSON","EQUIPMENT","ERROR_CODE"],"threshold":0.5}'

# Health checks
curl http://<spark-lan-ip>:8000/health
curl http://<spark-lan-ip>:8001/health
curl http://<spark-lan-ip>:8002/health
```

## 14. Disaster Recovery Checklist

To rebuild from scratch:

1. **Install Ubuntu 24.04** with NVIDIA kernel (`6.17.0-1008-nvidia` or later)
2. **Install Docker** (29.x+), ensure `--gpus all` works
3. **Install Tailscale**, join `<tailnet>`, set hostname to `spark`
4. **Create user** `<user>`, add SSH ed25519 public key
5. **Create directories:**
   ```bash
   mkdir -p ~/spark-vllm-docker ~/gliner-server ~/gliner-env/hf-cache ~/hf_cache/hub ~/litellm
   ```
6. **Clone/copy `spark-vllm-docker`** build system, build `vllm-node:latest`
7. **Download models** into `~/hf_cache/hub/`:
   - `Qwen/Qwen3.5-35B-A3B-FP8`
   - `Qwen/Qwen3-Embedding-4B`
   - `urchade/gliner_large-v2.1` (into `~/gliner-env/hf-cache/`)
8. **Pull embedding image:** `docker pull vllm/vllm-openai:cu130-nightly`
9. **Build GLiNER image:** copy `server.py` + `Dockerfile` to `~/gliner-server/`, run `docker build -t gliner-ner:latest .`
10. **Start containers in order** (Section 7), verifying health between each
11. **Verify all endpoints** (Section 13)
