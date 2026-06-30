# NVIDIA DGX Spark — Full System Configuration

> **Purpose:** Complete configuration reference to rebuild this system from scratch after a wipe.
> **Last verified:** 2026-06-30 (§6.1, §11, §12 #7 corrected to Entry 073/076/080 live state — native pre-quant FP8, BF16 KV, FLASH_ATTN, batched-tokens 32768)

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
| Kernel | `6.17.0-1014-nvidia` (NVIDIA custom, PREEMPT_DYNAMIC) — updated from 1008 via firmware update 2026-04-30 |
| CUDA | 13.0 (driver 580.142) — rolled back from 590.48.01 due to UMA memory leak |
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

| Image | Tag | Size | Source | Used By |
|-------|-----|------|--------|---------|
| `vllm-cu132-test` | `latest` | ~25 GB | Custom build (v0.19.1rc1.dev219+cu132) | qwen35 (LLM) |
| `vllm/vllm-openai` | `cu130-known-good-20260306` | ~20 GB | Docker Hub (v0.17.0rc1) | qwen3-embed, bge-m3 |
| `gliner-ner` | `latest` | ~7.7 GB | Custom build from `gliner-server/Dockerfile` | gliner |
| `ce-service` | `latest` | ~15 GB | Custom build from `nvcr.io/nvidia/pytorch:24.12-py3` | ce-service |
| `chromadb/chroma` | `latest` | ~0.5 GB | Docker Hub | chromadb |
| `neo4j` | `5-community` | ~0.6 GB | Docker Hub | neo4j |
| `prom/node-exporter` | `latest` | ~30 MB | Docker Hub | node-exporter |

## 6. Container Configurations

### 6.1 qwen35 — LLM Inference (Port 8000)

The primary LLM serving the Qwen3.6-35B-A3B mixture-of-experts model with **native pre-quantized FP8** (adopted 2026-05-18, Entry 073 — replaced on-the-fly FP8) and MTP=2 speculative decoding.

**Key details:**
- **Image:** `vllm-cu132-test:latest` (vLLM v0.19.1rc1.dev219+cu132, custom build)
- **Model:** `Qwen/Qwen3.6-35B-A3B-FP8` (**native pre-quantized FP8**, block-scaled) — adopted 2026-05-18 (Entry 073; was `Qwen/Qwen3.6-35B-A3B` + on-the-fly `--quantization fp8` from 2026-04-23)
- **Served as:** `spark-llm`
- **Max context length:** 131072 tokens (128K — bumped from 32K on 2026-05-10, Entry 065. Model native max is 262144; 128K chosen for safe KV cache margin)
- **GPU memory utilization:** 0.70
- **KV cache:** **BF16 (auto — do NOT set `--kv-cache-dtype fp8`)**, Entry 073. 504,912 tokens @ 131K, max concurrency 3.85× (BF16 KV uses ~2× memory/token vs the old FP8 KV's 1,123,584 tokens; non-binding for current workload)
- **Speculative decoding:** MTP=2 (acceptance ~80%)
- **MoE backend:** TRITON (auto-selected); FlashInfer for MoE kernels via `VLLM_FLASHINFER_MOE_BACKEND=latency`
- **FP8 kernel:** native pre-quant block-scaled FP8 (do NOT add `--quantization fp8`)
- **Attention backend:** **FLASH_ATTN** (auto-selected on SM121 — verified Entry 076; NOT FlashInfer)
- **Async scheduling:** Enabled
- **max-num-batched-tokens:** 32768 (was 4096 pre-2026-05-18; bumped on the pre-quant switch, Entry 073)
- **Performance (kernel 1021, Entry 080 harness):** 73.1 tok/s c1, 186.7 c4, 406.9 c8, 730.5 c16 aggregate (prior on-the-fly: 59.9/166.2/373.8/564.0)
- **Startup time:** ~435s cold (fresh FP8 Triton JIT; first ~20 reqs warm to full speed) — Entry 073
- **API endpoint:** `http://192.168.10.32:8000/v1` (WiFi) or `http://192.168.10.33:8000/v1` (Ethernet)

```bash
docker run -d \
  --name qwen35 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  -v /home/claude/.cache/triton-cu132:/root/.triton \
  --entrypoint python3 \
  vllm-cu132-test:latest \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
    --served-model-name spark-llm \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.70 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-num-batched-tokens 32768 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**Notes:**
- `--entrypoint python3` required — cu132 image uses NVIDIA base entrypoint
- `--max-num-batched-tokens 32768` (was 4096 pre-2026-05-18; bumped on the pre-quant switch, Entry 073)
- **Pre-quant model: do NOT add `--quantization fp8` or `--kv-cache-dtype fp8`** — inappropriate for native FP8 weights and impose 5-15% throughput cost; KV cache runs BF16 (auto) by design (Entry 073)
- `--speculative-config` enables Multi-Token Prediction (MTP=2 speculative tokens, acceptance ~80%)
- Attention backend auto-selects **FLASH_ATTN** on SM121 (verified Entry 076) — do NOT force FlashInfer for attention; FlashInfer is used only for MoE via `VLLM_FLASHINFER_MOE_BACKEND=latency`
- Triton cache at `/home/claude/.cache/triton-cu132` — separate from cu130 cache (preserved for rollback)
- `enable_thinking: false` in API requests must be at request top level (`chat_template_kwargs`), NOT inside `extra_body`
- **Rollback to on-the-fly FP8 (pre-Entry-073):** `cp /home/claude/docker-compose.yml.pre-fp8prequant /home/claude/docker-compose.yml && docker compose stop qwen35 && docker compose up -d qwen35` (~6 min; the backup preserves the exact prior config)

### 6.2 qwen3-embed — Embedding Model (Port 8001)

Embedding model for vector search / RAG pipelines.

**Key details:**
- **Image:** `vllm/vllm-openai:cu130-known-good-20260306` (vLLM v0.17.0rc1)
- **Model:** `Qwen/Qwen3-Embedding-4B`
- **Served as:** `qwen3-embedding-4b`
- **GPU memory utilization:** 0.10
- **Embedding dimension:** 2560
- **Max sequence length:** 8192 tokens
- **API endpoint:** `http://<spark-lan-ip>:8001/v1`

```bash
docker run -d \
  --name qwen3-embed \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8001:8001 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-known-good-20260306 \
    Qwen/Qwen3-Embedding-4B \
    --served-model-name qwen3-embedding-4b \
    --runner pooling \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.10 \
    --max-model-len 8192 \
    --enforce-eager
```

**Notes:**
- `--enforce-eager` required — pooling models don't support cudagraphs
- `--runner pooling` sets vLLM to embedding/pooling mode

### 6.3 gliner — Named Entity Recognition (Port 8002)

Custom NER service using GLiNER for domain-specific entity extraction.

**Key details:**
- **Image:** `gliner-ner:latest` (custom build)
- **Model:** `urchade/gliner_large-v2.1` (~900M params, ~2 GB VRAM)
- **API endpoint:** `http://<spark-lan-ip>:8002/v1/ner`
- **Health check:** `GET /health`

```bash
docker run -d \
  --name gliner \
  --restart unless-stopped \
  --gpus all \
  -p 8002:8002 \
  -v /home/davistroy/gliner-env/hf-cache:/root/.cache/huggingface \
  -e GLINER_MODEL=urchade/gliner_large-v2.1 \
  -e GLINER_DEVICE=cuda \
  gliner-ner:latest
```

**Notes:**
- First inference call takes ~10-15s (CUDA JIT kernel compilation), subsequent calls ~5-15ms
- Uses a separate user-writable HF cache (`gliner-env/hf-cache`), NOT the root-owned default cache
- Falls back to CPU automatically if CUDA fails

### 6.4 bge-m3 — Embedding Model, Alternate (Port 8004)

Alternate embedding model with 1024-dim vectors (40% smaller FAISS index than qwen3-embed's 2560-dim).

**Key details:**
- **Image:** `vllm/vllm-openai:cu130-known-good-20260306`
- **Model:** `BAAI/bge-m3` (560M params)
- **Served as:** `bge-m3`
- **GPU memory utilization:** 0.05
- **Max sequence:** 8192 tokens
- **API endpoint:** `http://<spark-lan-ip>:8004/v1`
- **Added:** 2026-04-19 for kb-analysis A/B vs Qwen3-Embedding-4B

```bash
docker run -d \
  --name bge-m3 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8004:8004 \
  -v /home/davistroy/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-known-good-20260306 \
    --model BAAI/bge-m3 \
    --served-model-name bge-m3 \
    --runner pooling \
    --port 8004 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.05 \
    --max-model-len 8192 \
    --enforce-eager
```

### 6.5 ce-service — Cross-Encoder Reranker (Port 8005)

Cross-encoder for semantic similarity scoring / duplicate detection.

**Key details:**
- **Image:** `ce-service:latest` (custom build from `nvcr.io/nvidia/pytorch:24.12-py3`)
- **Model:** `cross-encoder/stsb-roberta-large` (~1.4 GB)
- **API endpoint:** `http://<spark-lan-ip>:8005/ce/score` (POST), `/ce/health` (GET)
- **Inference:** ~14ms for 2 pairs on GPU
- **Max batch:** 512 pairs
- **Added:** 2026-04-19

```bash
docker run -d \
  --name ce-service \
  --restart unless-stopped \
  --gpus all \
  -p 8005:8005 \
  ce-service:latest
```

**Note:** Requires `transformers<4.49` pin — NVIDIA 24.12 ships PyTorch 2.5 incompatible with 4.49+.

### 6.6 Supporting Services

```bash
# ChromaDB — vector database (Port 8003)
docker run -d \
  --name chromadb \
  --restart unless-stopped \
  -p 8003:8000 \
  -v chromadb-data:/chroma/chroma \
  chromadb/chroma:latest

# Neo4j — graph database (Ports 7474, 7687)
docker run -d \
  --name neo4j \
  --restart unless-stopped \
  -p 7474:7474 -p 7687:7687 \
  -v neo4j-data:/data \
  neo4j:5-community

# Node Exporter — Prometheus metrics
docker run -d \
  --name node-exporter \
  --restart unless-stopped \
  --net host \
  prom/node-exporter:latest
```

### 6.7 LiteLLM Proxy (Not Running)

LiteLLM proxy configuration exists at `/home/<user>/litellm/config.yaml` but is not currently deployed.

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

With the current 8-container configuration (5 use GPU):

| Component | GPU MiB | Notes |
|-----------|---------|-------|
| qwen35 (Qwen3.6-35B-A3B-FP8 pre-quant + MTP, BF16 KV) | ~84,600 | 0.70 × 121.6 GiB (weights + BF16 KV cache + MTP drafter); live-measured 84,565 MiB 2026-06-30 |
| bge-m3 (BAAI/bge-m3) | ~12,200 | 0.05 × 121.6 GiB |
| gliner (gliner_large-v2.1) | ~2,000 | On CUDA |
| qwen3-embed (Qwen3-Embedding-4B) | ~1,700 | 0.10 × 121.6 GiB |
| ce-service (stsb-roberta-large) | ~1,500 | On CUDA |
| **Total allocated** | **~104,900 MiB** (~102 GiB) | |
| **Remaining for OS/buffers** | **~19.5 GiB** | |

## 12. Known Gotchas & Operational Rules

These are hard-won lessons. Do not ignore them.

1. **HF cache ownership:** `/home/<user>/.cache/huggingface` is root-owned (from vLLM docker volume mounts). Non-root processes must use a separate cache dir or set `HF_HOME` to a user-writable location. The GLiNER container uses `/home/<user>/gliner-env/hf-cache` for this reason.

2. **PyTorch CUDA on GB10:** Only cu130 nightly works. cu126 lacks sm_121 support entirely. cu128 detects the GPU but NVRTC JIT fails at inference.

3. **Docker GPU access:** Use `--gpus all` (not `--runtime nvidia`). The nvidia runtime is not configured on this system; GPU access is via device requests.

4. **vLLM GPU memory coordination:** Two vLLM containers cannot share GPU memory coordination. Each needs an explicit `--gpu-memory-utilization` value, and the sum must leave headroom for GLiNER and OS.

5. **FlashInfer MoE backend:** Set `VLLM_FLASHINFER_MOE_BACKEND=latency` if using environment-based configuration. The throughput backend has sm_121 kernel issues.

6. **NVFP4 quantization:** Community reports NVFP4 working on GB10 via nightly cu130 with flashinfer_cutlass backend (as of Apr 2026). Previously believed broken on SM 12.1.

7. **Pre-quantized FP8 (current production):** `Qwen/Qwen3.6-35B-A3B-FP8` native pre-quant is the production model since 2026-05-18 (Entry 073) — the old v0.19.0 hang was version-specific and does not occur on the cu132 build. Do NOT add `--quantization fp8` or `--kv-cache-dtype fp8` (KV is BF16/auto). The earlier `Qwen3.5-35B-A3B-FP8` hang was v0.19.0-only.

8. **Startup order is critical:** See Section 7. Simultaneous container startup causes CUDA memory allocation races and transient hangs.

9. **No passwordless sudo:** OS-level changes (sysctl, systemctl) require interactive sudo password entry.

## 13. API Quick Reference

```bash
# LLM — Chat completions
curl http://<spark-lan-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"spark-llm","messages":[{"role":"user","content":"Hello"}]}'

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

## 14. Firmware Update Procedure

Firmware updates (EC, UEFI, USB-PD) are applied via the DGX Spark web dashboard. **Critical: firmware updates can change the kernel version**, and the matching NVIDIA driver module package is NOT auto-installed.

**Post-firmware-update recovery:**
```bash
# 1. Check if nvidia-smi works after reboot
nvidia-smi
# 2. If exit code 9: install matching module package
sudo apt install linux-modules-nvidia-580-open-$(uname -r)
# 3. Load module (no reboot needed)
sudo modprobe nvidia
sudo systemctl restart nvidia-persistenced
# 4. Restart containers in order (Section 7)
```

Prebuilt module packages are pre-signed for Secure Boot — no MOK enrollment risk. See LAB_NOTEBOOK Entry 050 for incident details (2026-04-30, kernel 1008→1014).

## 15. Disaster Recovery Checklist

To rebuild from scratch:

1. **Install Ubuntu 24.04** with NVIDIA kernel (`6.17.0-1014-nvidia` or later)
2. **Install Docker** (29.x+), ensure `--gpus all` works
3. **Install NVIDIA driver modules:** `apt install linux-modules-nvidia-580-open-$(uname -r)`
4. **Install Tailscale**, join `<tailnet>`, set hostname to `spark`
5. **Create user** `<user>`, add SSH ed25519 public key
6. **Create directories:**
   ```bash
   mkdir -p ~/spark-vllm-docker ~/gliner-server ~/gliner-env/hf-cache ~/hf_cache/hub ~/litellm
   ```
7. **Build or load `vllm-cu132-test:latest`** (custom cu132+MTP image from `spark-vllm-docker`)
8. **Download models** into `~/.cache/huggingface/`:
   - `Qwen/Qwen3.6-35B-A3B`
   - `Qwen/Qwen3-Embedding-4B`
   - `BAAI/bge-m3`
   - `urchade/gliner_large-v2.1` (into `~/gliner-env/hf-cache/`)
   - `cross-encoder/stsb-roberta-large`
9. **Pull embedding image:** `docker pull vllm/vllm-openai:cu130-known-good-20260306`
10. **Build custom images:** `gliner-ner:latest` and `ce-service:latest`
11. **Apply sysctl tuning:** `vm.swappiness=1`, `vm.min_free_kbytes=262144` in `/etc/sysctl.d/99-spark-tuning.conf`
12. **Start containers in order** (Section 7), verifying health between each
13. **Verify all endpoints** (Section 13)
