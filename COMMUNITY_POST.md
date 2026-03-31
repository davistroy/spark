# DGX Spark: 13 → 49 tok/s with Qwen3.5-35B — Native SM121 Kernel Build Guide

**TL;DR:** The DGX Spark (GB10, SM121) ships with vLLM builds that lack native Blackwell kernels due to a CMake architecture guard bug. A multi-stage Docker build that compiles SM121 kernels from the v0.17.0rc1 source and injects only the .so files into the stock image takes throughput from 13.3 to 48.6 tok/s — a 3.65x improvement with Qwen3.5-35B-A3B (FP8). No model changes, no driver changes, no hardware mods.

---

## The Problem

The DGX Spark achieves ~13 tok/s with Qwen3.5-35B-A3B using the stock `vllm/vllm-openai:cu130-nightly` image (v0.17.0rc1, built March 6, 2026). Community members report 50 tok/s on the same hardware. The gap is not hardware — it's missing native kernels.

vLLM's CMake build system uses `cuda_archs_loose_intersection` with a `"12.0f"` (family) pattern to decide which Blackwell kernels to compile. This pattern is meant to match all SM12x architectures, but the pre-built Docker images don't compile for SM121 at all — the stock image has zero Blackwell cubins.

## Why You Can't Just Rebuild

We went through 8 build iterations to find the working approach. Here's what fails and why:

**Attempt: `pip install --no-build-isolation .` with `TORCH_CUDA_ARCH_LIST="12.1"`**
Fails on NVFP4 kernels: `ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'`. The SM121 (GB10) lacks the microscaling instructions that SM120 (datacenter Blackwell) has. The CMake guard incorrectly includes SM121 in NVFP4 compilation.

**Attempt: Patch the NVFP4 guard and rebuild the full vLLM package**
Compiles successfully, but breaks Qwen3.5 model loading. `pip install .` resolves the entire dependency tree, potentially changing the `transformers` library version. The stock image has a carefully curated transformers that recognizes `qwen3_5_moe` — rebuilding loses this.

**Attempt: Use community images (hellohal2064)**
Crash loop — built for a different model (Qwen3-Next-80B), different CUDA version (13.1 vs 13.0), incompatible entrypoint. Not a drop-in.

## The Solution: Multi-Stage .so Injection

The key insight: **only the compiled C extensions need to change.** The stock image's Python code, model support, and dependency versions are correct. We just need native Blackwell cubins in the .so files.

### Step 1: Get the right source

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# Checkout the EXACT commit matching your base image
git checkout e68de8adc  # v0.17.0rc1
```

Verify your base image version: `docker run --rm --entrypoint python3 YOUR_IMAGE -c "import vllm; print(vllm.__version__)"`

### Step 2: Patch NVFP4 for SM121

One line change in `CMakeLists.txt`. Find the NVFP4 section (~line 651):

```cmake
# BEFORE:
cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")
# AFTER:
cuda_archs_loose_intersection(FP4_ARCHS "12.0a" "${CUDA_ARCHS}")
```

Do the same for both the `VERSION_GREATER_EQUAL 13.0` and `else()` branches. This excludes SM121 from NVFP4 compilation (SM121 hardware doesn't support it) while keeping all other Blackwell kernels (scaled_mm, MoE, MLA, attention).

If your vLLM version also has `cmake/external_projects/qutlass.cmake`, apply the same change there.

### Step 3: Multi-stage Dockerfile

```dockerfile
# Stage 1: Compile SM121 kernels
FROM vllm/vllm-openai:cu130-nightly AS builder

RUN apt-get update && apt-get install -y git ninja-build && \
    rm -rf /var/lib/apt/lists/* && pip install "cmake>=3.26"
RUN ln -sf /usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so.13 \
    /usr/local/cuda/lib64/libnvrtc.so

COPY . /tmp/vllm-source/
WORKDIR /tmp/vllm-source

ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV MAX_JOBS=4
ENV VLLM_TARGET_DEVICE=cuda
ENV CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
ENV LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"

SHELL ["/bin/bash", "-c"]
RUN set -o pipefail && pip install --no-build-isolation . 2>&1 | tee /tmp/build.log

# Stage 2: Inject ONLY the .so files into the pristine stock image
FROM vllm/vllm-openai:cu130-nightly

COPY --from=builder /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so \
    /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so
COPY --from=builder /usr/local/lib/python3.12/dist-packages/vllm/_moe_C.abi3.so \
    /usr/local/lib/python3.12/dist-packages/vllm/_moe_C.abi3.so
```

### Step 4: Build (~90 min on ARM64)

```bash
docker build -f Dockerfile.sm121-inject -t vllm-custom:sm121-inject .
```

### Step 5: Verify

```bash
# Check for Blackwell cubins
docker run --rm --entrypoint bash vllm-custom:sm121-inject -c \
  'cuobjdump -lelf /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so | grep -c sm_120'
# Should show 50+ cubins (ours: 52)

# Verify version matches stock
docker run --rm --entrypoint python3 vllm-custom:sm121-inject -c \
  'import vllm; print(vllm.__version__)'
# Should match your base image version exactly
```

### Step 6: Launch

```bash
docker run -d --name qwen35 --gpus all --ipc host --shm-size 64gb \
  -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -v /path/to/huggingface/cache:/root/.cache/huggingface \
  vllm-custom:sm121-inject \
  Qwen/Qwen3.5-35B-A3B \
    --served-model-name qwen3.5-35b \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.65 \
    --quantization fp8 --kv-cache-dtype fp8
```

## Results

| Config | Single-Request | Improvement |
|--------|---------------|-------------|
| Stock image (no native kernels) | 13.3 tok/s | — |
| SM121-inject (52 Blackwell cubins) | **48.6 tok/s** | **3.65x** |

Consistent across 3 runs (48.7, 48.6, 48.6). Qwen3.5-35B-A3B, FP8, max_tokens=600.

## Key Learnings

1. **SM121 ≠ SM120 for microscaling.** SM121 (GB10/DGX Spark) lacks the `cvt.e2m1x2` instruction that SM120 (datacenter Blackwell) has. NVFP4 kernels will never compile for SM121 — this is a hardware limitation, not a software bug.

2. **Don't rebuild the full package.** `pip install .` resolves the entire dependency tree and can break model support. The multi-stage .so injection preserves the stock image's Python code and dependencies while replacing only the compiled kernels.

3. **`VLLM_TEST_FORCE_FP8_MARLIN=1` is essential.** Without this, vLLM may select a CUTLASS scaled_mm path that produces NaN logits on SM121. Marlin FP8 is stable and fast.

4. **The cubins say sm_120, not sm_121.** Despite setting `TORCH_CUDA_ARCH_LIST="12.1"`, the v0.17.0rc1 CMake produces sm_120 cubins. This is fine — SM121 runs sm_120 code natively via forward compatibility. All community builds (hellohal2064, namake-taro, sesmanovic) also target sm_120.

5. **`CPATH` and `LIBRARY_PATH` are required.** The stock vLLM Docker image uses pip-installed CUDA packages. Headers like `cusparse.h` are at `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include`, not the standard CUDA path.

## Environment

- NVIDIA DGX Spark (GB10, SM121, 128GB LPDDR5x unified memory)
- Driver 580.142, CUDA 13.0
- Base image: `vllm/vllm-openai:cu130-nightly` (v0.17.0rc1, March 6 2026)
- Model: Qwen/Qwen3.5-35B-A3B (FP8 on-the-fly quantization)

## What's Next

- Testing SM120-targeted build (`TORCH_CUDA_ARCH_LIST="12.0"`, zero patches, all kernels compile) for comparison
- Concurrent throughput benchmarks (8-18 simultaneous requests)
- Investigating the `marlin_utils_fp8.py` runtime check that still warns about missing native FP8 (the kernels are there, the Python check doesn't know it)
- Async scheduling and prefix caching on the new build

Happy to answer questions. The full lab notebook with every failed attempt and fix is available if anyone wants the gory details.
