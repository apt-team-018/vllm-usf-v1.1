# Omega17 vLLM Docker Build Guide

This document contains all successful commands and configurations for building and deploying vLLM with Omega17 model support on H100 GPU.

## Table of Contents
- [Docker Build Commands](#docker-build-commands)
- [Docker Hub Upload](#docker-hub-upload)
- [Running on H100](#running-on-h100)
- [Key Configuration Details](#key-configuration-details)
- [Architecture Overview](#architecture-overview)

---

## Docker Build Commands

### Fresh Build (No Cache)
Use this for building from scratch on a new server:

```bash
docker build \
  --platform linux/amd64 \
  --target vllm-openai \
  --no-cache \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg max_jobs=<NEW_SERVER_CORES> \
  --build-arg nvcc_threads=<HALF_OF_CORES> \
  --build-arg torch_cuda_arch_list='9.0 10.0' \
  --build-arg USE_SCCACHE=1 \
  --build-arg SCCACHE_S3_NO_CREDENTIALS=1 \
  -f docker/Dockerfile \
  -t vllm-openai:test-2.3 \
  .
```

**Build Arguments:**
- `max_jobs`: Number of parallel compilation jobs (use CPU core count)
- `nvcc_threads`: NVCC compilation threads (use half of CPU cores)
- `torch_cuda_arch_list`: '9.0 10.0' for H100 and future GPUs
- `USE_SCCACHE=1`: Enable ccache for faster rebuilds
- `RUN_WHEEL_CHECK=false`: Skip wheel validation for faster builds

**Estimated Build Time:** ~15 hours (depends on server specs)

---

### Incremental Build (With Cache)
Use this when you have a previous build and only need to apply code changes:

```bash
docker build \
  --platform linux/amd64 \
  --target vllm-openai \
  --cache-from vllm-openai:test-2.2 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg max_jobs=92 \
  --build-arg nvcc_threads=48 \
  --build-arg torch_cuda_arch_list='9.0 10.0' \
  --build-arg USE_SCCACHE=1 \
  --build-arg SCCACHE_S3_NO_CREDENTIALS=1 \
  -f docker/Dockerfile \
  -t vllm-openai:test-2.3 \
  .
```

**Note:** Replace `--cache-from vllm-openai:test-2.2` with your previous successful build tag.

**Estimated Build Time:** ~1-2 hours (only rebuilds changed layers)

---

## Docker Hub Upload

### Tag and Push to Docker Hub

```bash
# Tag the image with your Docker Hub username
docker tag vllm-openai:test-2.3 arpitsh018/vllm-openai:test-2.3

# Login to Docker Hub (you'll be prompted for password)
docker login

# Push the image
docker push arpitsh018/vllm-openai:test-2.3
```

### Optional: Push as Latest

```bash
# Tag as latest
docker tag vllm-openai:test-2.3 arpitsh018/vllm-openai:latest

# Push latest tag
docker push arpitsh018/vllm-openai:latest
```

**Note:** Image size is ~20-30GB. Ensure good internet connection for upload.

---

## Running on H100

### Basic Run Command

```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=your_token \
  arpitsh018/vllm-openai:test-2.3 \
  --served-model-name omega \
  --model arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32000
```

### Run with Custom Configuration

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=your_token \
  --shm-size 16g \
  arpitsh018/vllm-openai:test-2.3 \
  --served-model-name omega \
  --model arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32000 \
  --max-num-seqs 256 \
  --tensor-parallel-size 1
```

### Pull from Docker Hub

```bash
# Pull the image from Docker Hub
docker pull arpitsh018/vllm-openai:test-2.3

# Or pull latest
docker pull arpitsh018/vllm-openai:latest
```

---

## Key Configuration Details

### Model Information
- **Model ID:** `arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020`
- **Model Type:** `omega17_vl_exp`
- **Architecture:** `Omega17VLExpForConditionalGeneration`

### Custom Transformers
- **Repository:** https://github.com/apt-team-018/transformers-usf-exp.git
- **Purpose:** Provides native Omega17 model support
- **Installation:** Installed during Docker build with pre-release dependencies

### Docker Images
- **Base:** `vllm-openai:test-2.3`
- **Registry:** Docker Hub (`arpitsh018/vllm-openai:test-2.3`)

### GPU Configuration
- **Target GPU:** NVIDIA H100
- **CUDA Architecture:** 9.0 (H100), 10.0 (future proofing)
- **Memory Utilization:** 0.90 (90% GPU memory)

### vLLM Server Configuration
- **Port:** 8000
- **dtype:** bfloat16
- **Max Model Length:** 32000 tokens
- **Host:** 0.0.0.0 (accepts external connections)

---

## Architecture Overview

### Omega17 Config Integration

#### 1. Config Import (`vllm/transformers_utils/configs/__init__.py`)
```python
from transformers.models.omega17_vl.configuration_omega17_vl import (
    Omega17VLConfig,
    Omega17VLVisionConfig,
)
from transformers.models.omega17_vl_exp.configuration_omega17_vl_exp import (
    Omega17VLExpConfig,
    Omega17VLExpTextConfig,
    Omega17VLExpVisionConfig,
)
```

All configs are imported from the custom transformers package and exported in `__all__`.

#### 2. Config Registry (`vllm/transformers_utils/config.py`)
No explicit registry mapping needed for Omega17 models. vLLM falls back to `AutoConfig.from_pretrained()` which uses the custom transformers package.

#### 3. Model Registration (`vllm/model_executor/models/registry.py`)
```python
_MULTIMODAL_MODELS = {
    "Omega17VLExpForConditionalGeneration": ("omega17_vl_exp", "Omega17VLExpForConditionalGeneration"),
}
```

#### 4. Docker Build Verification
The Dockerfile includes a verification step to ensure all Omega17 configs are importable:

```dockerfile
RUN python3 -c "\
from transformers.models.omega17_vl.configuration_omega17_vl import Omega17VLConfig, Omega17VLVisionConfig; \
from transformers.models.omega17_vl_exp.configuration_omega17_vl_exp import Omega17VLExpConfig, Omega17VLExpTextConfig, Omega17VLExpVisionConfig; \
print('✓ All Omega17 configs imported successfully')"
```

This ensures the build fails early if the custom transformers package has issues.

---

## Files Modified

### Core vLLM Files
1. **`vllm/transformers_utils/configs/__init__.py`**
   - Added Omega17 config imports from custom transformers
   - Added configs to `__all__` export list

2. **`vllm/transformers_utils/config.py`**
   - Relies on AutoConfig fallback (no explicit registry mapping)

3. **`vllm/model_executor/models/omega17_vl_exp.py`**
   - Model implementation for Omega17VLExp

4. **`vllm/model_executor/models/registry.py`**
   - Registered Omega17VLExpForConditionalGeneration

5. **`vllm/v1/spec_decode/eagle.py`**
   - Added Omega17VLExpForConditionalGeneration handling for speculative decoding

### Docker Files
1. **`docker/Dockerfile`**
   - Added custom transformers installation step
   - Added Omega17 config verification
   - Uses `--prerelease=allow` for pre-release dependencies

2. **`requirements/common.txt`**
   - Standard transformers (overridden by Dockerfile)

---

## Troubleshooting

### Build Issues

**Problem:** `AttributeError: module 'vllm.transformers_utils.configs' has no attribute 'Omega17VLExpConfig'`

**Solution:** Ensure the custom transformers package is installed correctly in the Dockerfile after vLLM wheel installation.

---

**Problem:** Pre-release dependency conflicts

**Solution:** Use `--prerelease=allow` flag when installing custom transformers:
```bash
uv pip install --system --prerelease=allow git+https://github.com/apt-team-018/transformers-usf-exp.git
```

---

**Problem:** Slow Docker builds

**Solution:** 
- Use `--cache-from` with a previous build
- Ensure `USE_SCCACHE=1` is set
- Adjust `max_jobs` and `nvcc_threads` based on server specs

---

### Runtime Issues

**Problem:** Model not loading

**Solution:** Ensure `HUGGING_FACE_HUB_TOKEN` is set and has access to the model repo.

---

**Problem:** Out of memory errors

**Solution:** Adjust `--gpu-memory-utilization` (try 0.85 or 0.80) or reduce `--max-model-len`.

---

## Testing

### Quick Import Test
```bash
# Test Omega17 configs are available
docker run --rm arpitsh018/vllm-openai:test-2.3 python3 -c "\
from transformers.models.omega17_vl_exp import Omega17VLExpConfig; \
print('✓ Omega17 configs available')"
```

### Health Check
```bash
# Check if server is running
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

---

## Version History

- **test-2.3** - Omega17 support with config verification (Nov 2025)
- **test-2.2** - Initial Omega17 integration
- **test-2** - Base vLLM build

---

## Contact

- **Docker Hub:** arpitsh018
- **Model Repository:** https://huggingface.co/arpitsh018/omega-17-exp-vl-v0.1-checkpoint-1020
- **Custom Transformers:** https://github.com/apt-team-018/transformers-usf-exp.git

---

*Last Updated: November 2025*
