#!/bin/bash
# ============================================================
# Environment Setup — TPAB-RewardFlow SciWorld
# Target: RTX PRO 6000 Blackwell Workstation (sm_120, CUDA 12.8+ driver,
#         Python 3.10/3.11/3.12)
#
# Notes vs. the previous H200 (sm_90, cu124/torch2.6) target:
#   - PyTorch 2.6 has no sm_120 kernels. Bumped to 2.7.1 + cu128 wheels.
#   - vLLM 0.8.5 predates Blackwell support. Bumped to 0.9.2 (first stable
#     release with sm_100/sm_120 PTX + kernels).
#   - flash-attn / flashinfer wheels switched to the cu12torch2.7 / cu128
#     builds, which ship sm_120 binaries.
#   - TransformerEngine v2.2 cannot compile for sm_120. Bumped to v2.3.
#
# Usage:
#   bash rm/tpab_rewardflow/setup_env.sh [OPTIONS]
#
# Options:
#   --skip-megatron     Skip TransformerEngine + Megatron-LM (saves ~20 min)
#   --use-sglang        Also install SGLang (default: vllm only)
#   --skip-flash-attn   Skip Flash Attention wheel install
#   --conda-env NAME    conda activate NAME before installing
# ============================================================
set -e

# ---- Parse flags ----
USE_MEGATRON=1
USE_SGLANG=0
SKIP_FLASH_ATTN=0
CONDA_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-megatron)   USE_MEGATRON=0;    shift ;;
        --use-sglang)      USE_SGLANG=1;      shift ;;
        --skip-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
        --conda-env)       CONDA_ENV="$2";    shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---- Conda (optional) ----
if [[ -n "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export MAX_JOBS="${MAX_JOBS:-16}"

# ---- Repo root (rm/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."   # rm/

# ---- CUDA check ----
# Prefer nvcc when present; otherwise fall back to the driver-reported CUDA
# version from nvidia-smi (the toolkit is not strictly required for installing
# pre-built wheels, only a driver new enough for the wheel's CUDA runtime).
CUDA_MAJOR=0
CUDA_MINOR=0
if command -v nvcc &>/dev/null; then
    CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1 || echo "0")
    CUDA_MINOR=$(nvcc --version | grep -oP 'release [0-9]+\.\K[0-9]+' | head -1 || echo "0")
    echo "[setup] CUDA toolkit (nvcc): $CUDA_MAJOR.$CUDA_MINOR"
elif command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "0.0")
    CUDA_MAJOR=${CUDA_VER%%.*}
    CUDA_MINOR=${CUDA_VER##*.}
    echo "[setup] CUDA driver (nvidia-smi): $CUDA_VER (nvcc not installed)"
fi
# Blackwell sm_120 requires the cu128 runtime, which needs driver >= 555 and a
# reported CUDA Version >= 12.8 in nvidia-smi. We only hard-fail on majors below
# 12, and warn for 12.x < 12.8 so older toolkits can still attempt the install.
if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    echo "[ERROR] CUDA 12.8+ driver required for Blackwell sm_120. Found CUDA $CUDA_MAJOR.$CUDA_MINOR. Exiting."
    exit 1
fi
if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -lt 8 ]]; then
    echo "[WARN] CUDA $CUDA_MAJOR.$CUDA_MINOR detected; Blackwell wheels target cu128. Update the driver to 555+ if you hit runtime errors."
fi

# Expose nvcc / CUDA dev libs to source builds (TransformerEngine, etc.).
# When the system installs cuda-toolkit-12-x packages, nvcc lands under
# /usr/local/cuda. Make sure it is on PATH and CUDA_HOME is set so that
# TE's setup.py finds nvcc at the expected location.
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    echo "[setup] CUDA_HOME=${CUDA_HOME} (nvcc: $(${CUDA_HOME}/bin/nvcc --version | grep release))"
fi

# ---- Java (required by SciWorld / TextWorldExpress) ----
echo ""
echo "=== [1/7] Java ==="
if ! command -v java &>/dev/null; then
    echo "[setup] Installing OpenJDK 17..."
    SUDO=""
    [[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
    $SUDO apt-get update -qq && $SUDO apt-get install -y --no-install-recommends openjdk-17-jre-headless
else
    echo "[setup] $(java -version 2>&1 | head -1)"
fi

# ---- SGLang (optional) ----
if [[ "$USE_SGLANG" -eq 1 ]]; then
    echo ""
    echo "=== [2/7] SGLang ==="
    # SGLang 0.4.8 is the first release that ships cu128/torch2.7 wheels and
    # bundles a Blackwell-capable flashinfer build. Earlier 0.4.6.x is cu124.
    pip install "sglang[all]==0.4.8" --no-cache-dir \
        --find-links "https://flashinfer.ai/whl/cu128/torch2.7/flashinfer-python"
    pip install torch-memory-saver --no-cache-dir
else
    echo ""
    echo "=== [2/7] SGLang — skipped (pass --use-sglang to enable) ==="
fi

# ---- PyTorch + vLLM ----
echo ""
echo "=== [3/7] PyTorch 2.7 + vLLM 0.9.2 (cu128, Blackwell) ==="
# Pull torch from the PyTorch cu128 index so we get sm_120 kernels. vLLM 0.9.2
# resolves against torch==2.7.1 and ships its own cu128 wheel on PyPI, so the
# extra-index-url is enough — no manual wheel URLs needed.
pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "vllm==0.9.2" \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    "torchaudio==2.7.0" \
    "tensordict>=0.7.0,<0.8" \
    torchdata

# ---- Core ML / RL packages ----
echo ""
echo "=== [4/7] Core packages ==="
# Notes:
#   - transformers upper-bounded at <4.60: verl-agent pins to ~=4.51, and
#     leaving it open lets pip resolve to 5.x which breaks the import chain.
#   - pyext is intentionally dropped: it uses inspect.getargspec which was
#     removed in Python 3.11+, so the wheel build fails on Python 3.12.
#   - gym + gymnasium are required by agent_system/environments/env_package/
#     (SciWorld imports gym; other envs import gymnasium).
#   - ray upper-bounded at <2.49: Ray 2.49+ added the dashboard
#     `aggregator_agent` which calls
#     `Meter.create_histogram(..., explicit_bucket_boundaries_advisory=...)`.
#     That kwarg only exists in opentelemetry-api >=1.28, but the [post]
#     step pins opentelemetry to 1.26.x for vLLM 0.8.5 compatibility, so
#     newer Ray fails on `ray.init()` with a TypeError → raylet timeout.
pip install --no-cache-dir \
    "transformers[hf_xet]>=4.51.0,<4.60" \
    accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    "ray[default]>=2.43,<2.49" codetiming hydra-core pylatexenc \
    qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
    pytest py-spy pre-commit ruff \
    gym gymnasium \
    scienceworld

# ---- Flash Attention + FlashInfer (Blackwell = sm_120) ----
if [[ "$SKIP_FLASH_ATTN" -eq 0 ]]; then
    echo ""
    echo "=== [5/7] Flash Attention 2 + FlashInfer (cu128/torch2.7, sm_120) ==="
    # Detect Python version tag (cp310 / cp311 / cp312 / cp313)
    PY_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    echo "[setup] Python tag: $PY_TAG"
    # flash-attn 2.8.0.post2 is the first release whose pre-built wheels include
    # sm_120 binaries (alongside sm_80/86/89/90/100). The cu12torch2.7 wheel is
    # the one to match our torch 2.7.1 + cu128 install above.
    FLASH_WHL="flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    wget -nv "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/${FLASH_WHL}" \
         -O "/tmp/${FLASH_WHL}"
    pip install --no-cache-dir "/tmp/${FLASH_WHL}"
    rm -f "/tmp/${FLASH_WHL}"

    # flashinfer changed its packaging in v0.4.1: the old CUDA-specific
    # binary wheels (the 0.2.x line we were grabbing for H200) are no longer
    # hosted anywhere — even GitHub releases for v0.2.6.post1 have zero
    # assets. v0.4.1+ ships a single py3-none-any wheel that JITs kernels
    # at runtime, which is what we want on Blackwell anyway.
    # vllm 0.9.2 does not pin flashinfer; it falls back to its bundled
    # FlashAttention path when flashinfer is missing, so make the install
    # non-fatal.
    pip install --no-cache-dir flashinfer-python \
        --find-links https://flashinfer.ai/whl/flashinfer-python/ \
        || echo "[WARN] flashinfer install failed; vllm will fall back to its bundled attention kernels"
else
    echo ""
    echo "=== [5/7] Flash Attention — skipped ==="
fi

# ---- TransformerEngine + Megatron-LM ----
if [[ "$USE_MEGATRON" -eq 1 ]]; then
    echo ""
    echo "=== [6/7] TransformerEngine + Megatron-LM (10-20 min) ==="
    # TransformerEngine's setup.py does `import torch` at build time. pip's
    # default PEP 517 build isolation creates a fresh env without torch, so
    # the build fails with `ModuleNotFoundError: No module named 'torch'`.
    # Use --no-build-isolation so the build sees the torch installed in
    # step [3/7], and pre-install the other build-time prerequisites that
    # would otherwise have been pulled into the isolated env.
    pip install --no-cache-dir \
        "setuptools>=61" wheel "packaging>=23" "pybind11>=2.12" ninja cmake
    # TE v2.3 is the earliest tag that compiles sm_120 fused kernels (v2.2
    # caps the arch list at sm_100). Constrain TORCH_CUDA_ARCH_LIST so we
    # don't waste compile time on archs we don't have.
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
    NVTE_FRAMEWORK=pytorch pip3 install --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/TransformerEngine.git@v2.3"
    pip3 install --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3"
    pip install --no-cache-dir "nvidia-cudnn-cu12==9.8.0.87"
else
    echo ""
    echo "=== [6/7] Megatron-LM — skipped (remove --skip-megatron to enable) ==="
fi

# ---- Install verl-agent package (rm/) ----
echo ""
echo "=== [7/7] Installing verl-agent package ==="
pip install --no-cache-dir -e "$REPO_DIR"

# ---- Re-pin opentelemetry to ray-dashboard-compatible versions ----
# vllm 0.9.2 itself no longer caps opentelemetry-api below 1.27, but the ray
# dashboard side of the pin still matters: `datasets` + the verl-agent install
# pull opentelemetry-* >=1.42 and exporter-prometheus 0.63b1, which imports
# `OtelComponentTypeValues` from semconv. That symbol only exists in semconv
# >=0.48b0, but ray<2.49 (pinned in step [4/7]) trips its own dashboard
# regression with newer semconv builds, so we hold the 1.26 / 0.47b0 line.
echo ""
echo "=== [post] Pinning opentelemetry to ray-dashboard-compatible versions ==="
pip install --no-cache-dir \
    "opentelemetry-api>=1.26.0,<1.27.0" \
    "opentelemetry-sdk>=1.26.0,<1.27.0" \
    "opentelemetry-semantic-conventions==0.47b0" \
    "opentelemetry-exporter-prometheus==0.47b0"

# ---- Verify SciWorld ----
echo ""
echo "[setup] Verifying SciWorld..."
python3 - <<'EOF'
try:
    from scienceworld import ScienceWorldEnv
    sw = ScienceWorldEnv("")
    tasks = sw.get_task_names()
    print(f"[OK] SciWorld: {len(tasks)} tasks available")
    sw.close()
except Exception as e:
    print(f"[WARN] SciWorld init failed: {e}")
    print("[WARN] Make sure Java 17+ is installed: java -version")
EOF

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Required before training:"
echo "   export HF_TOKEN=<your_huggingface_token>   # model download"
echo "   export WANDB_API_KEY=<your_wandb_key>       # logging (optional)"
echo ""
echo " Run training:"
echo "   bash rm/tpab_rewardflow/run_tpab_rewardflow_sciworld.sh"
echo "============================================================"
