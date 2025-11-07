#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the virtual environment directory
VENV_DIR="venv"

# Check if the virtual environment activation script exists.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No virtual environment found. Creating one in '$VENV_DIR'..."
    # Create the virtual environment using python3.
    python3 -m venv "$VENV_DIR"

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo "Detecting GPU support to pick the correct PyTorch wheel..."
    TORCH_CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL:-"https://download.pytorch.org/whl/nightly/cu130"}
    TORCH_CPU_INDEX_URL=${PYTORCH_CPU_INDEX_URL:-"https://download.pytorch.org/whl/nightly/cpu"}

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "Detected NVIDIA GPU. Installing CUDA-enabled PyTorch from $TORCH_CUDA_INDEX_URL"
        TORCH_INDEX_URL="$TORCH_CUDA_INDEX_URL"
    else
        echo "No NVIDIA GPU detected; installing CPU-only PyTorch from $TORCH_CPU_INDEX_URL"
        TORCH_INDEX_URL="$TORCH_CPU_INDEX_URL"
    fi

    echo "Installing benchmark-specific dependencies..."
    pip install numpy Pillow py-cpuinfo
    pip install --upgrade torch torchvision --index-url "$TORCH_INDEX_URL"

    echo "Validating CUDA availability inside the virtualenv..."
    if python - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(2)

if not torch.cuda.is_available():
    sys.exit(1)

try:
    torch.zeros(1, device='cuda').sum().item()
except Exception:
    sys.exit(3)

sys.exit(0)
PY
    then
        echo "CUDA is available and kernels run successfully."
    else
        status=$?
        echo "CUDA unusable in this environment (exit $status); reinstalling CPU-only PyTorch."
        pip uninstall -y torch torchvision
        pip install --upgrade torch torchvision --index-url "$TORCH_CPU_INDEX_URL"
        TORCH_INDEX_URL="$TORCH_CPU_INDEX_URL"
    fi

    echo "Installing TorchLanc in editable mode..."
    pip install -e .

    echo "Setup complete."
else
    echo "Found existing virtual environment."
fi


echo ""
echo "=================================================================="
echo "Activating environment and running the benchmark race..."
echo "=================================================================="
echo ""

# Activate the venv (sources again to ensure it's active for this shell session)
# and run the benchmark.
source "$VENV_DIR/bin/activate"
python benchmark/benchmark.py --race

echo ""
echo "Benchmark complete."
