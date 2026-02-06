#!/bin/bash
# BlockFFN + Mamba Hybrid Environment Setup
# Run this script on WSL2 (Ubuntu) with NVIDIA GPU
# On Mac, mamba_ssm will fail to install - that's expected

set -e

ENV_NAME="blockffn-mamba"
PYTHON_VERSION="3.11"

echo "=== BlockFFN + Mamba Hybrid Environment Setup ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch with CUDA 12.1
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install HuggingFace libraries
echo ""
echo "Installing HuggingFace libraries..."
pip install transformers>=4.40.0 accelerate safetensors

# Install evaluation and utility packages
echo ""
echo "Installing evaluation and utility packages..."
pip install datasets matplotlib jupyter tqdm pandas

# Install bitsandbytes for 8-bit quantization (fallback for OOM)
echo ""
echo "Installing bitsandbytes..."
pip install bitsandbytes

# Install Mamba dependencies
echo ""
echo "Installing Mamba dependencies..."
echo "NOTE: This requires CUDA toolkit. If it fails on Mac, that's expected."
echo ""

# Install causal-conv1d first (required by mamba_ssm)
pip install ninja packaging wheel

# Try to install causal-conv1d
echo "Installing causal-conv1d..."
if pip install causal-conv1d>=1.4.0; then
    echo "causal-conv1d installed successfully"
else
    echo "WARNING: causal-conv1d installation failed."
    echo "Trying with --no-build-isolation..."
    pip install --no-build-isolation causal-conv1d>=1.4.0 || {
        echo "WARNING: causal-conv1d could not be installed."
        echo "This is expected on Mac. Mamba will use a linear fallback."
    }
fi

# Try to install mamba-ssm
echo ""
echo "Installing mamba-ssm..."
if pip install mamba-ssm; then
    echo "mamba-ssm installed successfully"
else
    echo "WARNING: mamba-ssm installation failed."
    echo "Trying with --no-build-isolation..."
    pip install --no-build-isolation mamba-ssm || {
        echo ""
        echo "=========================================="
        echo "WARNING: mamba-ssm could not be installed."
        echo "=========================================="
        echo ""
        echo "This is EXPECTED on Mac (no CUDA)."
        echo "The code will use a linear fallback for development."
        echo ""
        echo "On WSL2 with NVIDIA GPU, try:"
        echo "  1. Ensure CUDA toolkit is installed: nvcc --version"
        echo "  2. Install matching PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        echo "  3. Retry: pip install --no-build-isolation mamba-ssm"
        echo ""
    }
fi

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import transformers
print(f'Transformers: {transformers.__version__}')

try:
    from mamba_ssm import Mamba2
    print('Mamba2: Available')
except ImportError:
    print('Mamba2: Not available (will use fallback)')

print()
print('Environment setup complete!')
print(f'Activate with: conda activate {ENV_NAME}')
"

echo ""
echo "=== Setup Complete ==="
echo "To activate: conda activate $ENV_NAME"
echo "To run scripts: python scripts/01_download_model.py"
