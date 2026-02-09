#!/bin/bash
# =============================================================================
# BlockFFN Adaptive Inference — WSL + NVIDIA GPU Setup
# =============================================================================
# This script sets up the full framework on WSL with CUDA support.
#
# The BlockFFN-3B-SFT model's HuggingFace code has unconditional imports for
# 4 custom packages (blockffn_kernel, fmoe, megablocks, tree). However, the
# model config uses block_implementation="torch" so none of these kernels are
# actually called — pure PyTorch is used. We create lightweight stubs so the
# imports succeed without compiling any CUDA kernels.
#
# Usage:
#   cd ~/blockffn-mamba
#   bash scripts/setup_wsl.sh
# =============================================================================

set -e

echo "============================================"
echo "BlockFFN Adaptive Inference — WSL Setup"
echo "============================================"

# --- 1. Clean slate ---
echo ""
echo "[1/7] Creating fresh virtual environment..."
if [ -d ".venv" ]; then
    rm -rf .venv
fi
python3 -m venv .venv
source .venv/bin/activate

# --- 2. Core dependencies ---
echo ""
echo "[2/7] Installing PyTorch + CUDA..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "[3/7] Installing framework dependencies..."
# Pin transformers to 4.44.2 — BlockFFN's HuggingFace code uses imports
# removed in transformers 5.x (is_torch_fx_available, _prepare_4d_causal_attention_mask_for_sdpa, etc.)
pip install "transformers==4.44.2" psutil numpy tqdm

# --- 3. dm-tree (pure pip, the 'import tree' in model code) ---
echo ""
echo "[4/7] Installing dm-tree..."
pip install dm-tree

# --- 4. Create stubs for kernel packages ---
# The model's modeling_minicpm.py has these unconditional imports:
#   import tree                                          <- dm-tree (installed above)
#   from fmoe.linear import MOELinear                    <- stub
#   from fmoe.layers import _fmoe_general_global_forward <- stub
#   from megablocks import stk, sparse_act, dMoE         <- stub
#   from blockffn_kernel import BlockFFN as BlockFFNKernel <- stub
#
# Since block_implementation="torch" in config.json, none of these are called.
# We just need the imports to not crash.

echo ""
echo "[5/7] Creating kernel stubs (block_implementation='torch', kernels not called)..."

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# --- blockffn_kernel stub ---
mkdir -p "$SITE_PACKAGES/blockffn_kernel"
cat > "$SITE_PACKAGES/blockffn_kernel/__init__.py" << 'STUB'
"""Stub for blockffn_kernel — model uses block_implementation='torch'."""

class BlockFFN:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return None
STUB

# --- fmoe stub ---
mkdir -p "$SITE_PACKAGES/fmoe"
cat > "$SITE_PACKAGES/fmoe/__init__.py" << 'STUB'
"""Stub for fmoe (FastMoE) — model uses block_implementation='torch'."""
STUB

cat > "$SITE_PACKAGES/fmoe/linear.py" << 'STUB'
"""Stub for fmoe.linear"""
import torch.nn as nn

class MOELinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        raise NotImplementedError("fmoe stub — not used with block_implementation='torch'")
STUB

cat > "$SITE_PACKAGES/fmoe/layers.py" << 'STUB'
"""Stub for fmoe.layers"""

def _fmoe_general_global_forward(*args, **kwargs):
    raise NotImplementedError("fmoe stub — not used with block_implementation='torch'")
STUB

# --- megablocks stub ---
mkdir -p "$SITE_PACKAGES/megablocks"
cat > "$SITE_PACKAGES/megablocks/__init__.py" << 'STUB'
"""Stub for megablocks — model uses block_implementation='torch'."""

class _Placeholder:
    def __getattr__(self, name):
        return _Placeholder()
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("megablocks stub — not used with block_implementation='torch'")

stk = _Placeholder()
sparse_act = _Placeholder()

class dMoE:
    def __init__(self, *args, **kwargs):
        pass
STUB

echo "   Stubs created in: $SITE_PACKAGES"

# --- 5. Verify setup ---
echo ""
echo "[6/7] Verifying installation..."

python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import transformers
print(f'  Transformers: {transformers.__version__}')

# Verify stubs load
import tree
from fmoe.linear import MOELinear
from fmoe.layers import _fmoe_general_global_forward
from megablocks import stk, sparse_act, dMoE
from blockffn_kernel import BlockFFN
print('  Kernel stubs: OK')

# Verify framework
from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
print(f'  Framework:    OK ({len(PRESETS)} presets)')
"

# --- 6. Point at local cache if it exists ---
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "============================================"
echo "  NEXT STEPS"
echo "============================================"
echo ""
echo "  # If your BlockFFN model is already cached:"
echo "  export HF_HOME=~/blockffn-mamba/cache"
echo ""
echo "  # Quick smoke test (no model needed):"
echo "  python tests/test_unit_no_torch.py"
echo "  python tests/run_tests.py --phase A C --skip-model-tests"
echo ""
echo "  # GPT-2 tests on GPU:"
echo "  python tests/test_with_standard_model.py --device cuda"
echo ""
echo "  # Full BlockFFN tests:"
echo "  python tests/run_tests.py --phase A B C --device cuda"
echo ""
echo "  # Benchmark:"
echo "  python scripts/benchmark_harness.py --device cuda --preset balanced"
echo ""
