#!/bin/bash
# =============================================================================
# BlockFFN Adaptive Inference — Full Setup + Test + Benchmark
# =============================================================================
# Runs everything from scratch. Stops on any error so we can debug.
#
# Usage:
#   cd ~/blockffn-mamba
#   bash scripts/run_all.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_STEPS=12
RESULTS_FILE="test_results.txt"

step() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}[$1/$TOTAL_STEPS] $2${NC}"
    echo -e "${CYAN}============================================${NC}"
}

pass() {
    echo -e "${GREEN}  PASS: $1${NC}"
    echo "[PASS] $1" >> "$RESULTS_FILE"
}

fail() {
    echo -e "${RED}  FAIL: $1${NC}"
    echo "[FAIL] $1" >> "$RESULTS_FILE"
    echo ""
    echo -e "${RED}Stopped. Debug the error above, then re-run.${NC}"
    exit 1
}

# Clear results
> "$RESULTS_FILE"
echo "Started: $(date)" >> "$RESULTS_FILE"
echo ""

# =============================================================================
# PHASE 1: ENVIRONMENT SETUP
# =============================================================================

step 1 "Creating fresh virtual environment"
if [ -d ".venv" ]; then
    rm -rf .venv
fi
python3 -m venv .venv
source .venv/bin/activate
pass "venv created"

step 2 "Installing PyTorch + CUDA"
pip install --upgrade pip setuptools wheel -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
pip install transformers psutil numpy tqdm dm-tree -q
pass "dependencies installed"

step 3 "Creating kernel stubs"
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# blockffn_kernel
mkdir -p "$SITE_PACKAGES/blockffn_kernel"
cat > "$SITE_PACKAGES/blockffn_kernel/__init__.py" << 'STUB'
class BlockFFN:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return None
STUB

# fmoe
mkdir -p "$SITE_PACKAGES/fmoe"
cat > "$SITE_PACKAGES/fmoe/__init__.py" << 'STUB'
STUB
cat > "$SITE_PACKAGES/fmoe/linear.py" << 'STUB'
import torch.nn as nn
class MOELinear(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, *args, **kwargs): raise NotImplementedError("fmoe stub")
STUB
cat > "$SITE_PACKAGES/fmoe/layers.py" << 'STUB'
def _fmoe_general_global_forward(*args, **kwargs): raise NotImplementedError("fmoe stub")
STUB

# megablocks
mkdir -p "$SITE_PACKAGES/megablocks"
cat > "$SITE_PACKAGES/megablocks/__init__.py" << 'STUB'
class _Placeholder:
    def __getattr__(self, name): return _Placeholder()
    def __call__(self, *args, **kwargs): raise NotImplementedError("megablocks stub")
stk = _Placeholder()
sparse_act = _Placeholder()
class dMoE:
    def __init__(self, *args, **kwargs): pass
STUB

pass "kernel stubs created"

step 4 "Verifying environment"
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  PyTorch {torch.__version__}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import tree
from fmoe.linear import MOELinear
from fmoe.layers import _fmoe_general_global_forward
from megablocks import stk, sparse_act, dMoE
from blockffn_kernel import BlockFFN
print('  Stubs: OK')

from src.adaptive.config import PRESETS
print(f'  Framework: {len(PRESETS)} presets')
" || fail "environment verification"
pass "environment OK"

# =============================================================================
# PHASE 2: FRAMEWORK TESTS (no model needed)
# =============================================================================

step 5 "Unit tests (no torch)"
python tests/test_unit_no_torch.py || fail "unit tests"
pass "unit tests"

step 6 "Phase A+C model-free tests"
python tests/run_tests.py --phase A C --skip-model-tests || fail "phase A+C model-free"
pass "phase A+C model-free"

step 7 "GPT-2 surrogate tests on CUDA"
python tests/test_with_standard_model.py --device cuda || fail "GPT-2 surrogate tests"
pass "GPT-2 tests on CUDA"

# =============================================================================
# PHASE 3: BLOCKFFN MODEL
# =============================================================================

step 8 "Loading BlockFFN-3B-SFT"
# Use local cache if available
if [ -d "$HOME/blockffn-mamba/cache" ]; then
    export HF_HOME="$HOME/blockffn-mamba/cache"
    echo "  Using local cache: $HF_HOME"
fi

python << 'PYEOF' || fail "BlockFFN model loading"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("  Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "SparseLLM/BlockFFN-3B-SFT",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    "SparseLLM/BlockFFN-3B-SFT", trust_remote_code=True
)

num_layers = len(model.model.layers)
layer = model.model.layers[0]
mlp = layer.mlp
print(f"  Layers: {num_layers}")
print(f"  Layer type: {type(layer).__name__}")
print(f"  MLP type: {type(mlp).__name__}")
print(f"  Has router_proj: {hasattr(mlp, 'router_proj')}")
if hasattr(mlp, 'router_proj'):
    print(f"  Router shape: {mlp.router_proj.weight.shape}")
    print(f"  Num experts: {mlp.router_proj.weight.shape[0]}")

# Quick generation test
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(inputs["input_ids"], max_new_tokens=20, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"  Generation: {text}")

# Memory check
mem_used = torch.cuda.max_memory_allocated() / 1024**3
mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  VRAM used: {mem_used:.1f} / {mem_total:.1f} GB")
PYEOF
pass "BlockFFN loaded and generates text"

step 9 "Phase A correctness tests with BlockFFN"
python tests/run_tests.py --phase A --device cuda || fail "phase A with BlockFFN"
pass "phase A correctness"

step 10 "Phase B savings quantification"
python tests/run_tests.py --phase B --device cuda || fail "phase B savings"
pass "phase B savings"

step 11 "Phase C stress tests"
python tests/run_tests.py --phase C --device cuda --force || fail "phase C stress"
pass "phase C stress"

step 12 "Escalation strategy comparison"
python tests/test_escalation_strategies.py --device cuda || fail "escalation strategies"
pass "escalation strategies"

# =============================================================================
# FINAL RESULTS
# =============================================================================

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ALL DONE${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results:"
cat "$RESULTS_FILE"
echo ""
echo "Finished: $(date)" >> "$RESULTS_FILE"
echo -e "${YELLOW}Next: run benchmarks manually:${NC}"
echo "  source .venv/bin/activate"
echo "  export HF_HOME=~/blockffn-mamba/cache"
echo "  python scripts/benchmark_harness.py --device cuda --preset balanced"
echo "  python scripts/benchmark_harness.py --device cuda --preset aggressive"
echo "  python scripts/benchmark_harness.py --device cuda --preset fast"
echo ""
