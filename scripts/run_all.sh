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

TOTAL_STEPS=15
RESULTS_FILE="test_results.txt"
BENCHMARK_CSV="benchmark_results.csv"

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
# PHASE 0: FIND CACHED BLOCKFFN MODEL
# =============================================================================
# Search common HuggingFace cache locations for the BlockFFN model.
# This avoids re-downloading ~6GB if it's already on the machine.

echo -e "${CYAN}Searching for cached BlockFFN-3B-SFT model...${NC}"

MODEL_FOUND=false

# Search order: project cache, HF default, custom locations
SEARCH_PATHS=(
    "$HOME/blockffn-mamba/cache"
    "${HF_HOME:-}"
    "$HOME/.cache/huggingface"
    "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"
    "/mnt/c/Users/*/cache/huggingface"
    "/mnt/d/huggingface"
    "/mnt/e/huggingface"
)

for search_path in "${SEARCH_PATHS[@]}"; do
    # Skip empty paths
    [ -z "$search_path" ] && continue

    # Expand globs (for /mnt/c/Users/*)
    for expanded_path in $search_path; do
        [ -d "$expanded_path" ] || continue

        # Check for model files in HF hub cache structure
        # HuggingFace stores models under hub/models--{org}--{repo}/
        MODEL_DIR="$expanded_path/hub/models--SparseLLM--BlockFFN-3B-SFT"
        if [ -d "$MODEL_DIR" ]; then
            # Verify it has actual model files (not just metadata)
            if find "$MODEL_DIR" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | head -1 | grep -q .; then
                export HF_HOME="$expanded_path"
                MODEL_FOUND=true
                echo -e "${GREEN}  Found cached model at: $expanded_path${NC}"
                echo -e "  Model dir: $MODEL_DIR"
                # Show model size
                MODEL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
                echo -e "  Cache size: $MODEL_SIZE"
                break
            fi
        fi
    done
    $MODEL_FOUND && break
done

if ! $MODEL_FOUND; then
    echo -e "${YELLOW}  No cached model found. Will download from HuggingFace (~6GB).${NC}"
    echo -e "${YELLOW}  Searched: ${SEARCH_PATHS[*]}${NC}"
    # Default HF_HOME so the CACHED_MODULES cleanup below doesn't fail on unbound var
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi

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
# Pin transformers to 4.44.2 — BlockFFN's HuggingFace code uses imports
# removed in transformers 5.x (is_torch_fx_available, _prepare_4d_causal_attention_mask_for_sdpa, etc.)
pip install "transformers==4.44.2" psutil numpy tqdm dm-tree -q
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
# PHASE 3: BLOCKFFN MODEL TESTS
# =============================================================================

step 8 "Loading BlockFFN-3B-SFT"
echo "  HF_HOME=$HF_HOME"

# Clear stale cached model code (may have been downloaded with wrong transformers version)
CACHED_MODULES="$HF_HOME/modules/transformers_modules/SparseLLM"
if [ -d "$CACHED_MODULES" ]; then
    echo "  Clearing stale cached model code..."
    rm -rf "$CACHED_MODULES"
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
# PHASE 4: BENCHMARKS
# =============================================================================

step 13 "Benchmark — balanced preset"
python scripts/benchmark_harness.py \
    --device cuda \
    --presets balanced \
    --batch-sizes 1 \
    --context-lengths 128,256,512 \
    --num-tokens 50 \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --output "$BENCHMARK_CSV" \
    || fail "benchmark balanced"
pass "benchmark balanced"

step 14 "Benchmark — aggressive preset"
python scripts/benchmark_harness.py \
    --device cuda \
    --presets aggressive \
    --batch-sizes 1 \
    --context-lengths 128,256,512 \
    --num-tokens 50 \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --output "$BENCHMARK_CSV" \
    --resume benchmark_checkpoint.json \
    || fail "benchmark aggressive"
pass "benchmark aggressive"

step 15 "Benchmark — fast preset"
python scripts/benchmark_harness.py \
    --device cuda \
    --presets fast \
    --batch-sizes 1 \
    --context-lengths 128,256,512 \
    --num-tokens 50 \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --output "$BENCHMARK_CSV" \
    --resume benchmark_checkpoint.json \
    || fail "benchmark fast"
pass "benchmark fast"

# =============================================================================
# FINAL RESULTS
# =============================================================================

echo "Finished: $(date)" >> "$RESULTS_FILE"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ALL 15 STEPS COMPLETE${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${CYAN}TEST RESULTS:${NC}"
cat "$RESULTS_FILE"
echo ""

# Print benchmark summary
if [ -f "$BENCHMARK_CSV" ]; then
    echo -e "${CYAN}BENCHMARK SUMMARY ($BENCHMARK_CSV):${NC}"
    echo ""
    # Print CSV header + data as a readable table
    python << 'PYEOF'
import csv
import sys

try:
    with open("benchmark_results.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  No benchmark results found.")
        sys.exit(0)

    # Print compact summary
    print(f"  {'Preset':<14} {'Context':>8} {'TTFT(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10} {'tok/s':>8} {'Attn Save':>10} {'Esc Rate':>10} {'Peak MB':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for row in rows:
        print(f"  {row.get('preset','?'):<14} "
              f"{row.get('context_length','?'):>8} "
              f"{float(row.get('ttft_ms',0)):>10.1f} "
              f"{float(row.get('p50_token_latency_ms',0)):>10.1f} "
              f"{float(row.get('p95_token_latency_ms',0)):>10.1f} "
              f"{float(row.get('tokens_per_second',0)):>8.1f} "
              f"{float(row.get('theoretical_attention_savings',0))*100:>9.1f}% "
              f"{float(row.get('overall_escalation_rate',0))*100:>9.1f}% "
              f"{float(row.get('peak_memory_mb',0)):>10.1f}")
    print()
except Exception as e:
    print(f"  Could not parse benchmark CSV: {e}")
PYEOF
fi

echo -e "${GREEN}All tests passed. Benchmark data saved to: $BENCHMARK_CSV${NC}"
echo ""
