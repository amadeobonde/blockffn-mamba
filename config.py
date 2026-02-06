"""
Configuration for BlockFFN + Mamba Hybrid Experiments.

All hyperparameters and paths in one place for easy modification.
"""

import torch
from pathlib import Path

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "SparseLLM/BlockFFN-3B-SFT"
MODEL_CACHE_DIR = Path("./cache")

# Device configuration
# Change to "cpu" for Mac testing of non-GPU code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =============================================================================
# Mamba Sidecar Configuration
# =============================================================================

# Mamba2 hyperparameters
# d_state reduced from 128 to 64 to fit in RTX 3060 12GB
MAMBA_D_STATE = 64
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
MAMBA_HEADDIM = 64

# =============================================================================
# Experiment Configuration
# =============================================================================

# Alpha values to sweep: 0=pure attention, 1=pure Mamba
ALPHA_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Which layers to attach Mamba sidecars
# Options: "all", "middle", "first_half", "last_half", or list of ints
TARGET_LAYERS = "middle"

# Gate modes for hybrid layer
# "fixed": constant alpha for all tokens
# "routing": use BlockFFN routing sparsity as gate signal
# "learned": learn a gate projection
GATE_MODE = "routing"

# Sparsity threshold for hard gating
# Tokens with sparsity > threshold use Mamba
SPARSITY_THRESHOLD = 0.7

# Temperature for gate sigmoid (higher = sharper gating)
GATE_TEMPERATURE = 1.0

# =============================================================================
# Evaluation Configuration
# =============================================================================

EVAL_DATASET = "wikitext"
EVAL_SPLIT = "test"
MAX_EVAL_SAMPLES = 100
MAX_SEQ_LEN = 512

# Generation settings
MAX_NEW_TOKENS = 50
GENERATION_DO_SAMPLE = False  # Greedy for reproducibility

# =============================================================================
# Test Prompts for Qualitative Checks
# =============================================================================

TEST_PROMPTS = [
    "The capital of France is",
    "In a groundbreaking study, researchers discovered that",
    "def fibonacci(n):\n",
    "The quick brown fox jumps over the lazy dog. The dog then",
    "The key difference between machine learning and deep learning is",
    "To solve this equation, we first need to",
]

# =============================================================================
# Memory Management
# =============================================================================

# Maximum memory to use (GB) - leave buffer for system
MAX_MEMORY_GB = 11.0

# Use 8-bit quantization if OOM
USE_8BIT_FALLBACK = True

# Reduce sequence length if OOM
OOM_SEQ_LEN_FALLBACK = 256

# =============================================================================
# Output Paths
# =============================================================================

OUTPUT_DIR = Path("./outputs")
ROUTING_SIGNALS_PATH = OUTPUT_DIR / "routing_signals.json"
ALPHA_SWEEP_RESULTS_PATH = OUTPUT_DIR / "alpha_sweep_results.csv"
LAYER_REPLACEMENT_PATH = OUTPUT_DIR / "layer_replacement_results.csv"
GATED_HYBRID_RESULTS_PATH = OUTPUT_DIR / "gated_hybrid_results.csv"


def get_target_layer_indices(model) -> list:
    """
    Get layer indices based on TARGET_LAYERS configuration.

    Args:
        model: The loaded model with model.model.layers

    Returns:
        List of layer indices to modify
    """
    num_layers = len(model.model.layers)

    if isinstance(TARGET_LAYERS, list):
        return TARGET_LAYERS
    elif TARGET_LAYERS == "all":
        return list(range(num_layers))
    elif TARGET_LAYERS == "middle":
        # Middle third of layers
        start = num_layers // 3
        end = 2 * num_layers // 3
        return list(range(start, end))
    elif TARGET_LAYERS == "first_half":
        return list(range(num_layers // 2))
    elif TARGET_LAYERS == "last_half":
        return list(range(num_layers // 2, num_layers))
    else:
        raise ValueError(f"Unknown TARGET_LAYERS: {TARGET_LAYERS}")


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
