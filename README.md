# BlockFFN + Mamba Hybrid Prototype

A research prototype that attaches Mamba state-space model sidecars to BlockFFN-3B-SFT, testing whether the model's internal routing signal can dynamically gate between cheap sequence mixing (Mamba) and expensive sequence mixing (attention).

## Hypothesis

BlockFFN uses a ReLU + RMSNorm router that produces sparse activation patterns. Tokens that activate few experts (high sparsity) may be "easy" and suitable for processing with Mamba. Tokens that activate many experts (low sparsity) are "important" and should use full attention.

**Core Question**: Does semantic routing outperform uniform mixing?

## Hardware Requirements

- **Development**: MacBook (CPU only) - for code editing and structure testing
- **Experiments**: Windows/WSL2 with NVIDIA RTX 3060 12GB VRAM
  - BlockFFN-3B in float16: ~6GB
  - Mamba sidecar: ~0.5-1GB
  - Working memory: ~3-4GB buffer

## Quick Start

### 1. Environment Setup

```bash
# On WSL2/Linux with NVIDIA GPU
cd blockffn-mamba
chmod +x setup_env.sh
./setup_env.sh

# Activate environment
conda activate blockffn-mamba
```

**Note**: `mamba_ssm` requires CUDA and will fail to install on Mac. The code includes a linear fallback for development on CPU.

### 2. Run Scripts in Order

```bash
# 1. Download model (~6GB)
python scripts/01_download_model.py

# 2. Inspect model architecture (find router location)
python scripts/02_inspect_model.py

# 3. Extract routing signals (understand sparsity patterns)
python scripts/03_extract_routing.py

# 4. Test Mamba sidecar (verify no-op at alpha=0)
python scripts/04_test_sidecar.py

# 5. Sweep alpha values (find quality/efficiency tradeoff)
python scripts/05_sweep_alpha.py

# 6. Test single-layer replacement (find replaceable layers)
python scripts/06_replace_layer.py

# 7. Run gated hybrid (test routing-based gating)
python scripts/07_gated_hybrid.py
```

### 3. Visualize Results

```bash
jupyter notebook notebooks/visualize_routing.ipynb
```

## Project Structure

```
blockffn-mamba/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup_env.sh              # One-shot environment setup
├── config.py                 # All hyperparameters and paths
├── scripts/
│   ├── 01_download_model.py  # Download BlockFFN-3B-SFT
│   ├── 02_inspect_model.py   # Print architecture, find router
│   ├── 03_extract_routing.py # Capture routing signals
│   ├── 04_test_sidecar.py    # Verify Mamba integration
│   ├── 05_sweep_alpha.py     # Sweep alpha, measure perplexity
│   ├── 06_replace_layer.py   # Test layer replacement
│   └── 07_gated_hybrid.py    # Routing-gated hybrid experiment
├── src/
│   ├── __init__.py
│   ├── mamba_sidecar.py      # MambaSidecar and GatedHybridLayer
│   ├── routing_extractor.py  # Capture post-ReLU routing signals
│   └── eval_utils.py         # Perplexity, generation, benchmarks
├── notebooks/
│   └── visualize_routing.ipynb  # Plot results
├── outputs/                  # Generated results (CSV, JSON)
└── cache/                    # Model cache directory
```

## Key Components

### Routing Extractor (`src/routing_extractor.py`)

Captures post-ReLU, pre-RMSNorm routing activations:

```python
from src.routing_extractor import RoutingExtractor

extractor = RoutingExtractor(model)
outputs = model(input_ids)  # Signals captured automatically
signals = extractor.get_routing_signals()
stats = extractor.compute_sparsity_stats()
```

### Mamba Sidecar (`src/mamba_sidecar.py`)

Attaches Mamba2 as a parallel path with gating:

```python
from src.mamba_sidecar import attach_mamba_sidecar, set_all_alphas

# Attach to middle layers with routing-based gating
model = attach_mamba_sidecar(
    model,
    layer_indices=[10, 11, 12, 13, 14],
    alpha=0.5,
    gate_mode="routing",  # Uses routing sparsity as gate
)

# Or use fixed alpha
set_all_alphas(model, alpha=0.3)
```

### Gate Modes

1. **fixed**: Constant alpha for all tokens (baseline)
2. **routing**: Uses routing sparsity as gate signal
   - High sparsity → use Mamba (cheap)
   - Low sparsity → use attention (expensive)
3. **learned**: Learns gate projection from hidden states

## Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "SparseLLM/BlockFFN-3B-SFT"
DEVICE = "cuda"  # or "cpu" for Mac
DTYPE = torch.float16

# Mamba settings
MAMBA_D_STATE = 64  # Reduced for 12GB GPU
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# Experiment settings
ALPHA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TARGET_LAYERS = "middle"  # or list of indices
GATE_MODE = "routing"
SPARSITY_THRESHOLD = 0.7
```

## Expected Results

After running all scripts:

| Metric | Baseline | α=0.3 | Routing-Gated |
|--------|----------|-------|---------------|
| Perplexity | ~15 | ~16 | ~15.5 |
| Memory (GB) | 6.0 | 7.0 | 7.0 |
| Tok/s | 100 | 95 | 90 |

**Success criteria**: If routing-gated perplexity is lower than fixed-alpha at the same compute cost, the hypothesis is supported.

## Troubleshooting

### Model won't load
```bash
pip install transformers>=4.40.0
```

### mamba_ssm won't install on Mac
Expected behavior. The code uses a linear fallback for CPU development.

### mamba_ssm won't install on WSL2
```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Install with no build isolation
pip install --no-build-isolation mamba-ssm
```

### Out of memory on RTX 3060
```python
# In config.py, enable 8-bit:
USE_8BIT_FALLBACK = True

# Or reduce sequence length:
MAX_SEQ_LEN = 256
```

### Router hooks capture nothing
The model structure may differ. Run `02_inspect_model.py --weights` and look for modules containing "router", "gate", or "expert".

## Citation

If you use this code in your research:

```bibtex
@misc{blockffn-mamba-hybrid,
  title={BlockFFN + Mamba Hybrid: Routing-Gated Sequence Mixing},
  year={2025},
  note={Research prototype}
}
```

### BlockFFN Paper
```bibtex
@article{song2025blockffn,
  title={BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts
         with Chunk-Level Activation Sparsity},
  author={Chenyang Song and others},
  journal={arXiv preprint arXiv:2507.08771},
  year={2025}
}
```

### Mamba Paper
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Albert Gu and Tri Dao},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## License

Apache 2.0 (following BlockFFN's license)
