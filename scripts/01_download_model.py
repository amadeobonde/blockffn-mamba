#!/usr/bin/env python3
"""
Script 01: Download BlockFFN-3B-SFT Model

Downloads the model and tokenizer from HuggingFace, saves to local cache,
and prints model configuration and parameter count.

Usage:
    python scripts/01_download_model.py
    python scripts/01_download_model.py --cache-dir ./my_cache
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_NAME, MODEL_CACHE_DIR


def count_parameters(model) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_billions": total / 1e9,
        "trainable_billions": trainable / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description="Download BlockFFN-3B-SFT model")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model name on HuggingFace (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(MODEL_CACHE_DIR),
        help=f"Local cache directory (default: {MODEL_CACHE_DIR})",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only download tokenizer, skip model weights",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BlockFFN-3B-SFT Model Download")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Cache: {args.cache_dir}")
    print()

    # Create cache directory
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Import here to show any import errors clearly
    print("Loading transformers library...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Download tokenizer
    print()
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Padding side: {tokenizer.padding_side}")
    print(f"  Special tokens: {tokenizer.special_tokens_map}")

    if args.skip_model:
        print()
        print("Skipping model download (--skip-model flag set)")
        print("Done!")
        return

    # Download model (weights only, don't load to GPU)
    print()
    print("Downloading model weights...")
    print("(This may take a while for a 3B parameter model)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,  # Don't duplicate weights in memory
    )

    # Print configuration
    print()
    print("Model Configuration:")
    print("-" * 40)
    config = model.config
    important_attrs = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
    ]
    for attr in important_attrs:
        if hasattr(config, attr):
            print(f"  {attr}: {getattr(config, attr)}")

    # Print parameter count
    print()
    print("Parameter Count:")
    print("-" * 40)
    params = count_parameters(model)
    print(f"  Total: {params['total']:,} ({params['total_billions']:.2f}B)")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_billions']:.2f}B)")

    # Print model architecture summary
    print()
    print("Model Architecture Summary:")
    print("-" * 40)
    print(f"  Type: {type(model).__name__}")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print(f"  Number of layers: {len(model.model.layers)}")
        if len(model.model.layers) > 0:
            layer = model.model.layers[0]
            print(f"  Layer type: {type(layer).__name__}")

    print()
    print("=" * 60)
    print("Download complete!")
    print(f"Model cached at: {args.cache_dir}")
    print()
    print("Next step: Run script 02_inspect_model.py to examine layer structure")
    print("=" * 60)


if __name__ == "__main__":
    main()
