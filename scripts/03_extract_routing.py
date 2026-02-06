#!/usr/bin/env python3
"""
Script 03: Extract Routing Signals

Runs test prompts through BlockFFN and captures the router activation values
(post-ReLU, pre-RMSNorm). Analyzes sparsity patterns and saves results for
visualization.

This validates whether the routing signal correlates with semantic importance
and is suitable for gating between Mamba and attention.

Usage:
    python scripts/03_extract_routing.py
    python scripts/03_extract_routing.py --prompts "Custom prompt here"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from config import (
    MODEL_NAME,
    MODEL_CACHE_DIR,
    DEVICE,
    DTYPE,
    TEST_PROMPTS,
    ensure_output_dir,
    ROUTING_SIGNALS_PATH,
)
from src.routing_extractor import RoutingExtractor, extract_routing_from_prompt


def analyze_sparsity_per_token(tokens, sparsity_per_token, top_k=5):
    """Analyze which tokens have highest/lowest sparsity."""
    # sparsity_per_token: [seq_len]
    if sparsity_per_token is None:
        return

    sparsity = sparsity_per_token.numpy()

    print("\n  Token-level sparsity analysis:")
    print("  " + "-" * 50)

    # Most sparse tokens (least "important")
    sparse_indices = sparsity.argsort()[-top_k:][::-1]
    print(f"  Most sparse tokens (high sparsity = few active experts):")
    for idx in sparse_indices:
        if idx < len(tokens):
            print(f"    {idx:3d}: '{tokens[idx]:15s}' sparsity={sparsity[idx]:.3f}")

    # Least sparse tokens (most "important")
    dense_indices = sparsity.argsort()[:top_k]
    print(f"\n  Least sparse tokens (low sparsity = many active experts):")
    for idx in dense_indices:
        if idx < len(tokens):
            print(f"    {idx:3d}: '{tokens[idx]:15s}' sparsity={sparsity[idx]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Extract routing signals from BlockFFN")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts to analyze (default: use TEST_PROMPTS from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROUTING_SIGNALS_PATH),
        help="Output JSON file for routing signals",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to run on (default: {DEVICE})",
    )
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else TEST_PROMPTS

    print("=" * 60)
    print("BlockFFN Routing Signal Extraction")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {args.device}")
    print(f"Prompts to analyze: {len(prompts)}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto" if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if args.device == "cpu":
        model = model.to("cpu")

    print(f"Model loaded on: {next(model.parameters()).device}")

    # Set up routing extractor
    print("\nSetting up routing extractor...")
    extractor = RoutingExtractor(model)

    # Results storage
    results = {
        "model_name": MODEL_NAME,
        "prompts": [],
    }

    # Process each prompt
    print("\n" + "=" * 60)
    print("EXTRACTING ROUTING SIGNALS")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{len(prompts)}]")
        print(f"  Text: {prompt!r}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        batch_size, seq_len = input_ids.shape

        print(f"  Tokens: {seq_len}")

        # Clear previous signals
        extractor.clear()

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)

        # Get routing signals
        signals = extractor.get_routing_signals_reshaped(batch_size, seq_len)
        stats = extractor.compute_sparsity_stats()
        importance = extractor.compute_importance_scores()

        # Get token strings
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        print(f"\n  Routing signals captured from {len(signals)} layers")
        print(f"  Overall sparsity: {stats['overall']:.3f}")

        # Per-layer sparsity
        print("\n  Per-layer sparsity:")
        for layer_idx in sorted(stats["per_layer"].keys())[:5]:
            sparsity = stats["per_layer"][layer_idx]
            print(f"    Layer {layer_idx:2d}: {sparsity:.3f}")
        if len(stats["per_layer"]) > 5:
            print(f"    ... ({len(stats['per_layer']) - 5} more layers)")

        # Token-level analysis
        analyze_sparsity_per_token(tokens, stats.get("per_token"))

        # Store results for this prompt
        prompt_result = {
            "prompt": prompt,
            "tokens": tokens,
            "seq_len": seq_len,
            "overall_sparsity": stats["overall"],
            "per_layer_sparsity": {
                str(k): v for k, v in stats["per_layer"].items()
            },
        }

        # Store signal shapes (not full tensors, too large)
        prompt_result["signal_shapes"] = {
            str(k): list(v.shape) for k, v in signals.items()
        }

        results["prompts"].append(prompt_result)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    overall_sparsities = [p["overall_sparsity"] for p in results["prompts"]]
    print(f"\nOverall sparsity across all prompts:")
    print(f"  Mean: {sum(overall_sparsities) / len(overall_sparsities):.3f}")
    print(f"  Min:  {min(overall_sparsities):.3f}")
    print(f"  Max:  {max(overall_sparsities):.3f}")

    # Interpretation
    mean_sparsity = sum(overall_sparsities) / len(overall_sparsities)
    print("\nInterpretation:")
    if mean_sparsity > 0.7:
        print("  HIGH SPARSITY: Most tokens activate few experts.")
        print("  This is good for Mamba integration - many 'easy' tokens.")
    elif mean_sparsity > 0.5:
        print("  MODERATE SPARSITY: Mixed token importance.")
        print("  Routing-based gating should show differentiation.")
    else:
        print("  LOW SPARSITY: Most tokens activate many experts.")
        print("  May indicate all tokens are 'important'.")

    # Save results
    ensure_output_dir()
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Clean up extractor
    extractor.restore_original()

    print("\n" + "=" * 60)
    print("Next step: Run 04_test_sidecar.py to test Mamba integration")
    print("=" * 60)


if __name__ == "__main__":
    main()
