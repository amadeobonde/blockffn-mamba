#!/usr/bin/env python3
"""
Script 06: Replace Attention Layers with Mamba

Tests replacing attention entirely (alpha=1.0) in individual layers to find
which layers are most/least replaceable. This gives a floor for how much
attention matters in each layer.

Usage:
    python scripts/06_replace_layer.py
    python scripts/06_replace_layer.py --layers 10 11 12 13 14
"""

import argparse
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
    MAX_EVAL_SAMPLES,
    MAX_SEQ_LEN,
    MAMBA_D_STATE,
    MAMBA_D_CONV,
    MAMBA_EXPAND,
    MAMBA_HEADDIM,
    ensure_output_dir,
    LAYER_REPLACEMENT_PATH,
)
from src.mamba_sidecar import (
    attach_mamba_sidecar,
    detach_mamba_sidecars,
    set_all_alphas,
    get_hybrid_layers,
)
from src.eval_utils import compute_perplexity, save_results_csv


def main():
    parser = argparse.ArgumentParser(
        description="Test replacing attention with Mamba in individual layers"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to test (default: sample from all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to run on (default: {DEVICE})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples for perplexity (default: 50 for speed)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Layer Replacement Experiment")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {args.device}")
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

    num_layers = len(model.model.layers)
    print(f"Total layers: {num_layers}")

    # Determine which layers to test
    if args.layers:
        layers_to_test = args.layers
    else:
        # Sample layers: first, middle, last
        layers_to_test = [
            0,  # First
            num_layers // 4,  # Quarter
            num_layers // 2,  # Middle
            3 * num_layers // 4,  # Three-quarter
            num_layers - 1,  # Last
        ]
        # Add a few more in the middle where most processing happens
        mid_start = num_layers // 3
        mid_end = 2 * num_layers // 3
        for i in range(mid_start, mid_end, 2):
            if i not in layers_to_test:
                layers_to_test.append(i)
        layers_to_test = sorted(set(layers_to_test))

    print(f"Layers to test: {layers_to_test}")

    # Compute baseline
    print("\n" + "=" * 60)
    print("COMPUTING BASELINE")
    print("=" * 60)

    baseline_ppl = compute_perplexity(
        model, tokenizer,
        max_samples=args.max_samples,
        max_length=MAX_SEQ_LEN,
    )
    print(f"Baseline perplexity: {baseline_ppl['perplexity']:.2f}")

    # Test each layer
    print("\n" + "=" * 60)
    print("TESTING SINGLE LAYER REPLACEMENTS")
    print("=" * 60)

    mamba_config = {
        "d_model": model.config.hidden_size,
        "d_state": MAMBA_D_STATE,
        "d_conv": MAMBA_D_CONV,
        "expand": MAMBA_EXPAND,
        "headdim": MAMBA_HEADDIM,
    }

    results = [{"layer": "baseline", "perplexity": baseline_ppl["perplexity"], "delta": 0.0}]

    for layer_idx in tqdm(layers_to_test, desc="Testing layers"):
        print(f"\n--- Layer {layer_idx} ---")

        # Fresh model load would be cleanest but slow
        # Instead, attach to just this layer, test, then detach

        # Attach sidecar to just this layer
        model = attach_mamba_sidecar(
            model,
            layer_indices=[layer_idx],
            alpha=1.0,  # Full replacement
            gate_mode="fixed",
            mamba_config=mamba_config,
        )

        # Verify attachment
        hybrid_layers = get_hybrid_layers(model)
        assert len(hybrid_layers) == 1, f"Expected 1 hybrid layer, got {len(hybrid_layers)}"

        # Compute perplexity
        ppl = compute_perplexity(
            model, tokenizer,
            max_samples=args.max_samples,
            max_length=MAX_SEQ_LEN,
            show_progress=False,
        )

        delta = ppl["perplexity"] - baseline_ppl["perplexity"]
        pct_change = (delta / baseline_ppl["perplexity"]) * 100

        print(f"  Perplexity: {ppl['perplexity']:.2f} (delta: {delta:+.2f}, {pct_change:+.1f}%)")

        results.append({
            "layer": layer_idx,
            "perplexity": ppl["perplexity"],
            "delta": delta,
            "pct_change": pct_change,
        })

        # Detach sidecar
        model = detach_mamba_sidecars(model)

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nBaseline perplexity: {baseline_ppl['perplexity']:.2f}")
    print()
    print(f"{'Layer':<8} {'Perplexity':<12} {'Delta':<10} {'Change':<10}")
    print("-" * 45)

    layer_results = [r for r in results if r["layer"] != "baseline"]

    for r in sorted(layer_results, key=lambda x: x["delta"]):
        layer = r["layer"]
        ppl = r["perplexity"]
        delta = r["delta"]
        pct = r.get("pct_change", 0)
        print(f"{layer:<8} {ppl:<12.2f} {delta:+<10.2f} {pct:+.1f}%")

    # Find most and least replaceable
    if layer_results:
        most_replaceable = min(layer_results, key=lambda x: x["delta"])
        least_replaceable = max(layer_results, key=lambda x: x["delta"])

        print(f"\nMost replaceable layer: {most_replaceable['layer']}")
        print(f"  Perplexity: {most_replaceable['perplexity']:.2f}")
        print(f"  Delta: {most_replaceable['delta']:+.2f}")

        print(f"\nLeast replaceable layer: {least_replaceable['layer']}")
        print(f"  Perplexity: {least_replaceable['perplexity']:.2f}")
        print(f"  Delta: {least_replaceable['delta']:+.2f}")

        # Identify "safe" layers for replacement
        threshold = baseline_ppl["perplexity"] * 0.05  # 5% increase
        safe_layers = [r for r in layer_results if r["delta"] < threshold]
        print(f"\nLayers with <5% perplexity increase: {[r['layer'] for r in safe_layers]}")

    # Test replacing multiple layers
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE LAYER REPLACEMENT")
    print("=" * 60)

    # Find the safest layers
    if layer_results:
        sorted_by_impact = sorted(layer_results, key=lambda x: x["delta"])
        safest_layers = [r["layer"] for r in sorted_by_impact[:3]]

        print(f"\nReplacing safest 3 layers: {safest_layers}")

        model = attach_mamba_sidecar(
            model,
            layer_indices=safest_layers,
            alpha=1.0,
            gate_mode="fixed",
            mamba_config=mamba_config,
        )

        ppl = compute_perplexity(
            model, tokenizer,
            max_samples=args.max_samples,
            max_length=MAX_SEQ_LEN,
        )

        delta = ppl["perplexity"] - baseline_ppl["perplexity"]
        pct = (delta / baseline_ppl["perplexity"]) * 100

        print(f"  Combined perplexity: {ppl['perplexity']:.2f}")
        print(f"  Delta: {delta:+.2f} ({pct:+.1f}%)")

        results.append({
            "layer": f"combined_{safest_layers}",
            "perplexity": ppl["perplexity"],
            "delta": delta,
            "pct_change": pct,
        })

        model = detach_mamba_sidecars(model)

    # Save results
    ensure_output_dir()
    save_results_csv(results, str(LAYER_REPLACEMENT_PATH))

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {LAYER_REPLACEMENT_PATH}")
    print("\nNext step: Run 07_gated_hybrid.py for routing-based gating")
    print("=" * 60)


if __name__ == "__main__":
    main()
