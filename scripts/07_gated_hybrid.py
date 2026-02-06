#!/usr/bin/env python3
"""
Script 07: Gated Hybrid with Routing Signals

The main experiment: uses BlockFFN's routing signals to dynamically gate
between Mamba and attention. Tests whether semantic routing outperforms
uniform mixing.

Core hypothesis: Tokens the router considers "unimportant" (high sparsity)
should use Mamba; "important" tokens (low sparsity) should use attention.

Usage:
    python scripts/07_gated_hybrid.py
    python scripts/07_gated_hybrid.py --temperature 2.0
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
    TEST_PROMPTS,
    MAX_EVAL_SAMPLES,
    MAX_SEQ_LEN,
    MAX_NEW_TOKENS,
    MAMBA_D_STATE,
    MAMBA_D_CONV,
    MAMBA_EXPAND,
    MAMBA_HEADDIM,
    SPARSITY_THRESHOLD,
    GATE_TEMPERATURE,
    get_target_layer_indices,
    ensure_output_dir,
    GATED_HYBRID_RESULTS_PATH,
)
from src.routing_extractor import RoutingExtractor
from src.mamba_sidecar import (
    attach_mamba_sidecar,
    get_hybrid_layers,
    GatedHybridLayer,
)
from src.eval_utils import (
    compute_perplexity,
    generate_text,
    save_results_csv,
)


def wire_routing_to_gates(model, extractor):
    """
    Connect routing signal capture to hybrid layer gates.

    After each forward pass, the extractor captures routing signals.
    We then pass these to the hybrid layers for gate computation.
    """
    hybrid_layers = get_hybrid_layers(model)

    def update_gates():
        signals = extractor.get_routing_signals()
        for layer_idx, layer in hybrid_layers:
            if layer_idx in signals:
                layer.set_routing_signal(signals[layer_idx])

    return update_gates


class RoutingGatedModel:
    """
    Wrapper that automatically updates gates with routing signals.

    This is a convenience wrapper that ensures routing signals are
    extracted and passed to gates on each forward pass.
    """

    def __init__(self, model, tokenizer, target_layers):
        self.model = model
        self.tokenizer = tokenizer
        self.extractor = RoutingExtractor(model, target_layers)

    def forward_with_routing(self, input_ids):
        """Forward pass with automatic routing signal extraction."""
        # Clear previous signals
        self.extractor.clear()

        # Forward pass (extractor captures signals)
        outputs = self.model(input_ids)

        # Get signals and update gates
        signals = self.extractor.get_routing_signals()

        # Pass signals to hybrid layers
        for idx, layer in enumerate(self.model.model.layers):
            if isinstance(layer, GatedHybridLayer) and idx in signals:
                layer.set_routing_signal(signals[idx])

        return outputs

    def compute_perplexity_with_routing(self, max_samples=100, max_length=512):
        """
        Compute perplexity with routing-based gating.

        Note: This is a simplified version that doesn't handle the sliding
        window approach perfectly, but gives a reasonable approximation.
        """
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        all_text = "\n\n".join(dataset["text"][:max_samples])

        encodings = self.tokenizer(all_text, return_tensors="pt")
        input_ids = encodings["input_ids"]
        device = next(self.model.parameters()).device

        seq_len = input_ids.size(1)
        stride = max_length // 2
        nlls = []

        self.model.train(False)
        with torch.no_grad():
            for begin_idx in tqdm(range(0, seq_len, stride), desc="Perplexity"):
                end_idx = min(begin_idx + max_length, seq_len)

                input_chunk = input_ids[:, begin_idx:end_idx].to(device)

                # Clear and capture routing signals
                self.extractor.clear()
                outputs = self.model(input_chunk, labels=input_chunk)

                # Update gates with captured signals (for next iteration)
                signals = self.extractor.get_routing_signals()
                for idx, layer in enumerate(self.model.model.layers):
                    if isinstance(layer, GatedHybridLayer) and idx in signals:
                        layer.set_routing_signal(signals[idx])

                target_len = end_idx - begin_idx
                nlls.append(outputs.loss.item() * target_len)

                if end_idx == seq_len:
                    break

        total_loss = sum(nlls) / (seq_len - 1)
        perplexity = torch.exp(torch.tensor(total_loss)).item()

        return {"perplexity": perplexity, "loss": total_loss}


def main():
    parser = argparse.ArgumentParser(
        description="Test routing-based gating for BlockFFN + Mamba hybrid"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=GATE_TEMPERATURE,
        help=f"Gate temperature (default: {GATE_TEMPERATURE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SPARSITY_THRESHOLD,
        help=f"Sparsity threshold for hard gating (default: {SPARSITY_THRESHOLD})",
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
    print("Routing-Gated Hybrid Experiment")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {args.device}")
    print(f"Temperature: {args.temperature}")
    print(f"Sparsity threshold: {args.threshold}")
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

    results = [{"mode": "baseline", "perplexity": baseline_ppl["perplexity"]}]

    # Set up hybrid model with routing-based gating
    print("\n" + "=" * 60)
    print("SETTING UP ROUTING-GATED HYBRID")
    print("=" * 60)

    target_layers = get_target_layer_indices(model)
    mamba_config = {
        "d_model": model.config.hidden_size,
        "d_state": MAMBA_D_STATE,
        "d_conv": MAMBA_D_CONV,
        "expand": MAMBA_EXPAND,
        "headdim": MAMBA_HEADDIM,
    }

    model = attach_mamba_sidecar(
        model,
        layer_indices=target_layers,
        alpha=0.5,  # Starting point (will be overridden by routing)
        gate_mode="routing",  # Use routing signals for gating
        mamba_config=mamba_config,
        sparsity_threshold=args.threshold,
        temperature=args.temperature,
    )

    # Create routing-gated wrapper
    gated_model = RoutingGatedModel(model, tokenizer, target_layers)

    # Test 1: Soft routing-based gating
    print("\n" + "=" * 60)
    print("TEST 1: SOFT ROUTING-BASED GATING")
    print("=" * 60)

    # Update gate mode for all hybrid layers
    for idx, layer in get_hybrid_layers(model):
        layer.gate_mode = "routing"
        layer.hard_gating = False
        layer.temperature = args.temperature

    routing_ppl = gated_model.compute_perplexity_with_routing(
        max_samples=args.max_samples,
        max_length=MAX_SEQ_LEN,
    )
    print(f"Routing-gated perplexity: {routing_ppl['perplexity']:.2f}")

    delta = routing_ppl["perplexity"] - baseline_ppl["perplexity"]
    pct = (delta / baseline_ppl["perplexity"]) * 100
    print(f"Delta from baseline: {delta:+.2f} ({pct:+.1f}%)")

    results.append({
        "mode": "routing_soft",
        "temperature": args.temperature,
        "perplexity": routing_ppl["perplexity"],
        "delta": delta,
        "pct_change": pct,
    })

    # Test 2: Hard routing-based gating
    print("\n" + "=" * 60)
    print("TEST 2: HARD ROUTING-BASED GATING")
    print("=" * 60)

    for idx, layer in get_hybrid_layers(model):
        layer.hard_gating = True
        layer.sparsity_threshold = args.threshold

    hard_ppl = gated_model.compute_perplexity_with_routing(
        max_samples=args.max_samples,
        max_length=MAX_SEQ_LEN,
    )
    print(f"Hard-gated perplexity: {hard_ppl['perplexity']:.2f}")

    delta = hard_ppl["perplexity"] - baseline_ppl["perplexity"]
    pct = (delta / baseline_ppl["perplexity"]) * 100
    print(f"Delta from baseline: {delta:+.2f} ({pct:+.1f}%)")

    results.append({
        "mode": "routing_hard",
        "threshold": args.threshold,
        "perplexity": hard_ppl["perplexity"],
        "delta": delta,
        "pct_change": pct,
    })

    # Test 3: Fixed alpha comparison
    print("\n" + "=" * 60)
    print("TEST 3: FIXED ALPHA COMPARISON")
    print("=" * 60)

    for alpha in [0.3, 0.5, 0.7]:
        for idx, layer in get_hybrid_layers(model):
            layer.gate_mode = "fixed"
            layer.set_alpha(alpha)

        fixed_ppl = compute_perplexity(
            model, tokenizer,
            max_samples=args.max_samples,
            max_length=MAX_SEQ_LEN,
            show_progress=False,
        )

        delta = fixed_ppl["perplexity"] - baseline_ppl["perplexity"]
        pct = (delta / baseline_ppl["perplexity"]) * 100

        print(f"Fixed alpha={alpha}: {fixed_ppl['perplexity']:.2f} (delta: {delta:+.2f}, {pct:+.1f}%)")

        results.append({
            "mode": f"fixed_{alpha}",
            "alpha": alpha,
            "perplexity": fixed_ppl["perplexity"],
            "delta": delta,
            "pct_change": pct,
        })

    # Test 4: Temperature sweep for routing
    print("\n" + "=" * 60)
    print("TEST 4: TEMPERATURE SWEEP")
    print("=" * 60)

    for temp in [0.5, 1.0, 2.0, 4.0]:
        for idx, layer in get_hybrid_layers(model):
            layer.gate_mode = "routing"
            layer.hard_gating = False
            layer.temperature = temp

        temp_ppl = gated_model.compute_perplexity_with_routing(
            max_samples=args.max_samples,
            max_length=MAX_SEQ_LEN,
        )

        delta = temp_ppl["perplexity"] - baseline_ppl["perplexity"]
        pct = (delta / baseline_ppl["perplexity"]) * 100

        print(f"Temperature={temp}: {temp_ppl['perplexity']:.2f} (delta: {delta:+.2f}, {pct:+.1f}%)")

        results.append({
            "mode": f"routing_temp_{temp}",
            "temperature": temp,
            "perplexity": temp_ppl["perplexity"],
            "delta": delta,
            "pct_change": pct,
        })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Mode':<20} {'Perplexity':<12} {'Delta':<10} {'Change':<10}")
    print("-" * 55)

    for r in results:
        mode = r["mode"]
        ppl = r["perplexity"]
        delta = r.get("delta", 0.0)
        pct = r.get("pct_change", 0.0)
        print(f"{mode:<20} {ppl:<12.2f} {delta:+<10.2f} {pct:+.1f}%")

    # Analysis
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST")
    print("=" * 60)

    baseline = baseline_ppl["perplexity"]

    routing_results = [r for r in results if r["mode"].startswith("routing")]
    fixed_results = [r for r in results if r["mode"].startswith("fixed")]

    if routing_results and fixed_results:
        best_routing = min(routing_results, key=lambda x: x["perplexity"])
        best_fixed = min(fixed_results, key=lambda x: x["perplexity"])

        print(f"\nBest routing-based: {best_routing['mode']}")
        print(f"  Perplexity: {best_routing['perplexity']:.2f}")

        print(f"\nBest fixed alpha: {best_fixed['mode']}")
        print(f"  Perplexity: {best_fixed['perplexity']:.2f}")

        if best_routing["perplexity"] < best_fixed["perplexity"]:
            improvement = best_fixed["perplexity"] - best_routing["perplexity"]
            print(f"\nHYPOTHESIS SUPPORTED: Routing-based gating outperforms fixed alpha")
            print(f"  Improvement: {improvement:.2f} perplexity points")
        else:
            print(f"\nHYPOTHESIS NOT SUPPORTED: Fixed alpha performs better")
            print("  Possible reasons:")
            print("  - Routing signal may not correlate well with importance")
            print("  - Temperature/threshold tuning needed")
            print("  - More complex gating function may be needed")

    # Generation samples
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)

    # Reset to best routing config
    if routing_results:
        best = min(routing_results, key=lambda x: x["perplexity"])
        if "temp" in best["mode"]:
            temp = best.get("temperature", 1.0)
        else:
            temp = args.temperature

        for idx, layer in get_hybrid_layers(model):
            layer.gate_mode = "routing"
            layer.hard_gating = False
            layer.temperature = temp

    for prompt in TEST_PROMPTS[:2]:
        print(f"\nPrompt: {prompt!r}")
        text = generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        print(f"Output: {text[len(prompt):]!r}")

    # Save results
    ensure_output_dir()
    save_results_csv(results, str(GATED_HYBRID_RESULTS_PATH))

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {GATED_HYBRID_RESULTS_PATH}")
    print("\nNext step: Visualize results in notebooks/visualize_routing.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
