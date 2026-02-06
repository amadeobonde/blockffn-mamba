#!/usr/bin/env python3
"""
Script 04: Test Mamba Sidecar Integration

Attaches Mamba sidecars to the model with alpha=0 (no Mamba contribution)
and verifies that outputs match the original model exactly. This confirms
the integration doesn't break anything before we start blending.

Also measures memory and timing impact of the sidecar attachment.

Usage:
    python scripts/04_test_sidecar.py
    python scripts/04_test_sidecar.py --alpha 0.1  # Test with small Mamba contribution
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from copy import deepcopy

from config import (
    MODEL_NAME,
    MODEL_CACHE_DIR,
    DEVICE,
    DTYPE,
    TEST_PROMPTS,
    MAMBA_D_STATE,
    MAMBA_D_CONV,
    MAMBA_EXPAND,
    MAMBA_HEADDIM,
    get_target_layer_indices,
)
from src.mamba_sidecar import (
    MambaSidecar,
    GatedHybridLayer,
    attach_mamba_sidecar,
    detach_mamba_sidecars,
    set_all_alphas,
)
from src.eval_utils import MemoryTracker, compare_outputs


def main():
    parser = argparse.ArgumentParser(description="Test Mamba sidecar integration")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Gate alpha value (0=pure attention, 1=pure mamba)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to run on (default: {DEVICE})",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip output comparison (faster)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Mamba Sidecar Integration Test")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {args.device}")
    print(f"Alpha: {args.alpha}")
    print()

    # Memory tracking
    tracker = MemoryTracker()
    tracker.snapshot("start")

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

    tracker.snapshot("model_loaded")
    print(f"Model loaded on: {next(model.parameters()).device}")

    # Get original outputs for comparison
    print("\nGetting baseline outputs...")
    test_prompts = TEST_PROMPTS[:3]  # Use first 3 for speed

    baseline_outputs = {}
    model.train(False)
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(next(model.parameters()).device)
            outputs = model(input_ids)
            baseline_outputs[prompt] = outputs.logits.cpu().clone()

    tracker.snapshot("baseline_computed")

    # Attach Mamba sidecars
    print("\nAttaching Mamba sidecars...")
    target_layers = get_target_layer_indices(model)
    print(f"Target layers: {target_layers}")

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
        alpha=args.alpha,
        gate_mode="fixed",
        mamba_config=mamba_config,
    )

    tracker.snapshot("sidecar_attached")

    # Check which layers now have sidecars
    hybrid_count = sum(
        1 for layer in model.model.layers
        if isinstance(layer, GatedHybridLayer)
    )
    print(f"\nHybrid layers attached: {hybrid_count}")

    # Test forward pass with sidecar
    print("\nTesting forward pass with sidecar...")
    hybrid_outputs = {}
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(next(model.parameters()).device)
            outputs = model(input_ids)
            hybrid_outputs[prompt] = outputs.logits.cpu().clone()

    tracker.snapshot("hybrid_computed")

    # Compare outputs
    if not args.skip_comparison:
        print("\n" + "=" * 60)
        print("OUTPUT COMPARISON (alpha=0 should match exactly)")
        print("=" * 60)

        all_close = True
        max_diff_overall = 0.0

        for prompt in test_prompts:
            baseline = baseline_outputs[prompt]
            hybrid = hybrid_outputs[prompt]

            # For alpha=0, outputs should be identical
            # For alpha>0, they will differ
            if args.alpha == 0.0:
                is_close = torch.allclose(baseline, hybrid, atol=1e-4, rtol=1e-3)
            else:
                is_close = True  # Don't expect match for alpha > 0

            max_diff = (baseline - hybrid).abs().max().item()
            max_diff_overall = max(max_diff_overall, max_diff)

            status = "PASS" if is_close else "FAIL"
            print(f"\n  Prompt: {prompt[:40]}...")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Status: {status}")

            if not is_close:
                all_close = False

        print(f"\n  Overall max difference: {max_diff_overall:.6f}")

        if args.alpha == 0.0:
            if all_close:
                print("\n  RESULT: All outputs match baseline with alpha=0")
                print("  Mamba sidecar integration is correct!")
            else:
                print("\n  WARNING: Outputs differ from baseline with alpha=0")
                print("  This may indicate a bug in the sidecar integration.")
        else:
            print(f"\n  NOTE: With alpha={args.alpha}, outputs are expected to differ")

    # Memory report
    print("\n" + "=" * 60)
    print("MEMORY REPORT")
    print("=" * 60)
    print(tracker.report())

    memory_delta = tracker.get_delta("model_loaded", "sidecar_attached")
    print(f"\nMemory added by sidecar: {memory_delta:.2f} GB")

    # Timing test
    print("\n" + "=" * 60)
    print("TIMING TEST")
    print("=" * 60)

    import time

    # Warmup
    test_input = tokenizer(TEST_PROMPTS[0], return_tensors="pt")
    input_ids = test_input["input_ids"].to(next(model.parameters()).device)

    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time multiple runs
    n_runs = 10
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    seq_len = input_ids.shape[1]
    tokens_per_sec = seq_len / avg_time

    print(f"  Sequence length: {seq_len}")
    print(f"  Average time: {avg_time * 1000:.2f} ms")
    print(f"  Tokens/second: {tokens_per_sec:.1f}")

    # Test alpha sweep (quick)
    print("\n" + "=" * 60)
    print("QUICK ALPHA SWEEP")
    print("=" * 60)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        set_all_alphas(model, alpha)
        with torch.no_grad():
            outputs = model(input_ids)
            # Get logits for last token
            last_logit = outputs.logits[0, -1, :10].tolist()
            last_logit_str = ", ".join(f"{x:.2f}" for x in last_logit)
        print(f"  alpha={alpha:.2f}: last_logits[:10] = [{last_logit_str}]")

    # Detach sidecars
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    print("Detaching Mamba sidecars...")
    model = detach_mamba_sidecars(model)

    # Verify original layers restored
    hybrid_count_after = sum(
        1 for layer in model.model.layers
        if isinstance(layer, GatedHybridLayer)
    )
    print(f"Hybrid layers remaining: {hybrid_count_after}")

    tracker.snapshot("cleanup_done")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNext step: Run 05_sweep_alpha.py to find optimal alpha values")


if __name__ == "__main__":
    main()
