#!/usr/bin/env python3
"""
Script 05: Sweep Alpha Values

Systematically tests different alpha values (0=attention, 1=mamba) and
measures perplexity, memory, and throughput at each setting. Generates
sample text to qualitatively assess quality.

Saves results to CSV for analysis.

Usage:
    python scripts/05_sweep_alpha.py
    python scripts/05_sweep_alpha.py --alphas 0.0 0.1 0.2 0.3 0.5
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
    ALPHA_SWEEP,
    MAX_EVAL_SAMPLES,
    MAX_SEQ_LEN,
    MAX_NEW_TOKENS,
    MAMBA_D_STATE,
    MAMBA_D_CONV,
    MAMBA_EXPAND,
    MAMBA_HEADDIM,
    get_target_layer_indices,
    ensure_output_dir,
    ALPHA_SWEEP_RESULTS_PATH,
)
from src.mamba_sidecar import attach_mamba_sidecar, set_all_alphas
from src.eval_utils import (
    compute_perplexity,
    generate_text,
    MemoryTracker,
    save_results_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Sweep alpha values for hybrid model")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=ALPHA_SWEEP,
        help=f"Alpha values to test (default: {ALPHA_SWEEP})",
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
        default=MAX_EVAL_SAMPLES,
        help=f"Max samples for perplexity (default: {MAX_EVAL_SAMPLES})",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity computation (faster)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip text generation (faster)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Alpha Sweep for BlockFFN + Mamba Hybrid")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {args.device}")
    print(f"Alpha values: {args.alphas}")
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

    # Compute baseline (no Mamba)
    print("\n" + "=" * 60)
    print("COMPUTING BASELINE (No Mamba)")
    print("=" * 60)

    baseline_results = {"alpha": "baseline"}

    if not args.skip_perplexity:
        print("\nComputing baseline perplexity...")
        baseline_ppl = compute_perplexity(
            model, tokenizer,
            max_samples=args.max_samples,
            max_length=MAX_SEQ_LEN,
        )
        baseline_results["perplexity"] = baseline_ppl["perplexity"]
        print(f"  Baseline perplexity: {baseline_ppl['perplexity']:.2f}")
    else:
        baseline_results["perplexity"] = None

    if not args.skip_generation:
        print("\nGenerating baseline samples...")
        for i, prompt in enumerate(TEST_PROMPTS[:2]):
            text = generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
            baseline_results[f"generation_{i}"] = text
            print(f"\n  Prompt: {prompt!r}")
            print(f"  Output: {text[len(prompt):]!r}")

    # Attach Mamba sidecars
    print("\n" + "=" * 60)
    print("ATTACHING MAMBA SIDECARS")
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
        alpha=0.0,  # Start with alpha=0
        gate_mode="fixed",
        mamba_config=mamba_config,
    )

    # Sweep alpha values
    print("\n" + "=" * 60)
    print("SWEEPING ALPHA VALUES")
    print("=" * 60)

    results = [baseline_results]
    tracker = MemoryTracker()

    for alpha in tqdm(args.alphas, desc="Alpha sweep"):
        print(f"\n--- Alpha = {alpha} ---")

        # Set alpha for all hybrid layers
        set_all_alphas(model, alpha)

        result = {"alpha": alpha}
        tracker.reset_peak()
        tracker.snapshot(f"alpha_{alpha}_start")

        # Compute perplexity
        if not args.skip_perplexity:
            ppl = compute_perplexity(
                model, tokenizer,
                max_samples=args.max_samples,
                max_length=MAX_SEQ_LEN,
                show_progress=False,
            )
            result["perplexity"] = ppl["perplexity"]
            print(f"  Perplexity: {ppl['perplexity']:.2f}")
        else:
            result["perplexity"] = None

        tracker.snapshot(f"alpha_{alpha}_ppl_done")

        # Memory usage
        if torch.cuda.is_available():
            result["memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak memory: {result['memory_gb']:.2f} GB")
        else:
            result["memory_gb"] = None

        # Generate text samples
        if not args.skip_generation:
            for i, prompt in enumerate(TEST_PROMPTS[:2]):
                text = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                result[f"generation_{i}"] = text

                if alpha in [0.0, 0.5, 1.0]:  # Only print for key alphas
                    print(f"\n  Prompt: {prompt!r}")
                    print(f"  Output: {text[len(prompt):]!r}")

        # Throughput (quick test)
        import time
        test_input = tokenizer(TEST_PROMPTS[0], return_tensors="pt")
        input_ids = test_input["input_ids"].to(next(model.parameters()).device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time
        times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        seq_len = input_ids.shape[1]
        result["tokens_per_second"] = seq_len / avg_time
        print(f"  Throughput: {result['tokens_per_second']:.1f} tok/s")

        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    print(f"\n{'Alpha':<10} {'Perplexity':<12} {'Memory (GB)':<12} {'Tok/s':<10}")
    print("-" * 50)

    for r in results:
        alpha = r["alpha"]
        ppl = r.get("perplexity")
        mem = r.get("memory_gb")
        tps = r.get("tokens_per_second")

        ppl_str = f"{ppl:.2f}" if ppl else "N/A"
        mem_str = f"{mem:.2f}" if mem else "N/A"
        tps_str = f"{tps:.1f}" if tps else "N/A"

        print(f"{str(alpha):<10} {ppl_str:<12} {mem_str:<12} {tps_str:<10}")

    # Save results
    ensure_output_dir()
    save_results_csv(results, str(ALPHA_SWEEP_RESULTS_PATH))

    # Analysis
    if not args.skip_perplexity:
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        baseline_ppl = results[0].get("perplexity")
        if baseline_ppl:
            print(f"\nBaseline perplexity: {baseline_ppl:.2f}")

            for r in results[1:]:
                alpha = r["alpha"]
                ppl = r.get("perplexity")
                if ppl:
                    delta = ppl - baseline_ppl
                    pct = (delta / baseline_ppl) * 100
                    direction = "+" if delta > 0 else ""
                    print(f"  Alpha {alpha}: {ppl:.2f} ({direction}{delta:.2f}, {direction}{pct:.1f}%)")

            # Find best non-baseline alpha
            valid_results = [r for r in results[1:] if r.get("perplexity")]
            if valid_results:
                best = min(valid_results, key=lambda x: x["perplexity"])
                print(f"\nBest alpha: {best['alpha']} (perplexity: {best['perplexity']:.2f})")

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"Results saved to: {ALPHA_SWEEP_RESULTS_PATH}")
    print("\nNext step: Run 06_replace_layer.py to test full layer replacement")
    print("=" * 60)


if __name__ == "__main__":
    main()
