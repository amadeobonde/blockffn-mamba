#!/usr/bin/env python3
"""
Automated Policy Validation Runner

Runs policy validation across presets and generates comprehensive reports.
This is the main entry point for validating adaptive inference policies.

Usage:
    python scripts/run_policy_validation.py --presets fast balanced conservative --mode thorough
    python scripts/run_policy_validation.py --output-dir results/policy_validation/
"""

import argparse
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.policy_validator import PolicyValidator, PromptSource
from scripts.policy_report import print_comparison_report


def run_validation(
    presets: List[str],
    mode: str = "quick",
    source: str = "synthetic",
    prompts_file: Optional[str] = None,
    device: str = "mps",
    max_tokens: int = 50,
    model_name: str = "SparseLLM/BlockFFN-3B-SFT",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run policy validation across presets.

    Args:
        presets: List of preset names to validate
        mode: "quick" (10 prompts) or "thorough" (100 prompts)
        source: Prompt source ("synthetic", "wikitext", "file")
        prompts_file: Path to prompts file if source="file"
        device: Target device
        max_tokens: Maximum tokens to generate per prompt
        model_name: HuggingFace model name
        verbose: Enable verbose output

    Returns:
        Dictionary with validation results
    """
    # Determine prompt count
    prompt_count = 10 if mode == "quick" else 100

    # Load prompts
    if source == "file" and prompts_file:
        prompts = PromptSource.load_from_file(prompts_file)
    elif source == "wikitext":
        prompts = PromptSource.get_wikitext_prompts(prompt_count)
    else:
        prompts = PromptSource.get_synthetic_prompts(prompt_count)

    print(f"Loaded {len(prompts)} prompts from {source}")
    print(f"Presets to validate: {presets}")
    print(f"Device: {device}")
    print(f"Max tokens per prompt: {max_tokens}")
    print()

    # Run validation
    validator = PolicyValidator(
        model_name=model_name,
        device=device,
        verbose=verbose,
    )

    results = validator.run(presets, prompts, max_tokens)

    return results


def generate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of validation results.

    Args:
        results: Raw validation results

    Returns:
        Summary dictionary with key metrics
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "presets_tested": list(results.keys()),
        "recommendations": {},
    }

    # Find best preset for each metric
    metrics = {
        "lowest_escalation_rate": ("avg_escalation_rate", False),
        "highest_green_rate": ("avg_pct_green", True),
        "longest_streaks": ("avg_accepted_streak_length", True),
        "fastest": ("avg_tokens_per_second", True),
        "most_efficient": ("avg_pct_small_window", True),
    }

    for metric_name, (key, higher_is_better) in metrics.items():
        best_preset = None
        best_value = None

        for preset_name, preset_results in results.items():
            value = getattr(preset_results, key, None)
            if value is None:
                continue

            if best_value is None:
                best_preset = preset_name
                best_value = value
            elif higher_is_better and value > best_value:
                best_preset = preset_name
                best_value = value
            elif not higher_is_better and value < best_value:
                best_preset = preset_name
                best_value = value

        if best_preset:
            summary["recommendations"][metric_name] = {
                "preset": best_preset,
                "value": best_value,
            }

    # Overall recommendation
    # Balance between escalation rate and green rate
    scores = {}
    for preset_name, preset_results in results.items():
        esc_rate = preset_results.avg_escalation_rate
        green_rate = preset_results.avg_pct_green

        # Higher green rate is good, lower escalation rate is good
        # Score = green_rate - escalation_rate (higher is better)
        scores[preset_name] = green_rate - esc_rate

    if scores:
        best_overall = max(scores.items(), key=lambda x: x[1])
        summary["recommendations"]["best_overall"] = {
            "preset": best_overall[0],
            "score": best_overall[1],
            "reason": "Highest (green_rate - escalation_rate)",
        }

    return summary


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    include_summary: bool = True,
) -> Dict[str, str]:
    """
    Save validation results to files.

    Args:
        results: Validation results
        output_dir: Output directory
        include_summary: Whether to generate summary

    Returns:
        Dictionary of output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Save JSON results
    json_path = output_path / "validation_results.json"
    serializable = {}
    for k, v in results.items():
        d = asdict(v)
        serializable[k] = d

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    output_files["json"] = str(json_path)
    print(f"JSON results saved to: {json_path}")

    # Save CSV summary
    csv_path = output_path / "validation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "preset", "prompts", "tokens",
            "pct_small", "pct_medium", "pct_full",
            "pct_green", "pct_yellow", "pct_red",
            "escalation_rate", "suffix_escalations",
            "avg_streak", "max_streak",
            "avg_confidence", "tok_per_sec"
        ])
        for name, r in results.items():
            writer.writerow([
                name, r.total_prompts, r.total_tokens,
                f"{r.avg_pct_small_window:.3f}",
                f"{r.avg_pct_medium_window:.3f}",
                f"{r.avg_pct_full_window:.3f}",
                f"{r.avg_pct_green:.3f}",
                f"{r.avg_pct_yellow:.3f}",
                f"{r.avg_pct_red:.3f}",
                f"{r.avg_escalation_rate:.3f}",
                r.total_suffix_escalations,
                f"{r.avg_accepted_streak_length:.1f}",
                r.max_accepted_streak_observed,
                f"{r.avg_mean_confidence:.3f}",
                f"{r.avg_tokens_per_second:.1f}",
            ])
    output_files["csv"] = str(csv_path)
    print(f"CSV summary saved to: {csv_path}")

    # Save summary with recommendations
    if include_summary:
        summary = generate_summary(results)
        summary_path = output_path / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        output_files["summary"] = str(summary_path)
        print(f"Summary saved to: {summary_path}")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Automated Policy Validation Runner"
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["fast", "balanced", "conservative"],
        help="Presets to validate"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "thorough"],
        default="quick",
        help="quick=10 prompts, thorough=100 prompts"
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "wikitext", "file"],
        default="synthetic",
        help="Prompt source"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to prompts file (if source=file)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="mps",
        help="Target device"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per prompt"
    )
    parser.add_argument(
        "--model",
        default="SparseLLM/BlockFFN-3B-SFT",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/policy_validation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip printing comparison report"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BLOCKFFN POLICY VALIDATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run validation
    results = run_validation(
        presets=args.presets,
        mode=args.mode,
        source=args.source,
        prompts_file=args.prompts_file,
        device=args.device,
        max_tokens=args.max_tokens,
        model_name=args.model,
        verbose=args.verbose,
    )

    # Save results
    output_files = save_results(results, args.output_dir)

    # Print comparison report
    if not args.no_report:
        print()
        print_comparison_report(results)

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}/")
    for name, path in output_files.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
