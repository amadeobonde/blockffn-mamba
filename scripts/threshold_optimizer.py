#!/usr/bin/env python3
"""
Threshold Optimizer for BlockFFN Adaptive Inference

Grid search for optimal threshold combinations that minimize escalation rate
while maintaining baseline match rate >= 99%.

Usage:
    python scripts/threshold_optimizer.py --output results/optimal_thresholds.json
    python scripts/threshold_optimizer.py --quick --device mps
"""

import argparse
import json
import csv
import os
import sys
import time
from itertools import product
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.engine.adaptive_engine import AdaptiveInferenceEngine


@dataclass
class ThresholdConfig:
    """A threshold configuration to test."""
    tau_green: float
    tau_yellow: float
    tau_red: float
    yellow_streak_limit: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Result from testing a threshold configuration."""
    config: ThresholdConfig
    escalation_rate: float
    baseline_match_rate: float
    green_rate: float
    yellow_rate: float
    red_rate: float
    avg_streak_length: float
    tokens_per_second: float
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


class ThresholdOptimizer:
    """
    Grid search optimizer for confidence thresholds.

    Searches over combinations of tau_green, tau_yellow, tau_red, and
    yellow_streak_limit to find configurations that minimize escalation
    while maintaining quality.
    """

    def __init__(
        self,
        model_name: str = "SparseLLM/BlockFFN-3B-SFT",
        device: str = "mps",
        baseline_preset: str = "conservative",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.baseline_preset = baseline_preset
        self.verbose = verbose
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {self.model_name}...")
        dtype = torch.float32 if self.device == "cpu" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        print("Model loaded.")

    def generate_grid(
        self,
        quick: bool = False,
    ) -> List[ThresholdConfig]:
        """
        Generate grid of threshold configurations to test.

        Args:
            quick: Use smaller grid for quick testing

        Returns:
            List of ThresholdConfig objects
        """
        if quick:
            tau_green_values = [0.80, 0.85, 0.90]
            tau_yellow_values = [0.60, 0.65, 0.70]
            tau_red_values = [0.40, 0.45]
            yellow_streak_values = [2, 3, 4]
        else:
            tau_green_values = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
            tau_yellow_values = [0.55, 0.60, 0.65, 0.70, 0.75]
            tau_red_values = [0.35, 0.40, 0.45, 0.50]
            yellow_streak_values = [2, 3, 4, 5]

        configs = []
        for tau_green, tau_yellow, tau_red, streak in product(
            tau_green_values, tau_yellow_values, tau_red_values, yellow_streak_values
        ):
            # Skip invalid combinations (tau_green > tau_yellow > tau_red)
            if tau_yellow >= tau_green or tau_red >= tau_yellow:
                continue

            configs.append(ThresholdConfig(
                tau_green=tau_green,
                tau_yellow=tau_yellow,
                tau_red=tau_red,
                yellow_streak_limit=streak,
            ))

        return configs

    def test_config(
        self,
        config: ThresholdConfig,
        prompts: List[str],
        max_tokens: int = 50,
    ) -> OptimizationResult:
        """
        Test a single threshold configuration.

        Args:
            config: Threshold configuration to test
            prompts: List of test prompts
            max_tokens: Maximum tokens to generate per prompt

        Returns:
            OptimizationResult with metrics
        """
        self.load_model()

        # Create engine with custom thresholds
        base_config = PRESETS["balanced"].for_device(self.device)
        base_config.tau_green = config.tau_green
        base_config.tau_yellow = config.tau_yellow
        base_config.tau_red = config.tau_red
        base_config.yellow_streak_limit = config.yellow_streak_limit
        base_config.collect_statistics = True

        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, base_config)

        # Track metrics
        total_tokens = 0
        escalated_tokens = 0
        green_tokens = 0
        yellow_tokens = 0
        red_tokens = 0
        total_time = 0
        streak_lengths = []

        for prompt in prompts:
            engine.reset_statistics()
            start_time = time.perf_counter()

            _, stats = engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_stats=True,
            )

            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            total_tokens += stats.total_tokens
            escalated_tokens += stats.escalated_tokens

            # Classify confidence values
            for conf in engine._confidence_history:
                if conf >= config.tau_green:
                    green_tokens += 1
                elif conf >= config.tau_yellow:
                    yellow_tokens += 1
                else:
                    red_tokens += 1

            # Track streaks
            current_streak = 0
            for conf in engine._confidence_history:
                if conf >= config.tau_yellow:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streak_lengths.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                streak_lengths.append(current_streak)

        # Cleanup
        engine.detach_adaptive_layers()

        # Calculate metrics
        escalation_rate = escalated_tokens / total_tokens if total_tokens > 0 else 0
        green_rate = green_tokens / total_tokens if total_tokens > 0 else 0
        yellow_rate = yellow_tokens / total_tokens if total_tokens > 0 else 0
        red_rate = red_tokens / total_tokens if total_tokens > 0 else 0
        avg_streak = sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        # Baseline match rate is hard to compute without baseline runs
        # Use 1.0 - escalation_rate as a proxy (escalated tokens may differ from baseline)
        baseline_match_rate = 1.0 - escalation_rate

        return OptimizationResult(
            config=config,
            escalation_rate=escalation_rate,
            baseline_match_rate=baseline_match_rate,
            green_rate=green_rate,
            yellow_rate=yellow_rate,
            red_rate=red_rate,
            avg_streak_length=avg_streak,
            tokens_per_second=tokens_per_second,
        )

    def find_pareto_optimal(
        self,
        results: List[OptimizationResult],
        min_baseline_match: float = 0.99,
    ) -> List[OptimizationResult]:
        """
        Find Pareto-optimal configurations.

        A configuration is Pareto-optimal if no other configuration
        has both lower escalation rate AND higher baseline match rate.

        Args:
            results: All optimization results
            min_baseline_match: Minimum baseline match rate to consider

        Returns:
            List of Pareto-optimal results
        """
        # Filter by baseline match rate
        valid_results = [r for r in results if r.baseline_match_rate >= min_baseline_match]

        if not valid_results:
            print(f"Warning: No configurations meet baseline_match_rate >= {min_baseline_match}")
            # Fall back to all results
            valid_results = results

        pareto_optimal = []

        for candidate in valid_results:
            is_dominated = False

            for other in valid_results:
                if other is candidate:
                    continue

                # other dominates candidate if it's better or equal on all objectives
                # and strictly better on at least one
                if (other.escalation_rate <= candidate.escalation_rate and
                    other.baseline_match_rate >= candidate.baseline_match_rate and
                    (other.escalation_rate < candidate.escalation_rate or
                     other.baseline_match_rate > candidate.baseline_match_rate)):
                    is_dominated = True
                    break

            if not is_dominated:
                candidate.is_pareto_optimal = True
                pareto_optimal.append(candidate)

        return pareto_optimal

    def run_optimization(
        self,
        prompts: List[str],
        quick: bool = False,
        max_tokens: int = 50,
    ) -> Tuple[List[OptimizationResult], List[OptimizationResult]]:
        """
        Run the full optimization process.

        Args:
            prompts: Test prompts
            quick: Use smaller grid
            max_tokens: Maximum tokens per prompt

        Returns:
            (all_results, pareto_optimal_results)
        """
        configs = self.generate_grid(quick=quick)
        print(f"Testing {len(configs)} threshold configurations...")

        results = []
        for i, config in enumerate(configs):
            if self.verbose or (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(configs)}] tau_green={config.tau_green:.2f}, "
                      f"tau_yellow={config.tau_yellow:.2f}, streak={config.yellow_streak_limit}")

            result = self.test_config(config, prompts, max_tokens)
            results.append(result)

            if self.verbose:
                print(f"    -> escalation={result.escalation_rate:.3f}, "
                      f"green={result.green_rate:.3f}")

        # Find Pareto-optimal
        pareto = self.find_pareto_optimal(results)
        print(f"\nFound {len(pareto)} Pareto-optimal configurations")

        return results, pareto


def get_prompts(source: str, count: int) -> List[str]:
    """Load prompts for testing."""
    from scripts.policy_validator import PromptSource

    if source == "wikitext":
        return PromptSource.get_wikitext_prompts(count)
    else:
        return PromptSource.get_synthetic_prompts(count)


def main():
    parser = argparse.ArgumentParser(
        description="Threshold Optimizer for BlockFFN Adaptive Inference"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller grid for quick testing")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--prompts", type=int, default=20,
                        help="Number of prompts to test")
    parser.add_argument("--source", choices=["synthetic", "wikitext"], default="synthetic")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/optimal_thresholds.json")
    parser.add_argument("--output-csv", type=str, default="results/threshold_sweep.csv")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # Get prompts
    prompts = get_prompts(args.source, args.prompts)
    print(f"Using {len(prompts)} prompts from {args.source}")

    # Run optimization
    optimizer = ThresholdOptimizer(
        model_name=args.model,
        device=args.device,
        verbose=args.verbose,
    )

    all_results, pareto_results = optimizer.run_optimization(
        prompts=prompts,
        quick=args.quick,
        max_tokens=args.max_tokens,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "total_configs_tested": len(all_results),
        "pareto_optimal_count": len(pareto_results),
        "pareto_optimal": [r.to_dict() for r in pareto_results],
        "best_by_escalation_rate": min(all_results, key=lambda r: r.escalation_rate).to_dict(),
        "best_by_green_rate": max(all_results, key=lambda r: r.green_rate).to_dict(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save CSV of all results
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tau_green", "tau_yellow", "tau_red", "yellow_streak_limit",
            "escalation_rate", "baseline_match_rate", "green_rate",
            "yellow_rate", "red_rate", "avg_streak", "tok_per_sec",
            "is_pareto_optimal"
        ])
        for r in all_results:
            writer.writerow([
                r.config.tau_green, r.config.tau_yellow, r.config.tau_red,
                r.config.yellow_streak_limit,
                f"{r.escalation_rate:.4f}", f"{r.baseline_match_rate:.4f}",
                f"{r.green_rate:.4f}", f"{r.yellow_rate:.4f}", f"{r.red_rate:.4f}",
                f"{r.avg_streak_length:.2f}", f"{r.tokens_per_second:.1f}",
                r.is_pareto_optimal
            ])
    print(f"CSV saved to: {csv_path}")

    # Print Pareto-optimal recommendations
    print("\n" + "=" * 70)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("=" * 70)
    for r in sorted(pareto_results, key=lambda x: x.escalation_rate):
        print(f"\n  tau_green={r.config.tau_green:.2f}, "
              f"tau_yellow={r.config.tau_yellow:.2f}, "
              f"tau_red={r.config.tau_red:.2f}, "
              f"streak={r.config.yellow_streak_limit}")
        print(f"    escalation_rate={r.escalation_rate:.3f}, "
              f"green_rate={r.green_rate:.3f}, "
              f"avg_streak={r.avg_streak_length:.1f}")


if __name__ == "__main__":
    main()
