#!/usr/bin/env python3
"""
Policy Validator for BlockFFN Adaptive Inference

Runs batches of prompts through different policy presets and
collects detailed statistics for policy comparison.

Usage:
    python scripts/policy_validator.py --mode quick --device mps
    python scripts/policy_validator.py --mode thorough --source wikitext --output-json results.json
    python scripts/policy_validator.py --preset conservative balanced aggressive --output-csv results.csv
"""

import argparse
import json
import csv
import time
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.engine.adaptive_engine import AdaptiveInferenceEngine, GenerationStats
from src.verification.decision import EscalationLevel


@dataclass
class PromptResult:
    """Results from a single prompt."""
    prompt_id: int
    prompt_preview: str  # First 50 chars
    total_tokens: int

    # Window distribution
    pct_small_window: float
    pct_medium_window: float
    pct_full_window: float

    # Classification rates
    pct_green: float
    pct_yellow: float
    pct_red: float

    # Escalation
    escalation_rate: float
    suffix_escalations: int
    token_escalations: int

    # Streaks
    mean_accepted_streak: float
    max_accepted_streak: int

    # Confidence
    mean_confidence: float
    min_confidence: float
    max_confidence: float

    # Timing
    generation_time_ms: float
    tokens_per_second: float


@dataclass
class PresetResults:
    """Aggregated results for a preset."""
    preset_name: str
    config_summary: Dict[str, Any]

    # Aggregates across all prompts
    total_prompts: int
    total_tokens: int

    # Window distribution (averages)
    avg_pct_small_window: float
    avg_pct_medium_window: float
    avg_pct_full_window: float

    # Classification rates (averages)
    avg_pct_green: float
    avg_pct_yellow: float
    avg_pct_red: float

    # Escalation
    avg_escalation_rate: float
    total_suffix_escalations: int
    avg_suffix_escalation_frequency: float  # per prompt

    # Streaks
    avg_accepted_streak_length: float
    max_accepted_streak_observed: int

    # Confidence distribution
    confidence_histogram: Dict[str, int]  # bucketed
    avg_mean_confidence: float

    # Performance
    avg_tokens_per_second: float

    # Per-prompt details
    prompt_results: List[PromptResult] = field(default_factory=list)


class PromptSource:
    """Prompt source utilities."""

    @staticmethod
    def get_synthetic_prompts(count: int) -> List[str]:
        """Generate synthetic prompts for testing."""
        prompts = [
            # Factual/simple
            "The capital of France is",
            "Water boils at",
            "The speed of light is approximately",
            "The largest planet in our solar system is",
            "Python was created by",

            # Reasoning
            "If all roses are flowers and some flowers fade quickly, then",
            "To solve this math problem, first we need to",
            "The best approach to debugging this code would be to",

            # Code
            "def fibonacci(n):\n    '''Calculate the nth Fibonacci number'''\n    if n <= 1:\n        return n\n",
            "import numpy as np\n\ndef matrix_multiply(A, B):\n",
            "class DatabaseConnection:\n    def __init__(self, host, port):\n",

            # Creative
            "Once upon a time in a distant galaxy,",
            "The old lighthouse keeper had a secret that",
            "In the depths of the ancient forest,",

            # Technical
            "In machine learning, gradient descent works by",
            "The transformer architecture uses attention mechanisms to",
            "Neural networks learn representations by",

            # Long context
            "The quick brown fox jumps over the lazy dog. " * 5 + "This classic sentence",
            "In the beginning, there was nothing but darkness. " * 3 + "Then came",
        ]

        # Extend if needed
        while len(prompts) < count:
            prompts.extend(prompts[:count - len(prompts)])

        return prompts[:count]

    @staticmethod
    def get_wikitext_prompts(count: int) -> List[str]:
        """Load prompts from WikiText dataset."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

            prompts = []
            for item in dataset:
                text = item["text"].strip()
                if len(text) > 50:  # Skip short entries
                    prompts.append(text[:500])  # Truncate long ones
                if len(prompts) >= count:
                    break

            return prompts
        except ImportError:
            print("Warning: 'datasets' package not installed. Using synthetic prompts.")
            return PromptSource.get_synthetic_prompts(count)

    @staticmethod
    def load_from_file(filepath: str) -> List[str]:
        """Load prompts from a text file (one per line)."""
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]


class PolicyValidator:
    """Main validator class."""

    def __init__(
        self,
        model_name: str = "SparseLLM/BlockFFN-3B-SFT",
        device: str = "mps",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.device = device
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

    def validate_preset(
        self,
        preset_name: str,
        prompts: List[str],
        max_tokens: int = 50,
    ) -> PresetResults:
        """Run validation for a single preset."""
        self.load_model()

        # Get config
        config = AdaptiveInferenceConfig.from_preset(preset_name)
        config = config.for_device(self.device)
        config.collect_statistics = True

        # Create engine
        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

        prompt_results = []
        all_confidence_values = []

        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:40]}...")

            # Reset and run
            engine.reset_statistics()

            start_time = time.perf_counter()
            _, stats = engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_stats=True,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Get layer statistics for window distribution
            layer_stats = engine.get_layer_statistics()
            window_dist = self._aggregate_window_stats(layer_stats)

            # Track confidence for histogram
            all_confidence_values.extend(engine._confidence_history)

            # Compute classification rates from confidence history
            classification = self._classify_confidences(
                engine._confidence_history, config
            )

            # Compute streak stats from confidence history
            streak_stats = self._compute_streak_stats(
                engine._confidence_history, config
            )

            result = PromptResult(
                prompt_id=i,
                prompt_preview=prompt[:50],
                total_tokens=stats.total_tokens,
                pct_small_window=window_dist.get("pct_small_window", 0),
                pct_medium_window=window_dist.get("pct_medium_window", 0),
                pct_full_window=window_dist.get("pct_full_window", 0),
                pct_green=classification["pct_green"],
                pct_yellow=classification["pct_yellow"],
                pct_red=classification["pct_red"],
                escalation_rate=stats.escalation_rate,
                suffix_escalations=stats.escalation_by_level.get(EscalationLevel.SUFFIX, 0),
                token_escalations=stats.escalation_by_level.get(EscalationLevel.TOKEN, 0),
                mean_accepted_streak=streak_stats["mean_streak"],
                max_accepted_streak=streak_stats["max_streak"],
                mean_confidence=stats.mean_confidence,
                min_confidence=min(engine._confidence_history) if engine._confidence_history else 0,
                max_confidence=max(engine._confidence_history) if engine._confidence_history else 0,
                generation_time_ms=elapsed_ms,
                tokens_per_second=stats.tokens_per_second,
            )
            prompt_results.append(result)

        # Detach layers
        engine.detach_adaptive_layers()

        # Aggregate results
        return self._aggregate_preset_results(
            preset_name, config, prompt_results, all_confidence_values
        )

    def _aggregate_window_stats(self, layer_stats: Dict) -> Dict[str, float]:
        """Aggregate window statistics across layers."""
        if not layer_stats:
            return {}

        small = [s.get("pct_small_window", 0) for s in layer_stats.values()]
        medium = [s.get("pct_medium_window", 0) for s in layer_stats.values()]
        full = [s.get("pct_full_window", 0) for s in layer_stats.values()]

        return {
            "pct_small_window": sum(small) / len(small) if small else 0,
            "pct_medium_window": sum(medium) / len(medium) if medium else 0,
            "pct_full_window": sum(full) / len(full) if full else 0,
        }

    def _classify_confidences(
        self, confidences: List[float], config: AdaptiveInferenceConfig
    ) -> Dict[str, float]:
        """Classify confidence values into green/yellow/red."""
        if not confidences:
            return {"pct_green": 0, "pct_yellow": 0, "pct_red": 0}

        green = sum(1 for c in confidences if c >= config.tau_green)
        yellow = sum(1 for c in confidences if config.tau_yellow <= c < config.tau_green)
        red = sum(1 for c in confidences if c < config.tau_yellow)
        total = len(confidences)

        return {
            "pct_green": green / total,
            "pct_yellow": yellow / total,
            "pct_red": red / total,
        }

    def _compute_streak_stats(
        self, confidences: List[float], config: AdaptiveInferenceConfig
    ) -> Dict[str, float]:
        """Compute accepted streak statistics."""
        if not confidences:
            return {"mean_streak": 0, "max_streak": 0}

        streaks = []
        current_streak = 0

        for c in confidences:
            if c >= config.tau_yellow:  # Accepted (green or yellow)
                current_streak += 1
            else:  # Red - escalation
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0

        # Don't forget the last streak
        if current_streak > 0:
            streaks.append(current_streak)

        return {
            "mean_streak": sum(streaks) / len(streaks) if streaks else 0,
            "max_streak": max(streaks) if streaks else 0,
        }

    def _aggregate_preset_results(
        self,
        preset_name: str,
        config: AdaptiveInferenceConfig,
        prompt_results: List[PromptResult],
        all_confidences: List[float],
    ) -> PresetResults:
        """Aggregate per-prompt results into preset summary."""
        n = len(prompt_results)
        if n == 0:
            return PresetResults(
                preset_name=preset_name,
                config_summary={},
                total_prompts=0,
                total_tokens=0,
                avg_pct_small_window=0,
                avg_pct_medium_window=0,
                avg_pct_full_window=0,
                avg_pct_green=0,
                avg_pct_yellow=0,
                avg_pct_red=0,
                avg_escalation_rate=0,
                total_suffix_escalations=0,
                avg_suffix_escalation_frequency=0,
                avg_accepted_streak_length=0,
                max_accepted_streak_observed=0,
                confidence_histogram={},
                avg_mean_confidence=0,
                avg_tokens_per_second=0,
            )

        # Confidence histogram
        histogram = self._build_confidence_histogram(all_confidences)

        return PresetResults(
            preset_name=preset_name,
            config_summary={
                "tau_green": config.tau_green,
                "tau_yellow": config.tau_yellow,
                "tau_red": config.tau_red,
                "small_window": config.small_window,
                "medium_window": config.medium_window,
                "high_sparsity_threshold": config.high_sparsity_threshold,
                "yellow_streak_limit": config.yellow_streak_limit,
                "verification_stride": config.verification_stride,
                "enable_cheap_gate": config.enable_cheap_gate,
            },
            total_prompts=n,
            total_tokens=sum(r.total_tokens for r in prompt_results),
            avg_pct_small_window=sum(r.pct_small_window for r in prompt_results) / n,
            avg_pct_medium_window=sum(r.pct_medium_window for r in prompt_results) / n,
            avg_pct_full_window=sum(r.pct_full_window for r in prompt_results) / n,
            avg_pct_green=sum(r.pct_green for r in prompt_results) / n,
            avg_pct_yellow=sum(r.pct_yellow for r in prompt_results) / n,
            avg_pct_red=sum(r.pct_red for r in prompt_results) / n,
            avg_escalation_rate=sum(r.escalation_rate for r in prompt_results) / n,
            total_suffix_escalations=sum(r.suffix_escalations for r in prompt_results),
            avg_suffix_escalation_frequency=sum(r.suffix_escalations for r in prompt_results) / n,
            avg_accepted_streak_length=sum(r.mean_accepted_streak for r in prompt_results) / n,
            max_accepted_streak_observed=max(r.max_accepted_streak for r in prompt_results),
            confidence_histogram=histogram,
            avg_mean_confidence=sum(r.mean_confidence for r in prompt_results) / n,
            avg_tokens_per_second=sum(r.tokens_per_second for r in prompt_results) / n,
            prompt_results=prompt_results,
        )

    def _build_confidence_histogram(self, confidences: List[float]) -> Dict[str, int]:
        """Build histogram of confidence values."""
        buckets = {
            "0.00-0.50": 0,
            "0.50-0.65": 0,
            "0.65-0.75": 0,
            "0.75-0.85": 0,
            "0.85-0.95": 0,
            "0.95-1.00": 0,
        }

        for c in confidences:
            if c < 0.50:
                buckets["0.00-0.50"] += 1
            elif c < 0.65:
                buckets["0.50-0.65"] += 1
            elif c < 0.75:
                buckets["0.65-0.75"] += 1
            elif c < 0.85:
                buckets["0.75-0.85"] += 1
            elif c < 0.95:
                buckets["0.85-0.95"] += 1
            else:
                buckets["0.95-1.00"] += 1

        return buckets

    def run(
        self,
        presets: List[str],
        prompts: List[str],
        max_tokens: int = 50,
    ) -> Dict[str, PresetResults]:
        """Run validation for multiple presets."""
        results = {}

        for preset in presets:
            print(f"\n{'='*60}")
            print(f"Validating preset: {preset}")
            print(f"{'='*60}")

            results[preset] = self.validate_preset(preset, prompts, max_tokens)

        return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print a formatted comparison report to console."""
    from scripts.policy_report import print_comparison_report as _print_report
    _print_report(results)


def main():
    parser = argparse.ArgumentParser(
        description="Policy Validator for BlockFFN Adaptive Inference"
    )
    parser.add_argument("--mode", choices=["quick", "thorough"], default="quick",
                        help="quick=10 prompts, thorough=100 prompts")
    parser.add_argument("--source", choices=["synthetic", "wikitext", "file"], default="synthetic")
    parser.add_argument("--prompts", type=str, help="Path to prompts file")
    parser.add_argument("--preset", nargs="+",
                        default=["conservative", "balanced", "aggressive"])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Determine prompt count
    prompt_count = 10 if args.mode == "quick" else 100

    # Load prompts
    if args.source == "file" and args.prompts:
        prompts = PromptSource.load_from_file(args.prompts)
    elif args.source == "wikitext":
        prompts = PromptSource.get_wikitext_prompts(prompt_count)
    else:
        prompts = PromptSource.get_synthetic_prompts(prompt_count)

    print(f"Loaded {len(prompts)} prompts from {args.source}")

    # Validate presets
    valid_presets = list(PRESETS.keys())
    for preset in args.preset:
        if preset not in valid_presets:
            print(f"Error: Unknown preset '{preset}'. Available: {valid_presets}")
            sys.exit(1)

    # Run validation
    validator = PolicyValidator(
        model_name=args.model,
        device=args.device,
        verbose=args.verbose,
    )

    results = validator.run(args.preset, prompts, args.max_tokens)

    # Output results
    if args.output_json:
        # Convert to serializable format
        serializable = {}
        for k, v in results.items():
            d = asdict(v)
            # Convert EscalationLevel keys to strings
            for pr in d.get("prompt_results", []):
                if "escalation_by_level" in pr:
                    pr["escalation_by_level"] = {
                        str(level): count
                        for level, count in pr["escalation_by_level"].items()
                    }
            serializable[k] = d

        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nJSON results saved to: {args.output_json}")

    if args.output_csv:
        # Write summary CSV
        with open(args.output_csv, "w", newline="") as f:
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
        print(f"CSV results saved to: {args.output_csv}")

    # Print summary to console
    print_comparison_report(results)


if __name__ == "__main__":
    main()
