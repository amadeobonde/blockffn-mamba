#!/usr/bin/env python3
"""
Verification Scheduling Tuning Sweep

Systematically tests verification scheduling configurations to find
optimal settings that reduce overhead while maintaining quality.

Tests:
- Token stride: verify every k tokens (k=1,2,4,8)
- Layer gate: verify only in last N% of layers
- Cheap gate: margin-first gating

Usage:
    python scripts/verification_tuning_sweep.py --output results/scheduling_sweep.csv
"""

import argparse
import json
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.engine.adaptive_engine import AdaptiveInferenceEngine


# Scheduling configurations to test
SCHEDULING_CONFIGS = [
    # Baseline: verify every token
    {"name": "baseline", "verification_stride": 1, "enable_cheap_gate": False, "verification_layer_ratio": 1.0},

    # Stride experiments (verify every k tokens)
    {"name": "stride_2", "verification_stride": 2, "enable_cheap_gate": False, "verification_layer_ratio": 1.0},
    {"name": "stride_4", "verification_stride": 4, "enable_cheap_gate": False, "verification_layer_ratio": 1.0},
    {"name": "stride_8", "verification_stride": 8, "enable_cheap_gate": False, "verification_layer_ratio": 1.0},

    # Layer-gate experiments (verify only top N% layers)
    {"name": "layer_75", "verification_stride": 1, "enable_cheap_gate": False, "verification_layer_ratio": 0.75},
    {"name": "layer_50", "verification_stride": 1, "enable_cheap_gate": False, "verification_layer_ratio": 0.50},
    {"name": "layer_33", "verification_stride": 1, "enable_cheap_gate": False, "verification_layer_ratio": 0.33},
    {"name": "layer_25", "verification_stride": 1, "enable_cheap_gate": False, "verification_layer_ratio": 0.25},

    # Cheap-gate experiments (margin-first, skip entropy)
    {"name": "cheap_gate_90", "verification_stride": 1, "enable_cheap_gate": True,
     "cheap_gate_high_threshold": 0.90, "cheap_gate_low_threshold": 0.40, "verification_layer_ratio": 1.0},
    {"name": "cheap_gate_95", "verification_stride": 1, "enable_cheap_gate": True,
     "cheap_gate_high_threshold": 0.95, "cheap_gate_low_threshold": 0.35, "verification_layer_ratio": 1.0},

    # Combined experiments (best of all)
    {"name": "combined_4_33", "verification_stride": 4, "enable_cheap_gate": True,
     "cheap_gate_high_threshold": 0.90, "cheap_gate_low_threshold": 0.40, "verification_layer_ratio": 0.33},
    {"name": "combined_2_50", "verification_stride": 2, "enable_cheap_gate": True,
     "cheap_gate_high_threshold": 0.90, "cheap_gate_low_threshold": 0.40, "verification_layer_ratio": 0.50},
]


@dataclass
class SchedulingSweepResult:
    """Result from testing a scheduling configuration."""
    config_name: str
    verification_stride: int
    verification_layer_ratio: float
    enable_cheap_gate: bool

    # Performance metrics
    total_tokens: int
    verified_tokens: int
    verification_rate: float
    verification_time_ms: float
    avg_verification_time_per_token_ms: float

    # Timing breakdown
    total_margin_time_ms: float
    total_entropy_time_ms: float
    entropy_skip_rate: float

    # Quality metrics
    escalation_rate: float
    green_rate: float
    mean_confidence: float
    avg_streak_length: float

    # Throughput
    tokens_per_second: float
    speedup_vs_baseline: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VerificationTuningSweep:
    """
    Runs verification scheduling sweep experiments.
    """

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
        self.baseline_time: Optional[float] = None

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

    def test_config(
        self,
        config_dict: Dict[str, Any],
        prompts: List[str],
        max_tokens: int = 50,
    ) -> SchedulingSweepResult:
        """
        Test a single scheduling configuration.

        Args:
            config_dict: Scheduling configuration parameters
            prompts: List of test prompts
            max_tokens: Maximum tokens to generate per prompt

        Returns:
            SchedulingSweepResult with metrics
        """
        self.load_model()

        # Create engine with custom scheduling config
        base_config = PRESETS["balanced"].for_device(self.device)
        base_config.collect_statistics = True

        # Apply scheduling config
        for key, value in config_dict.items():
            if key != "name" and hasattr(base_config, key):
                setattr(base_config, key, value)

        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, base_config)

        # Track metrics
        total_tokens = 0
        total_time = 0
        all_confidences = []
        streak_lengths = []
        escalated_tokens = 0

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
            all_confidences.extend(engine._confidence_history)

            # Track streaks
            current_streak = 0
            for conf in engine._confidence_history:
                if conf >= base_config.tau_yellow:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streak_lengths.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                streak_lengths.append(current_streak)

        # Get scheduling stats
        sched_stats = engine.verification_scheduler.get_stats()
        scheduling_stats = sched_stats.get("scheduling_stats", {})

        # Cleanup
        engine.detach_adaptive_layers()

        # Calculate metrics
        verified_tokens = scheduling_stats.get("verified_tokens", total_tokens)
        verification_rate = verified_tokens / total_tokens if total_tokens > 0 else 1.0
        verification_time_ms = scheduling_stats.get("total_verification_time_ms", 0)
        avg_verification_time = verification_time_ms / verified_tokens if verified_tokens > 0 else 0

        margin_time_ms = scheduling_stats.get("total_margin_time_ms", 0)
        entropy_time_ms = scheduling_stats.get("total_entropy_time_ms", 0)
        entropy_skip_rate = scheduling_stats.get("entropy_skip_rate", 0)

        escalation_rate = escalated_tokens / total_tokens if total_tokens > 0 else 0
        green_tokens = sum(1 for c in all_confidences if c >= base_config.tau_green)
        green_rate = green_tokens / len(all_confidences) if all_confidences else 0
        mean_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        avg_streak = sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        # Calculate speedup vs baseline
        speedup = 1.0
        if self.baseline_time is not None and total_time > 0:
            speedup = self.baseline_time / total_time

        # If this is the baseline, record the time
        if config_dict.get("name") == "baseline":
            self.baseline_time = total_time

        return SchedulingSweepResult(
            config_name=config_dict.get("name", "unknown"),
            verification_stride=config_dict.get("verification_stride", 1),
            verification_layer_ratio=config_dict.get("verification_layer_ratio", 1.0),
            enable_cheap_gate=config_dict.get("enable_cheap_gate", False),
            total_tokens=total_tokens,
            verified_tokens=verified_tokens,
            verification_rate=verification_rate,
            verification_time_ms=verification_time_ms,
            avg_verification_time_per_token_ms=avg_verification_time,
            total_margin_time_ms=margin_time_ms,
            total_entropy_time_ms=entropy_time_ms,
            entropy_skip_rate=entropy_skip_rate,
            escalation_rate=escalation_rate,
            green_rate=green_rate,
            mean_confidence=mean_confidence,
            avg_streak_length=avg_streak,
            tokens_per_second=tokens_per_second,
            speedup_vs_baseline=speedup,
        )

    def run_sweep(
        self,
        prompts: List[str],
        configs: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 50,
    ) -> List[SchedulingSweepResult]:
        """
        Run the full scheduling sweep.

        Args:
            prompts: Test prompts
            configs: Configurations to test (uses defaults if None)
            max_tokens: Maximum tokens per prompt

        Returns:
            List of SchedulingSweepResult
        """
        if configs is None:
            configs = SCHEDULING_CONFIGS

        print(f"Testing {len(configs)} scheduling configurations...")

        results = []
        for i, config in enumerate(configs):
            config_name = config.get("name", f"config_{i}")
            print(f"  [{i+1}/{len(configs)}] {config_name}...")

            result = self.test_config(config, prompts, max_tokens)
            results.append(result)

            if self.verbose:
                print(f"    -> verification_rate={result.verification_rate:.2f}, "
                      f"escalation={result.escalation_rate:.3f}, "
                      f"speedup={result.speedup_vs_baseline:.2f}x")

        # Update speedup for all results now that we have baseline
        if self.baseline_time is not None:
            for result in results:
                if result.config_name != "baseline" and result.tokens_per_second > 0:
                    # Recalculate based on tokens_per_second
                    baseline_tps = results[0].tokens_per_second
                    if baseline_tps > 0:
                        result.speedup_vs_baseline = result.tokens_per_second / baseline_tps

        return results


def get_prompts(source: str, count: int) -> List[str]:
    """Load prompts for testing."""
    from scripts.policy_validator import PromptSource

    if source == "wikitext":
        return PromptSource.get_wikitext_prompts(count)
    else:
        return PromptSource.get_synthetic_prompts(count)


def main():
    parser = argparse.ArgumentParser(
        description="Verification Scheduling Tuning Sweep"
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--prompts", type=int, default=20,
                        help="Number of prompts to test")
    parser.add_argument("--source", choices=["synthetic", "wikitext"], default="synthetic")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/scheduling_sweep.csv")
    parser.add_argument("--output-json", type=str, default="results/scheduling_sweep.json")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print("VERIFICATION SCHEDULING TUNING SWEEP")
    print("=" * 70)

    # Get prompts
    prompts = get_prompts(args.source, args.prompts)
    print(f"Using {len(prompts)} prompts from {args.source}")

    # Run sweep
    sweep = VerificationTuningSweep(
        model_name=args.model,
        device=args.device,
        verbose=args.verbose,
    )

    results = sweep.run_sweep(prompts=prompts, max_tokens=args.max_tokens)

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config_name", "verification_stride", "layer_ratio", "cheap_gate",
            "total_tokens", "verified_tokens", "verification_rate",
            "verification_time_ms", "avg_verification_time_ms",
            "margin_time_ms", "entropy_time_ms", "entropy_skip_rate",
            "escalation_rate", "green_rate", "mean_confidence",
            "avg_streak", "tok_per_sec", "speedup"
        ])
        for r in results:
            writer.writerow([
                r.config_name, r.verification_stride, r.verification_layer_ratio,
                r.enable_cheap_gate,
                r.total_tokens, r.verified_tokens, f"{r.verification_rate:.3f}",
                f"{r.verification_time_ms:.2f}", f"{r.avg_verification_time_per_token_ms:.4f}",
                f"{r.total_margin_time_ms:.2f}", f"{r.total_entropy_time_ms:.2f}",
                f"{r.entropy_skip_rate:.3f}",
                f"{r.escalation_rate:.3f}", f"{r.green_rate:.3f}",
                f"{r.mean_confidence:.3f}",
                f"{r.avg_streak_length:.1f}", f"{r.tokens_per_second:.1f}",
                f"{r.speedup_vs_baseline:.2f}"
            ])
    print(f"\nCSV saved to: {output_path}")

    # Save JSON
    json_path = Path(args.output_json)
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"JSON saved to: {json_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SCHEDULING SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Config':<20} {'VerifyRate':<12} {'EscRate':<10} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.config_name:<20} {r.verification_rate:.2f}         "
              f"{r.escalation_rate:.3f}      {r.speedup_vs_baseline:.2f}x")

    # Find best configuration
    best = max(results, key=lambda r: r.speedup_vs_baseline if r.escalation_rate < 0.15 else 0)
    print("\n" + "-" * 70)
    print(f"RECOMMENDED: {best.config_name} (speedup={best.speedup_vs_baseline:.2f}x, "
          f"escalation={best.escalation_rate:.3f})")


if __name__ == "__main__":
    main()
