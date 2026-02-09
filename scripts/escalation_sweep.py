#!/usr/bin/env python3
"""
Escalation Strategy Sweep

Compares different escalation strategies to find the most effective
approach with minimal compute overhead.

Tests:
- Layer-only escalation (cheapest)
- Token-only escalation
- Suffix escalation with varying lengths (L=2,4,8,16)

Usage:
    python scripts/escalation_sweep.py --output results/escalation_sweep.csv
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
from src.verification.decision import EscalationLevel


# Escalation configurations to test
ESCALATION_CONFIGS = [
    # Layer-only (cheapest)
    {
        "name": "layer_only",
        "enable_layer_escalation": True,
        "enable_token_escalation": False,
        "enable_suffix_escalation": False,
        "suffix_escalation_length": 0,
    },

    # Token-only
    {
        "name": "token_only",
        "enable_layer_escalation": False,
        "enable_token_escalation": True,
        "enable_suffix_escalation": False,
        "suffix_escalation_length": 0,
    },

    # Suffix to end (original behavior)
    {
        "name": "suffix_unbounded",
        "enable_layer_escalation": False,
        "enable_token_escalation": False,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 0,
    },

    # Bounded suffix with varying lengths
    {
        "name": "suffix_2",
        "enable_layer_escalation": False,
        "enable_token_escalation": False,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 2,
    },
    {
        "name": "suffix_4",
        "enable_layer_escalation": False,
        "enable_token_escalation": False,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 4,
    },
    {
        "name": "suffix_8",
        "enable_layer_escalation": False,
        "enable_token_escalation": False,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 8,
    },
    {
        "name": "suffix_16",
        "enable_layer_escalation": False,
        "enable_token_escalation": False,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 16,
    },

    # Combined strategies
    {
        "name": "layer_then_token",
        "enable_layer_escalation": True,
        "enable_token_escalation": True,
        "enable_suffix_escalation": False,
        "suffix_escalation_length": 0,
    },
    {
        "name": "token_then_suffix_4",
        "enable_layer_escalation": False,
        "enable_token_escalation": True,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 4,
    },
    {
        "name": "all_enabled_suffix_8",
        "enable_layer_escalation": True,
        "enable_token_escalation": True,
        "enable_suffix_escalation": True,
        "suffix_escalation_length": 8,
    },
]


@dataclass
class EscalationSweepResult:
    """Result from testing an escalation configuration."""
    config_name: str
    enable_layer: bool
    enable_token: bool
    enable_suffix: bool
    suffix_length: int

    # Escalation counts
    total_tokens: int
    total_escalations: int
    layer_escalations: int
    token_escalations: int
    suffix_escalations: int

    # Rates
    escalation_rate: float
    recovery_rate: float  # How often escalation corrected the output

    # Quality
    green_rate: float
    mean_confidence: float
    avg_streak_length: float

    # Cascade detection
    cascade_triggers: int
    cascade_rate: float

    # Performance
    tokens_per_second: float
    avg_escalation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EscalationSweep:
    """
    Runs escalation strategy comparison experiments.
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
    ) -> EscalationSweepResult:
        """
        Test a single escalation configuration.

        Args:
            config_dict: Escalation configuration parameters
            prompts: List of test prompts
            max_tokens: Maximum tokens to generate per prompt

        Returns:
            EscalationSweepResult with metrics
        """
        self.load_model()

        # Create engine with custom escalation config
        base_config = PRESETS["balanced"].for_device(self.device)
        base_config.collect_statistics = True

        # Apply escalation config
        for key, value in config_dict.items():
            if key != "name" and hasattr(base_config, key):
                setattr(base_config, key, value)

        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, base_config)

        # Track metrics
        total_tokens = 0
        total_escalations = 0
        layer_escalations = 0
        token_escalations = 0
        suffix_escalations = 0
        cascade_triggers = 0
        total_time = 0
        escalation_times = []
        all_confidences = []
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
            total_escalations += stats.escalated_tokens

            # Count by level
            layer_escalations += stats.escalation_by_level.get(EscalationLevel.LAYER, 0)
            token_escalations += stats.escalation_by_level.get(EscalationLevel.TOKEN, 0)
            suffix_escalations += stats.escalation_by_level.get(EscalationLevel.SUFFIX, 0)

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

            # Check for cascade triggers
            decision_stats = engine.get_decision_statistics()
            if decision_stats.get("suffix_escalation_count", 0) >= base_config.cascade_threshold:
                cascade_triggers += 1

        # Cleanup
        engine.detach_adaptive_layers()

        # Calculate metrics
        escalation_rate = total_escalations / total_tokens if total_tokens > 0 else 0
        green_tokens = sum(1 for c in all_confidences if c >= base_config.tau_green)
        green_rate = green_tokens / len(all_confidences) if all_confidences else 0
        mean_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        avg_streak = sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        cascade_rate = cascade_triggers / len(prompts) if prompts else 0

        # Recovery rate is approximated by escalation effectiveness
        # (in practice this would need baseline comparison)
        recovery_rate = 1.0 - escalation_rate  # Simplified approximation

        avg_escalation_time = 0
        if escalation_times:
            avg_escalation_time = sum(escalation_times) / len(escalation_times) * 1000

        return EscalationSweepResult(
            config_name=config_dict.get("name", "unknown"),
            enable_layer=config_dict.get("enable_layer_escalation", False),
            enable_token=config_dict.get("enable_token_escalation", False),
            enable_suffix=config_dict.get("enable_suffix_escalation", False),
            suffix_length=config_dict.get("suffix_escalation_length", 0),
            total_tokens=total_tokens,
            total_escalations=total_escalations,
            layer_escalations=layer_escalations,
            token_escalations=token_escalations,
            suffix_escalations=suffix_escalations,
            escalation_rate=escalation_rate,
            recovery_rate=recovery_rate,
            green_rate=green_rate,
            mean_confidence=mean_confidence,
            avg_streak_length=avg_streak,
            cascade_triggers=cascade_triggers,
            cascade_rate=cascade_rate,
            tokens_per_second=tokens_per_second,
            avg_escalation_time_ms=avg_escalation_time,
        )

    def run_sweep(
        self,
        prompts: List[str],
        configs: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 50,
    ) -> List[EscalationSweepResult]:
        """
        Run the full escalation sweep.

        Args:
            prompts: Test prompts
            configs: Configurations to test (uses defaults if None)
            max_tokens: Maximum tokens per prompt

        Returns:
            List of EscalationSweepResult
        """
        if configs is None:
            configs = ESCALATION_CONFIGS

        print(f"Testing {len(configs)} escalation configurations...")

        results = []
        for i, config in enumerate(configs):
            config_name = config.get("name", f"config_{i}")
            print(f"  [{i+1}/{len(configs)}] {config_name}...")

            result = self.test_config(config, prompts, max_tokens)
            results.append(result)

            if self.verbose:
                print(f"    -> escalation_rate={result.escalation_rate:.3f}, "
                      f"cascade_rate={result.cascade_rate:.3f}")

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
        description="Escalation Strategy Sweep"
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--prompts", type=int, default=20,
                        help="Number of prompts to test")
    parser.add_argument("--source", choices=["synthetic", "wikitext"], default="synthetic")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/escalation_sweep.csv")
    parser.add_argument("--output-json", type=str, default="results/escalation_sweep.json")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print("ESCALATION STRATEGY SWEEP")
    print("=" * 70)

    # Get prompts
    prompts = get_prompts(args.source, args.prompts)
    print(f"Using {len(prompts)} prompts from {args.source}")

    # Run sweep
    sweep = EscalationSweep(
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
            "config_name", "enable_layer", "enable_token", "enable_suffix", "suffix_length",
            "total_tokens", "total_escalations", "layer_esc", "token_esc", "suffix_esc",
            "escalation_rate", "recovery_rate", "green_rate", "mean_confidence",
            "avg_streak", "cascade_triggers", "cascade_rate", "tok_per_sec"
        ])
        for r in results:
            writer.writerow([
                r.config_name, r.enable_layer, r.enable_token, r.enable_suffix, r.suffix_length,
                r.total_tokens, r.total_escalations, r.layer_escalations,
                r.token_escalations, r.suffix_escalations,
                f"{r.escalation_rate:.4f}", f"{r.recovery_rate:.4f}",
                f"{r.green_rate:.4f}", f"{r.mean_confidence:.4f}",
                f"{r.avg_streak_length:.2f}", r.cascade_triggers,
                f"{r.cascade_rate:.4f}", f"{r.tokens_per_second:.1f}"
            ])
    print(f"\nCSV saved to: {output_path}")

    # Save JSON
    json_path = Path(args.output_json)
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"JSON saved to: {json_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("ESCALATION SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Config':<25} {'EscRate':<10} {'Cascades':<10} {'GreenRate':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.config_name:<25} {r.escalation_rate:.3f}      "
              f"{r.cascade_triggers:<10} {r.green_rate:.3f}")

    # Find best configuration (lowest escalation rate with no cascades)
    no_cascade = [r for r in results if r.cascade_triggers == 0]
    if no_cascade:
        best = min(no_cascade, key=lambda r: r.escalation_rate)
    else:
        best = min(results, key=lambda r: r.escalation_rate)

    print("\n" + "-" * 70)
    print(f"RECOMMENDED: {best.config_name} (escalation={best.escalation_rate:.3f}, "
          f"cascades={best.cascade_triggers})")


if __name__ == "__main__":
    main()
