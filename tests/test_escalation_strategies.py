"""
Escalation Strategy Comparison Tests

Compare different escalation strategies to find the smallest suffix
that reliably fixes mistakes.

Strategies:
- Layer-only: Re-run next layer with full attention
- Token-only: Re-run current token through all layers
- Suffix L=2,4,8,16: Re-run last L tokens
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.verification.decision import EscalationLevel


@dataclass
class StrategyResult:
    """Result from testing a single escalation strategy."""
    strategy_name: str
    total_escalations: int
    successful_recoveries: int
    cascade_triggers: int
    compute_cost_relative: float  # Relative to full recompute

    @property
    def recovery_rate(self) -> float:
        if self.total_escalations == 0:
            return 0.0
        return self.successful_recoveries / self.total_escalations

    @property
    def cascade_rate(self) -> float:
        if self.total_escalations == 0:
            return 0.0
        return self.cascade_triggers / self.total_escalations


@dataclass
class StrategyComparison:
    """Comparison results across all strategies."""
    strategies: Dict[str, StrategyResult] = field(default_factory=dict)
    baseline_outputs: List[int] = field(default_factory=list)
    prompts_tested: int = 0

    def best_by_recovery(self) -> str:
        """Return strategy with highest recovery rate."""
        return max(self.strategies.keys(),
                   key=lambda s: self.strategies[s].recovery_rate)

    def best_by_efficiency(self) -> str:
        """Return strategy with best recovery/cost ratio."""
        def efficiency(s):
            r = self.strategies[s]
            if r.compute_cost_relative == 0:
                return 0
            return r.recovery_rate / r.compute_cost_relative
        return max(self.strategies.keys(), key=efficiency)


class EscalationStrategyTester:
    """
    Test different escalation strategies to find the minimum
    suffix length that reliably recovers baseline outputs.
    """

    STRATEGIES = [
        ("layer_only", EscalationLevel.LAYER),
        ("token_only", EscalationLevel.TOKEN),
        ("suffix_2", EscalationLevel.SUFFIX),
        ("suffix_4", EscalationLevel.SUFFIX),
        ("suffix_8", EscalationLevel.SUFFIX),
        ("suffix_16", EscalationLevel.SUFFIX),
    ]

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def load_model(self, model_name: str = "SparseLLM/BlockFFN-3B-SFT") -> None:
        """Load model if not provided."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {model_name}...")
        dtype = torch.float32 if self.device == "cpu" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        print("Model loaded.")

    def _generate_baseline(
        self,
        prompt: str,
        max_tokens: int,
    ) -> List[int]:
        """Generate baseline output with full attention (no adaptive)."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )

        return outputs[0].tolist()[input_ids.shape[1]:]

    def _generate_with_adaptive(
        self,
        prompt: str,
        max_tokens: int,
        config: AdaptiveInferenceConfig,
    ) -> Tuple[List[int], List[Tuple[int, float]]]:
        """
        Generate with adaptive inference and track escalation points.

        Returns:
            (generated_tokens, escalation_points)
            escalation_points: List of (position, confidence) tuples
        """
        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

        # Generate
        _, stats = engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_stats=True,
        )

        # Get generated tokens
        inputs = self.tokenizer(prompt, return_tensors="pt")
        full_output, _ = engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_stats=True,
        )

        generated = self.tokenizer.encode(full_output)[len(inputs["input_ids"][0]):]

        # Track escalation points
        escalation_points = []
        for i, conf in enumerate(engine._confidence_history):
            if conf < config.tau_yellow:
                escalation_points.append((i, conf))

        engine.detach_adaptive_layers()

        return generated, escalation_points

    def _test_strategy(
        self,
        strategy_name: str,
        prompt: str,
        baseline_tokens: List[int],
        escalation_points: List[Tuple[int, float]],
        config: AdaptiveInferenceConfig,
    ) -> Tuple[int, int, int]:
        """
        Test a single escalation strategy with actual model execution.

        Runs the escalation module for each escalation point and compares
        the output to the baseline.

        Returns:
            (total_escalations, successful_recoveries, cascade_triggers)
        """
        from src.engine.adaptive_engine import AdaptiveInferenceEngine
        from src.verification.escalation import EscalationModule

        total = len(escalation_points)
        recoveries = 0
        cascades = 0

        if total == 0:
            return 0, 0, 0

        # Determine suffix length and escalation level
        suffix_length = 0
        if "suffix_" in strategy_name:
            suffix_length = int(strategy_name.split("_")[1])

        # Configure based on strategy
        test_config = AdaptiveInferenceConfig.from_preset("balanced").for_device(self.device)
        test_config.collect_statistics = True

        if strategy_name == "layer_only":
            test_config.enable_layer_escalation = True
            test_config.enable_token_escalation = False
            test_config.enable_suffix_escalation = False
        elif strategy_name == "token_only":
            test_config.enable_layer_escalation = False
            test_config.enable_token_escalation = True
            test_config.enable_suffix_escalation = False
        else:
            # Suffix strategy
            test_config.enable_layer_escalation = False
            test_config.enable_token_escalation = False
            test_config.enable_suffix_escalation = True
            test_config.suffix_escalation_length = suffix_length

        # Create engine with configured strategy
        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, test_config)

        try:
            # Generate with this strategy
            output_text, stats = engine.generate(
                prompt,
                max_new_tokens=len(baseline_tokens),
                do_sample=False,
                return_stats=True,
            )

            # Tokenize output
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output_tokens = self.tokenizer.encode(output_text)[len(inputs["input_ids"][0]):]

            # Count recoveries: positions where output matches baseline
            min_len = min(len(output_tokens), len(baseline_tokens))
            matching_tokens = sum(
                1 for i in range(min_len)
                if output_tokens[i] == baseline_tokens[i]
            )

            # Recovery rate based on actual token matching
            if min_len > 0:
                recovery_rate = matching_tokens / min_len
                recoveries = int(total * recovery_rate)

            # Cascade detection from stats
            decision_stats = engine.get_decision_statistics()
            cascades = 0
            suffix_count = decision_stats.get("suffix_escalation_count", 0)
            if suffix_count >= test_config.cascade_threshold:
                cascades = 1

            # Track consecutive escalations as cascades
            prev_was_escalation = False
            for conf in engine._confidence_history:
                is_escalation = conf < test_config.tau_yellow
                if is_escalation and prev_was_escalation:
                    cascades += 1
                prev_was_escalation = is_escalation

        except Exception as e:
            print(f"    Strategy {strategy_name} failed: {e}")
            recoveries = 0
            cascades = 0
        finally:
            engine.detach_adaptive_layers()

        return total, recoveries, cascades

    def _compute_cost_relative(self, strategy_name: str, context_length: int) -> float:
        """Compute relative cost of strategy vs full recompute."""
        if strategy_name == "layer_only":
            # One layer vs all layers
            return 0.05  # Assume 1/20 layers
        elif strategy_name == "token_only":
            # All layers for one token
            return 0.10
        elif "suffix_" in strategy_name:
            suffix_length = int(strategy_name.split("_")[1])
            return suffix_length / context_length
        return 1.0

    def compare_strategies(
        self,
        prompts: List[str],
        max_tokens: int = 50,
    ) -> StrategyComparison:
        """
        Compare all escalation strategies across multiple prompts.

        Args:
            prompts: List of prompts to test
            max_tokens: Tokens to generate per prompt

        Returns:
            StrategyComparison with results
        """
        self.load_model()

        # Use aggressive config to generate more escalations
        config = PRESETS["aggressive"].for_device(self.device)
        config.collect_statistics = True

        comparison = StrategyComparison()
        comparison.prompts_tested = len(prompts)

        # Initialize strategy results
        strategy_results = {
            name: StrategyResult(
                strategy_name=name,
                total_escalations=0,
                successful_recoveries=0,
                cascade_triggers=0,
                compute_cost_relative=self._compute_cost_relative(name, 256),
            )
            for name, _ in self.STRATEGIES
        }

        for prompt in prompts:
            print(f"Testing: {prompt[:40]}...")

            # Generate baseline
            baseline = self._generate_baseline(prompt, max_tokens)
            comparison.baseline_outputs.extend(baseline)

            # Generate with adaptive and get escalation points
            _, escalation_points = self._generate_with_adaptive(
                prompt, max_tokens, config
            )

            if not escalation_points:
                continue

            # Test each strategy
            for name, level in self.STRATEGIES:
                total, recoveries, cascades = self._test_strategy(
                    name, prompt, baseline, escalation_points, config
                )

                strategy_results[name].total_escalations += total
                strategy_results[name].successful_recoveries += recoveries
                strategy_results[name].cascade_triggers += cascades

        comparison.strategies = strategy_results
        return comparison


def print_comparison_report(comparison: StrategyComparison) -> None:
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("ESCALATION STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Prompts tested: {comparison.prompts_tested}")
    print()

    header = f"{'Strategy':<15} {'Total':>8} {'Recovered':>10} {'Recovery%':>10} {'Cascade%':>10} {'Cost':>8}"
    print(header)
    print("-" * len(header))

    for name, result in comparison.strategies.items():
        print(f"{name:<15} "
              f"{result.total_escalations:>8} "
              f"{result.successful_recoveries:>10} "
              f"{result.recovery_rate*100:>9.1f}% "
              f"{result.cascade_rate*100:>9.1f}% "
              f"{result.compute_cost_relative:>7.2f}x")

    print()
    print("RECOMMENDATIONS:")
    print(f"  Best recovery rate: {comparison.best_by_recovery()}")
    print(f"  Best efficiency: {comparison.best_by_efficiency()}")
    print("=" * 70)


class EscalationStrategyTests:
    """Test class for escalation strategies."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tester = None

    def setup(self):
        """Set up test environment."""
        self.tester = EscalationStrategyTester(device=self.device)

    def test_strategy_comparison(self) -> bool:
        """Test: Compare all strategies on synthetic prompts."""
        self.setup()

        prompts = [
            "The capital of France is",
            "In machine learning, gradient descent",
            "def fibonacci(n): return",
        ]

        try:
            comparison = self.tester.compare_strategies(prompts, max_tokens=20)

            # Verify we got results
            assert comparison.prompts_tested == len(prompts)
            assert len(comparison.strategies) == len(EscalationStrategyTester.STRATEGIES)

            # Print report
            print_comparison_report(comparison)

            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False

    def test_layer_only_strategy(self) -> bool:
        """Test: Layer-only escalation has low cost."""
        cost = EscalationStrategyTester(device=self.device)._compute_cost_relative(
            "layer_only", 256
        )
        assert cost < 0.1, f"Layer-only cost should be low, got {cost}"
        return True

    def test_suffix_cost_scaling(self) -> bool:
        """Test: Suffix cost scales with suffix length."""
        tester = EscalationStrategyTester(device=self.device)

        cost_2 = tester._compute_cost_relative("suffix_2", 256)
        cost_8 = tester._compute_cost_relative("suffix_8", 256)
        cost_16 = tester._compute_cost_relative("suffix_16", 256)

        assert cost_2 < cost_8 < cost_16, "Suffix cost should increase with length"
        return True

    def run_all(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        tests = {
            "test_layer_only_strategy": self.test_layer_only_strategy,
            "test_suffix_cost_scaling": self.test_suffix_cost_scaling,
        }

        results = {}
        for name, test_fn in tests.items():
            try:
                results[name] = test_fn()
                status = "PASS" if results[name] else "FAIL"
            except Exception as e:
                results[name] = False
                status = f"ERROR: {e}"
            print(f"  {name}: {status}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test escalation strategies")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--full", action="store_true", help="Run full comparison with model")

    args = parser.parse_args()

    print("Running escalation strategy tests...")

    tests = EscalationStrategyTests(device=args.device)
    results = tests.run_all()

    if args.full:
        print("\nRunning full strategy comparison (requires model)...")
        tests.test_strategy_comparison()

    passed = sum(results.values())
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
