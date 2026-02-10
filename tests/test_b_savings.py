"""
Phase B: Compute Savings Quantification

Only run these tests AFTER Phase A passes.

B1. Attention savings measurement
B2. FFN/CLS savings measurement
B3. Combined throughput analysis
B4. Scaling behavior (context length vs savings)
"""

import torch
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS


@dataclass
class SavingsMetrics:
    """Computed savings metrics."""
    # Attention
    mean_sparsity: float = 0.0
    pct_small_window: float = 0.0
    pct_medium_window: float = 0.0
    pct_full_window: float = 0.0
    theoretical_attention_savings: float = 0.0

    # FFN/CLS
    cls_expert_reuse_rate: float = 0.0
    cls_compute_savings: float = 0.0
    cls_chunks_processed: int = 0

    # Combined
    escalation_rate: float = 0.0
    reduced_compute_rate: float = 0.0

    # Timing
    baseline_time_ms: float = 0.0
    adaptive_time_ms: float = 0.0
    actual_speedup: float = 0.0

    def __str__(self) -> str:
        return (
            f"ATTENTION SAVINGS:\n"
            f"  Mean routing sparsity: {self.mean_sparsity*100:.1f}%\n"
            f"  Window distribution: small={self.pct_small_window*100:.1f}%, "
            f"medium={self.pct_medium_window*100:.1f}%, "
            f"full={self.pct_full_window*100:.1f}%\n"
            f"  Theoretical attention savings: {self.theoretical_attention_savings*100:.1f}%\n"
            f"\n"
            f"FFN/CLS SAVINGS:\n"
            f"  Expert reuse rate: {self.cls_expert_reuse_rate*100:.1f}%\n"
            f"  CLS compute savings: {self.cls_compute_savings*100:.1f}%\n"
            f"  Chunks processed: {self.cls_chunks_processed}\n"
            f"\n"
            f"OVERALL:\n"
            f"  Reduced compute tokens: {self.reduced_compute_rate*100:.1f}%\n"
            f"  Escalation rate: {self.escalation_rate*100:.1f}%\n"
            f"  Baseline time: {self.baseline_time_ms:.2f}ms\n"
            f"  Adaptive time: {self.adaptive_time_ms:.2f}ms\n"
            f"  Actual speedup: {self.actual_speedup:.2f}x"
        )


@dataclass
class ScalingPoint:
    """Single point in scaling analysis."""
    context_length: int
    attention_savings: float
    cls_savings: float
    actual_speedup: float
    escalation_rate: float


class SavingsTestSuite:
    """
    Phase B: Compute Savings Quantification

    Measures actual compute savings from adaptive inference,
    broken down by component.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results: List[SavingsMetrics] = []

    def load_model(self, model_name: str = "SparseLLM/BlockFFN-3B-SFT") -> None:
        """Load model if not provided."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        print("Model loaded.")

    def _time_forward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        num_runs: int = 5,
        warmup: int = 2,
    ) -> float:
        """Time forward pass, return mean time in ms."""
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)

        # Timed runs
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)

            torch.cuda.synchronize() if self.device == "cuda" else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        return sum(times) / len(times)

    # =========================================================================
    # B1: Attention Savings
    # =========================================================================

    def test_b1_attention_savings(
        self,
        prompts: Optional[List[str]] = None,
    ) -> SavingsMetrics:
        """
        B1: Measure attention compute savings.

        Quantifies:
        - What fraction of tokens use small/medium/full windows
        - Theoretical attention FLOP savings
        - Actual time difference
        """
        if prompts is None:
            prompts = [
                "The quick brown fox jumps over the lazy dog. " * 5,
                "In machine learning, neural networks are computational systems " * 3,
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
            ]

        self.load_model()

        from src.attention.adaptive_attention import (
            attach_adaptive_attention,
            detach_adaptive_attention,
            get_all_statistics,
        )
        from src.adaptive.window_mapper import compute_effective_attention_cost

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = False  # Isolate attention savings
        config.collect_statistics = True

        all_metrics = SavingsMetrics()
        baseline_times = []
        adaptive_times = []
        sparsities = []
        window_dists = {"small": [], "medium": [], "full": []}

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            seq_len = inputs["input_ids"].shape[1]

            # Baseline timing
            baseline_time = self._time_forward(
                self.model, inputs["input_ids"], inputs.get("attention_mask")
            )
            baseline_times.append(baseline_time)

            # Adaptive
            attach_adaptive_attention(self.model, config)

            adaptive_time = self._time_forward(
                self.model, inputs["input_ids"], inputs.get("attention_mask")
            )
            adaptive_times.append(adaptive_time)

            # Collect statistics
            stats = get_all_statistics(self.model)
            for layer_stats in stats.values():
                sparsities.append(layer_stats.get("mean_sparsity", 0))
                window_dists["small"].append(layer_stats.get("pct_small_window", 0))
                window_dists["medium"].append(layer_stats.get("pct_medium_window", 0))
                window_dists["full"].append(layer_stats.get("pct_full_window", 0))

            detach_adaptive_attention(self.model)

        # Aggregate
        all_metrics.mean_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0
        all_metrics.pct_small_window = sum(window_dists["small"]) / len(window_dists["small"]) if window_dists["small"] else 0
        all_metrics.pct_medium_window = sum(window_dists["medium"]) / len(window_dists["medium"]) if window_dists["medium"] else 0
        all_metrics.pct_full_window = sum(window_dists["full"]) / len(window_dists["full"]) if window_dists["full"] else 0

        # Theoretical savings based on window distribution
        # Small window = 16/seq_len of full cost, etc.
        avg_seq_len = 50  # approximate
        theoretical_cost = (
            all_metrics.pct_small_window * (config.small_window / avg_seq_len) +
            all_metrics.pct_medium_window * (config.medium_window / avg_seq_len) +
            all_metrics.pct_full_window * 1.0
        )
        all_metrics.theoretical_attention_savings = 1.0 - theoretical_cost

        all_metrics.baseline_time_ms = sum(baseline_times) / len(baseline_times)
        all_metrics.adaptive_time_ms = sum(adaptive_times) / len(adaptive_times)
        all_metrics.actual_speedup = all_metrics.baseline_time_ms / all_metrics.adaptive_time_ms if all_metrics.adaptive_time_ms > 0 else 1.0

        self.results.append(all_metrics)
        return all_metrics

    # =========================================================================
    # B2: FFN/CLS Savings
    # =========================================================================

    def test_b2_cls_savings(
        self,
        prompts: Optional[List[str]] = None,
    ) -> SavingsMetrics:
        """
        B2: Measure CLS (chunk-level sparsity) savings.

        Quantifies:
        - Expert reuse rate within chunks
        - Actual expert compute savings
        - Impact on wall-clock time
        """
        if prompts is None:
            prompts = [
                "The quick brown fox jumps over the lazy dog. " * 5,
                "In machine learning, neural networks are computational systems " * 3,
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
            ]

        self.load_model()

        from src.ffn.cls_ffn import (
            attach_cls_ffn,
            detach_cls_ffn,
            get_all_cls_statistics,
        )

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True
        config.collect_statistics = True

        all_metrics = SavingsMetrics()
        baseline_times = []
        adaptive_times = []
        reuse_rates = []
        compute_savings = []
        chunks_processed = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Baseline timing (no CLS)
            baseline_time = self._time_forward(
                self.model, inputs["input_ids"], inputs.get("attention_mask")
            )
            baseline_times.append(baseline_time)

            # With CLS
            attach_cls_ffn(self.model, config)

            adaptive_time = self._time_forward(
                self.model, inputs["input_ids"], inputs.get("attention_mask")
            )
            adaptive_times.append(adaptive_time)

            # Collect statistics
            cls_stats = get_all_cls_statistics(self.model)
            for layer_stats in cls_stats.values():
                reuse_rates.append(layer_stats.get("reuse_rate", 0))
                compute_savings.append(layer_stats.get("compute_savings", 0))
                chunks_processed.append(layer_stats.get("chunks_processed", 0))

            detach_cls_ffn(self.model)

        # Aggregate
        all_metrics.cls_expert_reuse_rate = sum(reuse_rates) / len(reuse_rates) if reuse_rates else 0
        all_metrics.cls_compute_savings = sum(compute_savings) / len(compute_savings) if compute_savings else 0
        all_metrics.cls_chunks_processed = sum(chunks_processed)

        all_metrics.baseline_time_ms = sum(baseline_times) / len(baseline_times)
        all_metrics.adaptive_time_ms = sum(adaptive_times) / len(adaptive_times)
        all_metrics.actual_speedup = all_metrics.baseline_time_ms / all_metrics.adaptive_time_ms if all_metrics.adaptive_time_ms > 0 else 1.0

        self.results.append(all_metrics)
        return all_metrics

    # =========================================================================
    # B3: Combined Throughput
    # =========================================================================

    def test_b3_combined_throughput(
        self,
        prompt: str = "The meaning of life is",
        num_tokens: int = 50,
        num_runs: int = 3,
    ) -> SavingsMetrics:
        """
        B3: End-to-end throughput with all optimizations.

        Measures actual tokens/second with full adaptive pipeline.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        # Baseline: no adaptive layers
        baseline_times = []
        for _ in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.perf_counter()

            with torch.no_grad():
                _ = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=num_tokens,
                    do_sample=False,
                )

            torch.cuda.synchronize() if self.device == "cuda" else None
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)

        # Adaptive: full pipeline
        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True
        config.collect_statistics = True

        engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

        adaptive_times = []
        stats_list = []
        for _ in range(num_runs):
            engine.reset_statistics()

            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.perf_counter()

            _, stats = engine.generate(
                prompt,
                max_new_tokens=num_tokens,
                do_sample=False,
                return_stats=True,
            )

            torch.cuda.synchronize() if self.device == "cuda" else None
            end = time.perf_counter()
            adaptive_times.append((end - start) * 1000)
            stats_list.append(stats)

        engine.detach_adaptive_layers()

        # Aggregate
        metrics = SavingsMetrics()
        metrics.baseline_time_ms = sum(baseline_times) / len(baseline_times)
        metrics.adaptive_time_ms = sum(adaptive_times) / len(adaptive_times)
        metrics.actual_speedup = metrics.baseline_time_ms / metrics.adaptive_time_ms if metrics.adaptive_time_ms > 0 else 1.0

        # From last stats
        last_stats = stats_list[-1]
        metrics.reduced_compute_rate = last_stats.reduced_compute_rate
        metrics.escalation_rate = last_stats.escalation_rate
        metrics.mean_sparsity = last_stats.mean_sparsity
        metrics.cls_expert_reuse_rate = last_stats.cls_expert_reuse_rate
        metrics.cls_compute_savings = last_stats.cls_compute_savings

        self.results.append(metrics)
        return metrics

    # =========================================================================
    # B4: Scaling Analysis
    # =========================================================================

    def test_b4_scaling_analysis(
        self,
        context_lengths: Optional[List[int]] = None,
    ) -> List[ScalingPoint]:
        """
        B4: How savings scale with context length.

        Key insight: Adaptive attention should provide MORE savings
        as context grows (O(n) vs O(n^2) for full attention).
        """
        if context_lengths is None:
            context_lengths = [32, 64, 128, 256, 512]

        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        scaling_points = []

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True
        config.collect_statistics = True

        base_prompt = "word "

        for ctx_len in context_lengths:
            # Create prompt of target length
            prompt = base_prompt * (ctx_len // 2)  # Approximate target length

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            actual_len = inputs["input_ids"].shape[1]

            if actual_len < 10:
                continue

            # Baseline timing
            baseline_time = self._time_forward(
                self.model, inputs["input_ids"], inputs.get("attention_mask")
            )

            # Adaptive
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

            with torch.no_grad():
                _ = self.model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))

            adaptive_time = self._time_forward(
                self.model, inputs["input_ids"], inputs.get("attention_mask")
            )

            # Get statistics
            all_stats = engine.get_all_statistics()
            attn_stats = all_stats.get("attention_layers", {})
            cls_stats = all_stats.get("cls_summary", {})

            # Compute savings
            mean_sparsity = 0
            for layer_stats in attn_stats.values():
                mean_sparsity += layer_stats.get("mean_sparsity", 0)
            if attn_stats:
                mean_sparsity /= len(attn_stats)

            engine.detach_adaptive_layers()

            point = ScalingPoint(
                context_length=actual_len,
                attention_savings=mean_sparsity,  # Proxy for attention savings
                cls_savings=cls_stats.get("compute_savings", 0),
                actual_speedup=baseline_time / adaptive_time if adaptive_time > 0 else 1.0,
                escalation_rate=0,  # Would need generation to measure
            )
            scaling_points.append(point)

            print(f"  Context {actual_len}: speedup={point.actual_speedup:.2f}x, "
                  f"attn_savings={point.attention_savings*100:.1f}%")

        return scaling_points

    # =========================================================================
    # Test Runner
    # =========================================================================

    def run_all(self) -> Dict[str, Any]:
        """Run all Phase B tests."""
        print("\n" + "=" * 70)
        print("PHASE B: COMPUTE SAVINGS QUANTIFICATION")
        print("=" * 70)
        print("\nPrecondition: Phase A must have passed.\n")

        results = {}

        # B1: Attention
        print("\n[B1] Attention Savings...")
        try:
            metrics = self.test_b1_attention_savings()
            print(metrics)
            results["B1_attention"] = metrics
        except Exception as e:
            print(f"[FAIL] B1: {e}")
            results["B1_attention"] = None

        # B2: CLS
        print("\n[B2] CLS/FFN Savings...")
        try:
            metrics = self.test_b2_cls_savings()
            print(metrics)
            results["B2_cls"] = metrics
        except Exception as e:
            print(f"[FAIL] B2: {e}")
            results["B2_cls"] = None

        # B3: Combined
        print("\n[B3] Combined Throughput...")
        try:
            metrics = self.test_b3_combined_throughput()
            print(metrics)
            results["B3_combined"] = metrics
        except Exception as e:
            print(f"[FAIL] B3: {e}")
            results["B3_combined"] = None

        # B4: Scaling
        print("\n[B4] Scaling Analysis...")
        try:
            points = self.test_b4_scaling_analysis()
            results["B4_scaling"] = points
            print("\nScaling summary:")
            for p in points:
                print(f"  len={p.context_length}: {p.actual_speedup:.2f}x speedup")
        except Exception as e:
            print(f"[FAIL] B4: {e}")
            results["B4_scaling"] = None

        print("\n" + "=" * 70)
        print("PHASE B COMPLETE")
        print("=" * 70)

        return results


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase B: Savings Quantification")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    suite = SavingsTestSuite(device=args.device)
    suite.run_all()
