"""
Phase C: Stress Tests

These tests probe failure modes and edge cases that kill most projects.

C1. Batch processing stress
C2. Memory pressure tests
C3. Adversarial inputs (designed to break adaptive logic)
C4. State corruption detection
C5. Determinism verification
"""

import torch
import time
import gc
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS


@dataclass
class StressTestResult:
    """Result from a stress test."""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f"[{status}] {self.test_name}"
        if self.error_message:
            msg += f"\n  Error: {self.error_message}"
        if self.metrics:
            msg += "\n  Metrics:"
            for k, v in self.metrics.items():
                msg += f"\n    {k}: {v}"
        return msg


class StressTestSuite:
    """
    Phase C: Stress Tests

    These tests probe failure modes that kill most optimization projects:
    - Batch processing edge cases
    - Memory pressure and leaks
    - Adversarial inputs designed to break adaptive logic
    - State corruption across calls
    - Non-determinism bugs
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
        self.results: List[StressTestResult] = []

    def load_model(self, model_name: str = "SparseLLM/BlockFFN-3B-SFT") -> None:
        """Load model if not provided."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        print("Model loaded.")

    # =========================================================================
    # C1: Batch Processing Stress
    # =========================================================================

    def test_c1_variable_length_batch(self) -> StressTestResult:
        """
        C1.1: Variable-length sequences in a batch.

        Tests that adaptive attention handles padding correctly
        when batch elements have different lengths.
        """
        self.load_model()

        from src.attention.adaptive_attention import (
            attach_adaptive_attention,
            detach_adaptive_attention,
        )

        config = AdaptiveInferenceConfig.from_preset("balanced")

        # Create batch with varying lengths
        prompts = [
            "Short.",
            "This is a medium length sentence for testing purposes.",
            "This is a much longer sentence that should have significantly more tokens " * 3,
        ]

        try:
            # Tokenize with padding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Baseline
            with torch.no_grad():
                baseline_out = self.model(**inputs)
                baseline_logits = baseline_out.logits

            # Adaptive
            attach_adaptive_attention(self.model, config)
            with torch.no_grad():
                adaptive_out = self.model(**inputs)
                adaptive_logits = adaptive_out.logits
            detach_adaptive_attention(self.model)

            # Check that outputs have same shape
            shape_match = baseline_logits.shape == adaptive_logits.shape

            # Check non-padded positions match
            # (Padded positions may differ due to masking)
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(baseline_logits)

            # Only compare where not padded
            baseline_masked = baseline_logits * mask_expanded
            adaptive_masked = adaptive_logits * mask_expanded

            diff = (baseline_masked - adaptive_masked).abs().max().item()

            passed = shape_match and diff < 0.1

            return StressTestResult(
                test_name="C1.1 Variable-Length Batch",
                passed=passed,
                metrics={
                    "shape_match": shape_match,
                    "max_diff_unpadded": diff,
                    "batch_size": len(prompts),
                    "seq_lengths": [inputs["attention_mask"][i].sum().item() for i in range(len(prompts))],
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C1.1 Variable-Length Batch",
                passed=False,
                error_message=str(e),
            )

    def test_c1_large_batch(self) -> StressTestResult:
        """
        C1.2: Large batch size stress test.

        Tests memory handling with larger batches.
        """
        self.load_model()

        from src.attention.adaptive_attention import (
            attach_adaptive_attention,
            detach_adaptive_attention,
        )

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.collect_statistics = False  # Reduce overhead

        # Try progressively larger batches
        batch_sizes = [1, 2, 4, 8]
        if self.device == "cuda":
            batch_sizes.extend([16, 32])

        max_successful_batch = 0
        error_msg = None

        prompt = "The quick brown fox jumps over the lazy dog."

        attach_adaptive_attention(self.model, config)

        try:
            for batch_size in batch_sizes:
                try:
                    inputs = self.tokenizer(
                        [prompt] * batch_size,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                    with torch.no_grad():
                        _ = self.model(**inputs)

                    max_successful_batch = batch_size

                    # Clean up
                    del inputs
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    raise

        except Exception as e:
            error_msg = str(e)
        finally:
            detach_adaptive_attention(self.model)

        # Should handle at least batch size 4
        passed = max_successful_batch >= 4

        return StressTestResult(
            test_name="C1.2 Large Batch Stress",
            passed=passed,
            error_message=error_msg,
            metrics={
                "max_successful_batch": max_successful_batch,
                "target_minimum": 4,
            },
        )

    # =========================================================================
    # C2: Memory Pressure
    # =========================================================================

    def test_c2_memory_leak_detection(self) -> StressTestResult:
        """
        C2.1: Memory leak detection.

        Run multiple iterations and check for memory growth.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True

        prompt = "The meaning of life is"
        num_iterations = 10

        # Track memory
        memory_samples = []

        def get_memory():
            if self.device == "cuda":
                return torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                import psutil
                return psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            # Initial memory
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            initial_memory = get_memory()
            memory_samples.append(initial_memory)

            for i in range(num_iterations):
                engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

                # Generate
                _ = engine.generate(prompt, max_new_tokens=20, do_sample=False)

                # Clean up
                engine.detach_adaptive_layers()
                del engine

                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                memory_samples.append(get_memory())

            # Check for significant memory growth
            # Allow some variance, but flag if trending upward
            final_memory = memory_samples[-1]
            memory_growth = final_memory - initial_memory

            # Growth should be minimal (< 10% of initial)
            growth_ratio = memory_growth / initial_memory if initial_memory > 0 else 0
            passed = growth_ratio < 0.1

            return StressTestResult(
                test_name="C2.1 Memory Leak Detection",
                passed=passed,
                metrics={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "growth_mb": memory_growth,
                    "growth_ratio": growth_ratio,
                    "samples": memory_samples,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C2.1 Memory Leak Detection",
                passed=False,
                error_message=str(e),
            )

    def test_c2_cache_cleanup(self) -> StressTestResult:
        """
        C2.2: Verify caches are properly cleaned up.

        Check that expert result caches don't grow unbounded.
        """
        self.load_model()

        from src.ffn.cls_ffn import attach_cls_ffn, detach_cls_ffn, get_cls_layers
        from src.ffn.expert_batcher import ChunkExpertBatcher

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True
        config.expert_cache_mb = 64  # Small cache for testing

        try:
            attach_cls_ffn(self.model, config)

            # Get CLS layers
            cls_layers = get_cls_layers(self.model)

            # Run several forward passes
            prompts = [
                "Short prompt one.",
                "Slightly longer prompt for testing cache behavior.",
                "An even longer prompt to see how the cache handles multiple sequences " * 2,
            ]

            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    _ = self.model(**inputs)

            # Check cache sizes
            cache_sizes = []
            for idx, cls_layer in cls_layers:
                if hasattr(cls_layer, "chunk_batcher"):
                    batcher = cls_layer.chunk_batcher
                    if hasattr(batcher, "cache") and batcher.cache is not None:
                        cache_sizes.append(batcher.cache.current_size_bytes)

            detach_cls_ffn(self.model)

            # All caches should be within limit
            max_cache_mb = max(cache_sizes) / (1024 * 1024) if cache_sizes else 0
            passed = max_cache_mb <= config.expert_cache_mb

            return StressTestResult(
                test_name="C2.2 Cache Cleanup",
                passed=passed,
                metrics={
                    "max_cache_mb": max_cache_mb,
                    "limit_mb": config.expert_cache_mb,
                    "num_caches": len(cache_sizes),
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C2.2 Cache Cleanup",
                passed=False,
                error_message=str(e),
            )

    # =========================================================================
    # C3: Adversarial Inputs
    # =========================================================================

    def test_c3_all_easy_sequence(self) -> StressTestResult:
        """
        C3.1: Sequence designed to appear "all easy".

        Repetitive, predictable tokens that should trigger
        maximum window reduction.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        config = AdaptiveInferenceConfig.from_preset("aggressive")
        config.enable_cls_optimization = True

        # Highly repetitive prompt
        prompt = "the " * 50

        try:
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

            output, stats = engine.generate(
                prompt,
                max_new_tokens=20,
                do_sample=False,
                return_stats=True,
            )

            engine.detach_adaptive_layers()

            # Should have high reduced compute rate
            # But output should still be coherent
            passed = (
                stats.reduced_compute_rate > 0.5 and  # Most tokens reduced
                stats.escalation_rate < 0.3  # Not too many escalations
            )

            return StressTestResult(
                test_name="C3.1 All-Easy Sequence",
                passed=passed,
                metrics={
                    "reduced_compute_rate": stats.reduced_compute_rate,
                    "escalation_rate": stats.escalation_rate,
                    "mean_confidence": stats.mean_confidence,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C3.1 All-Easy Sequence",
                passed=False,
                error_message=str(e),
            )

    def test_c3_all_hard_sequence(self) -> StressTestResult:
        """
        C3.2: Sequence designed to appear "all hard".

        Complex reasoning that should trigger maximum escalation.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        config = AdaptiveInferenceConfig.from_preset("conservative")
        config.enable_cls_optimization = True

        # Complex reasoning prompt
        prompt = """Given that:
1. All mammals are warm-blooded
2. All dogs are mammals
3. Fido is a dog
4. Warm-blooded animals need food to survive

What can we conclude about Fido? Let's reason step by step:"""

        try:
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

            output, stats = engine.generate(
                prompt,
                max_new_tokens=50,
                do_sample=False,
                return_stats=True,
            )

            engine.detach_adaptive_layers()

            # Escalation should be higher for hard sequences
            # But should still produce output
            passed = stats.total_tokens > 0  # At minimum, we got output

            return StressTestResult(
                test_name="C3.2 All-Hard Sequence",
                passed=passed,
                metrics={
                    "total_tokens": stats.total_tokens,
                    "escalation_rate": stats.escalation_rate,
                    "mean_confidence": stats.mean_confidence,
                    "reduced_compute_rate": stats.reduced_compute_rate,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C3.2 All-Hard Sequence",
                passed=False,
                error_message=str(e),
            )

    def test_c3_mixed_difficulty(self) -> StressTestResult:
        """
        C3.3: Sequence with abrupt difficulty transitions.

        Tests handling of sudden changes from easy to hard.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True

        # Easy followed by hard
        prompt = """The cat sat on the mat. The dog ran in the park. The bird sang a song.

Now solve this differential equation: dy/dx + y*cos(x) = sin(x)*cos(x)

Solution:"""

        try:
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

            output, stats = engine.generate(
                prompt,
                max_new_tokens=30,
                do_sample=False,
                return_stats=True,
            )

            engine.detach_adaptive_layers()

            # Should handle transition gracefully
            passed = (
                stats.total_tokens > 0 and
                stats.escalation_rate < 0.8  # Not cascading failures
            )

            return StressTestResult(
                test_name="C3.3 Mixed Difficulty Transition",
                passed=passed,
                metrics={
                    "total_tokens": stats.total_tokens,
                    "escalation_rate": stats.escalation_rate,
                    "reduced_compute_rate": stats.reduced_compute_rate,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C3.3 Mixed Difficulty Transition",
                passed=False,
                error_message=str(e),
            )

    # =========================================================================
    # C4: State Corruption Detection
    # =========================================================================

    def test_c4_repeated_attach_detach(self) -> StressTestResult:
        """
        C4.1: Repeated attach/detach cycles.

        Verify no state corruption from multiple wrapping/unwrapping.
        """
        self.load_model()

        from src.attention.adaptive_attention import (
            attach_adaptive_attention,
            detach_adaptive_attention,
        )
        from src.ffn.cls_ffn import attach_cls_ffn, detach_cls_ffn

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True

        prompt = "The quick brown fox"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            # Get baseline
            with torch.no_grad():
                baseline_out = self.model(**inputs)
                baseline_logits = baseline_out.logits

            # Multiple attach/detach cycles
            for cycle in range(5):
                attach_adaptive_attention(self.model, config)
                attach_cls_ffn(self.model, config)

                with torch.no_grad():
                    _ = self.model(**inputs)

                detach_cls_ffn(self.model)
                detach_adaptive_attention(self.model)

            # Check baseline is restored
            with torch.no_grad():
                restored_out = self.model(**inputs)
                restored_logits = restored_out.logits

            # Should be identical to baseline
            diff = (baseline_logits - restored_logits).abs().max().item()
            passed = diff < 1e-5

            return StressTestResult(
                test_name="C4.1 Repeated Attach/Detach",
                passed=passed,
                metrics={
                    "cycles": 5,
                    "max_diff_after_restore": diff,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C4.1 Repeated Attach/Detach",
                passed=False,
                error_message=str(e),
            )

    def test_c4_statistics_isolation(self) -> StressTestResult:
        """
        C4.2: Statistics don't leak between runs.

        Verify reset_statistics actually clears all state.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True
        config.collect_statistics = True

        try:
            engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

            # Run 1
            _, stats1 = engine.generate("Hello world", max_new_tokens=10, return_stats=True)
            tokens1 = stats1.total_tokens

            # Reset
            engine.reset_statistics()

            # Verify reset
            zero_after_reset = engine.stats.total_tokens == 0

            # Run 2
            _, stats2 = engine.generate("Goodbye world", max_new_tokens=10, return_stats=True)
            tokens2 = stats2.total_tokens

            engine.detach_adaptive_layers()

            # Stats should be independent
            passed = zero_after_reset and tokens2 > 0

            return StressTestResult(
                test_name="C4.2 Statistics Isolation",
                passed=passed,
                metrics={
                    "run1_tokens": tokens1,
                    "zero_after_reset": zero_after_reset,
                    "run2_tokens": tokens2,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C4.2 Statistics Isolation",
                passed=False,
                error_message=str(e),
            )

    # =========================================================================
    # C5: Determinism
    # =========================================================================

    def test_c5_deterministic_output(self) -> StressTestResult:
        """
        C5.1: Same input produces same output.

        Verify no non-determinism in adaptive logic.
        """
        self.load_model()

        from src.engine.adaptive_engine import AdaptiveInferenceEngine

        config = AdaptiveInferenceConfig.from_preset("balanced")
        config.enable_cls_optimization = True

        prompt = "The capital of France is"
        num_runs = 3

        try:
            outputs = []

            for _ in range(num_runs):
                engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

                # Set seed for reproducibility
                torch.manual_seed(42)

                output = engine.generate(
                    prompt,
                    max_new_tokens=10,
                    do_sample=False,  # Greedy
                )
                outputs.append(output)

                engine.detach_adaptive_layers()

            # All outputs should be identical
            all_same = all(o == outputs[0] for o in outputs)

            return StressTestResult(
                test_name="C5.1 Deterministic Output",
                passed=all_same,
                metrics={
                    "num_runs": num_runs,
                    "outputs": outputs,
                    "all_identical": all_same,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C5.1 Deterministic Output",
                passed=False,
                error_message=str(e),
            )

    def test_c5_escalation_determinism(self) -> StressTestResult:
        """
        C5.2: Escalation decisions are deterministic.

        Same confidence sequence produces same escalation pattern.
        """
        from src.verification.decision import EscalationDecisionMaker

        config = AdaptiveInferenceConfig.from_preset("balanced")

        # Fixed confidence sequence
        confidences = [0.90, 0.75, 0.80, 0.72, 0.71, 0.70, 0.85, 0.60]

        try:
            decision_sequences = []

            for _ in range(3):
                dm = EscalationDecisionMaker(config)
                decisions = []
                for i, conf in enumerate(confidences):
                    d = dm.decide(conf, position=i)
                    decisions.append(d.level.name)
                decision_sequences.append(decisions)

            # All should be identical
            all_same = all(seq == decision_sequences[0] for seq in decision_sequences)

            return StressTestResult(
                test_name="C5.2 Escalation Determinism",
                passed=all_same,
                metrics={
                    "sequences": decision_sequences,
                    "all_identical": all_same,
                },
            )

        except Exception as e:
            return StressTestResult(
                test_name="C5.2 Escalation Determinism",
                passed=False,
                error_message=str(e),
            )

    # =========================================================================
    # Test Runner
    # =========================================================================

    def run_all(self, stop_on_failure: bool = False) -> bool:
        """
        Run all Phase C tests.

        Args:
            stop_on_failure: If True, stop at first failure

        Returns:
            True if all tests passed
        """
        print("\n" + "=" * 70)
        print("PHASE C: STRESS TESTS")
        print("=" * 70)
        print("\nThese tests probe failure modes that kill most projects.\n")

        tests = [
            # C1: Batch
            ("C1.1", self.test_c1_variable_length_batch),
            ("C1.2", self.test_c1_large_batch),
            # C2: Memory
            ("C2.1", self.test_c2_memory_leak_detection),
            ("C2.2", self.test_c2_cache_cleanup),
            # C3: Adversarial
            ("C3.1", self.test_c3_all_easy_sequence),
            ("C3.2", self.test_c3_all_hard_sequence),
            ("C3.3", self.test_c3_mixed_difficulty),
            # C4: State
            ("C4.1", self.test_c4_repeated_attach_detach),
            ("C4.2", self.test_c4_statistics_isolation),
            # C5: Determinism
            ("C5.1", self.test_c5_deterministic_output),
            ("C5.2", self.test_c5_escalation_determinism),
        ]

        passed_count = 0
        failed_count = 0

        for test_id, test_fn in tests:
            print(f"\nRunning {test_id}...")
            try:
                result = test_fn()
                self.results.append(result)
                print(result)

                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    if stop_on_failure:
                        print(f"\n*** STOP: {test_id} failed ***")
                        break
            except Exception as e:
                print(f"[FAIL] {test_id}: Exception - {e}")
                failed_count += 1
                if stop_on_failure:
                    raise

        print("\n" + "=" * 70)
        print(f"PHASE C SUMMARY: {passed_count} passed, {failed_count} failed")
        print("=" * 70)

        return failed_count == 0


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase C: Stress Tests")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    suite = StressTestSuite(device=args.device)

    success = suite.run_all(stop_on_failure=args.stop_on_failure)
    sys.exit(0 if success else 1)
