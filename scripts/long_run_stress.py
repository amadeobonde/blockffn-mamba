#!/usr/bin/env python3
"""
Long-Run Stress Testing

Extended stress testing for production readiness validation.
Runs for 1-2 hours with mixed batches and random prompts.

Monitors:
- Memory growth (should be < 5%)
- Rolling escalation rate
- Output determinism
- Stable statistics distributions

Usage:
    python scripts/long_run_stress.py --duration-hours 2 --output results/stress_test.json
"""

import argparse
import json
import gc
import os
import sys
import time
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import deque

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.config import AdaptiveInferenceConfig, PRESETS
from src.engine.adaptive_engine import AdaptiveInferenceEngine


@dataclass
class MemorySnapshot:
    """Memory usage at a point in time."""
    timestamp: str
    elapsed_minutes: float
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float


@dataclass
class RollingStats:
    """Rolling statistics over a window of tokens."""
    timestamp: str
    elapsed_minutes: float
    tokens_in_window: int
    escalation_rate: float
    green_rate: float
    mean_confidence: float
    avg_streak_length: float


@dataclass
class DeterminismCheck:
    """Result of a determinism check."""
    prompt_id: int
    is_deterministic: bool
    run1_tokens: List[int]
    run2_tokens: List[int]
    mismatch_position: Optional[int]


@dataclass
class StressTestResult:
    """Complete stress test result."""
    start_time: str
    end_time: str
    duration_hours: float
    total_prompts_processed: int
    total_tokens_generated: int

    # Memory tracking
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    initial_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_pct: float = 0.0

    # Rolling statistics
    rolling_stats: List[RollingStats] = field(default_factory=list)
    avg_escalation_rate: float = 0.0
    escalation_rate_std: float = 0.0

    # Determinism checks
    determinism_checks: List[DeterminismCheck] = field(default_factory=list)
    determinism_pass_rate: float = 0.0

    # Overall health
    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["memory_snapshots"] = [asdict(s) for s in self.memory_snapshots]
        d["rolling_stats"] = [asdict(s) for s in self.rolling_stats]
        d["determinism_checks"] = [asdict(c) for c in self.determinism_checks]
        return d


class LongRunStressTest:
    """
    Extended stress testing for BlockFFN adaptive inference.
    """

    def __init__(
        self,
        model_name: str = "SparseLLM/BlockFFN-3B-SFT",
        device: str = "mps",
        preset: str = "balanced",
        rolling_window_size: int = 1000,
        memory_sample_interval_minutes: float = 5.0,
        determinism_check_interval: int = 50,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.preset = preset
        self.rolling_window_size = rolling_window_size
        self.memory_sample_interval = memory_sample_interval_minutes
        self.determinism_check_interval = determinism_check_interval
        self.verbose = verbose

        self.model = None
        self.tokenizer = None
        self.engine = None

        # Rolling stats tracking
        self.confidence_window = deque(maxlen=rolling_window_size)
        self.escalation_window = deque(maxlen=rolling_window_size)
        self.streak_lengths = []

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

        config = PRESETS[self.preset].for_device(self.device)
        config.collect_statistics = True
        self.engine = AdaptiveInferenceEngine(self.model, self.tokenizer, config)

        print("Model loaded.")

    def get_memory_mb(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        elif self.device == "mps":
            # MPS doesn't have direct memory tracking like CUDA
            # Use system memory as a proxy
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            allocated = mem_info.rss / 1024 / 1024
            reserved = mem_info.vms / 1024 / 1024
            peak = allocated  # No peak tracking for MPS
        else:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            allocated = mem_info.rss / 1024 / 1024
            reserved = mem_info.vms / 1024 / 1024
            peak = allocated

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_mb": peak,
        }

    def get_prompts(self, count: int = 100) -> List[str]:
        """Get diverse prompts for testing."""
        from scripts.policy_validator import PromptSource

        # Mix of sources
        synthetic = PromptSource.get_synthetic_prompts(count // 2)
        try:
            wikitext = PromptSource.get_wikitext_prompts(count // 2)
        except Exception:
            wikitext = synthetic[:count // 2]

        prompts = synthetic + wikitext
        random.shuffle(prompts)
        return prompts[:count]

    def check_determinism(
        self,
        prompt: str,
        max_tokens: int = 30,
    ) -> DeterminismCheck:
        """
        Check if generation is deterministic.

        Runs the same prompt twice and compares outputs.
        """
        self.engine.reset_statistics()
        output1, _ = self.engine.generate(
            prompt, max_new_tokens=max_tokens, do_sample=False, return_stats=True
        )

        self.engine.reset_statistics()
        output2, _ = self.engine.generate(
            prompt, max_new_tokens=max_tokens, do_sample=False, return_stats=True
        )

        # Tokenize outputs
        tokens1 = self.tokenizer.encode(output1)
        tokens2 = self.tokenizer.encode(output2)

        is_deterministic = tokens1 == tokens2
        mismatch_pos = None

        if not is_deterministic:
            for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
                if t1 != t2:
                    mismatch_pos = i
                    break
            if mismatch_pos is None and len(tokens1) != len(tokens2):
                mismatch_pos = min(len(tokens1), len(tokens2))

        return DeterminismCheck(
            prompt_id=0,  # Will be set by caller
            is_deterministic=is_deterministic,
            run1_tokens=tokens1,
            run2_tokens=tokens2,
            mismatch_position=mismatch_pos,
        )

    def run_stress_test(
        self,
        duration_hours: float = 1.0,
        max_tokens: int = 50,
    ) -> StressTestResult:
        """
        Run extended stress test.

        Args:
            duration_hours: Duration of the test in hours
            max_tokens: Maximum tokens per generation

        Returns:
            StressTestResult with all metrics
        """
        self.load_model()

        result = StressTestResult(
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_hours=duration_hours,
            total_prompts_processed=0,
            total_tokens_generated=0,
        )

        # Get initial memory
        initial_mem = self.get_memory_mb()
        result.initial_memory_mb = initial_mem["allocated_mb"]
        result.memory_snapshots.append(MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            elapsed_minutes=0,
            allocated_mb=initial_mem["allocated_mb"],
            reserved_mb=initial_mem["reserved_mb"],
            peak_allocated_mb=initial_mem["peak_mb"],
        ))

        # Get prompts
        prompts = self.get_prompts(1000)  # Large pool to cycle through

        start_time = time.time()
        duration_seconds = duration_hours * 3600
        last_memory_sample = start_time
        last_rolling_stat = start_time
        prompt_idx = 0
        determinism_counter = 0

        print(f"\nStarting stress test for {duration_hours} hours...")
        print(f"Memory sample interval: {self.memory_sample_interval} minutes")
        print(f"Rolling window size: {self.rolling_window_size} tokens")
        print()

        while (time.time() - start_time) < duration_seconds:
            # Get next prompt
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Generate
            self.engine.reset_statistics()
            try:
                _, stats = self.engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    return_stats=True,
                )

                result.total_prompts_processed += 1
                result.total_tokens_generated += stats.total_tokens

                # Update rolling windows
                for conf in self.engine._confidence_history:
                    self.confidence_window.append(conf)
                    if conf >= self.engine.config.tau_yellow:
                        self.escalation_window.append(0)
                    else:
                        self.escalation_window.append(1)

            except Exception as e:
                result.errors.append(f"Generation error at prompt {prompt_idx}: {str(e)}")
                continue

            # Determinism check
            determinism_counter += 1
            if determinism_counter >= self.determinism_check_interval:
                determinism_counter = 0
                check = self.check_determinism(prompt, max_tokens=20)
                check.prompt_id = result.total_prompts_processed
                result.determinism_checks.append(check)

                if not check.is_deterministic:
                    result.warnings.append(
                        f"Non-deterministic output at prompt {check.prompt_id}"
                    )

            # Memory sample
            current_time = time.time()
            if (current_time - last_memory_sample) >= (self.memory_sample_interval * 60):
                last_memory_sample = current_time
                elapsed_minutes = (current_time - start_time) / 60

                mem = self.get_memory_mb()
                result.memory_snapshots.append(MemorySnapshot(
                    timestamp=datetime.now().isoformat(),
                    elapsed_minutes=elapsed_minutes,
                    allocated_mb=mem["allocated_mb"],
                    reserved_mb=mem["reserved_mb"],
                    peak_allocated_mb=mem["peak_mb"],
                ))

                if self.verbose:
                    print(f"  [{elapsed_minutes:.1f} min] Memory: {mem['allocated_mb']:.1f} MB, "
                          f"Prompts: {result.total_prompts_processed}, "
                          f"Tokens: {result.total_tokens_generated}")

                # Force garbage collection
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # Rolling stats sample (every minute)
            if (current_time - last_rolling_stat) >= 60:
                last_rolling_stat = current_time
                elapsed_minutes = (current_time - start_time) / 60

                if len(self.confidence_window) > 0:
                    config = self.engine.config
                    green_count = sum(1 for c in self.confidence_window if c >= config.tau_green)
                    esc_rate = sum(self.escalation_window) / len(self.escalation_window)

                    result.rolling_stats.append(RollingStats(
                        timestamp=datetime.now().isoformat(),
                        elapsed_minutes=elapsed_minutes,
                        tokens_in_window=len(self.confidence_window),
                        escalation_rate=esc_rate,
                        green_rate=green_count / len(self.confidence_window),
                        mean_confidence=sum(self.confidence_window) / len(self.confidence_window),
                        avg_streak_length=0,  # Would need proper tracking
                    ))

        # Finalize
        result.end_time = datetime.now().isoformat()

        # Final memory sample
        final_mem = self.get_memory_mb()
        result.final_memory_mb = final_mem["allocated_mb"]
        result.peak_memory_mb = max(s.peak_allocated_mb for s in result.memory_snapshots)

        # Calculate memory growth
        if result.initial_memory_mb > 0:
            result.memory_growth_pct = (
                (result.final_memory_mb - result.initial_memory_mb) /
                result.initial_memory_mb
            )

        # Calculate overall stats
        if result.rolling_stats:
            esc_rates = [s.escalation_rate for s in result.rolling_stats]
            result.avg_escalation_rate = sum(esc_rates) / len(esc_rates)
            if len(esc_rates) > 1:
                mean = result.avg_escalation_rate
                result.escalation_rate_std = (
                    sum((r - mean) ** 2 for r in esc_rates) / len(esc_rates)
                ) ** 0.5

        # Determinism pass rate
        if result.determinism_checks:
            passed = sum(1 for c in result.determinism_checks if c.is_deterministic)
            result.determinism_pass_rate = passed / len(result.determinism_checks)

        # Check health
        if result.memory_growth_pct > 0.05:
            result.warnings.append(f"Memory growth {result.memory_growth_pct*100:.1f}% exceeds 5%")
            result.is_healthy = False

        if result.determinism_pass_rate < 1.0:
            result.warnings.append(f"Determinism pass rate {result.determinism_pass_rate*100:.1f}% < 100%")

        if result.errors:
            result.is_healthy = False

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Long-Run Stress Testing for BlockFFN"
    )
    parser.add_argument("--duration-hours", type=float, default=1.0,
                        help="Duration of stress test in hours")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--preset", default="balanced",
                        help="Preset to test")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens per generation")
    parser.add_argument("--output", type=str, default="results/stress_test.json",
                        help="Output file path")
    parser.add_argument("--model", default="SparseLLM/BlockFFN-3B-SFT")
    parser.add_argument("--memory-interval", type=float, default=5.0,
                        help="Memory sample interval in minutes")
    parser.add_argument("--rolling-window", type=int, default=1000,
                        help="Rolling statistics window size")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print("LONG-RUN STRESS TEST")
    print("=" * 70)
    print(f"Duration: {args.duration_hours} hours")
    print(f"Device: {args.device}")
    print(f"Preset: {args.preset}")
    print()

    # Run stress test
    tester = LongRunStressTest(
        model_name=args.model,
        device=args.device,
        preset=args.preset,
        rolling_window_size=args.rolling_window,
        memory_sample_interval_minutes=args.memory_interval,
        verbose=args.verbose,
    )

    result = tester.run_stress_test(
        duration_hours=args.duration_hours,
        max_tokens=args.max_tokens,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"Duration: {result.duration_hours:.2f} hours")
    print(f"Prompts processed: {result.total_prompts_processed}")
    print(f"Tokens generated: {result.total_tokens_generated}")
    print()
    print("MEMORY:")
    print(f"  Initial: {result.initial_memory_mb:.1f} MB")
    print(f"  Final: {result.final_memory_mb:.1f} MB")
    print(f"  Peak: {result.peak_memory_mb:.1f} MB")
    print(f"  Growth: {result.memory_growth_pct*100:.1f}%")
    print()
    print("QUALITY:")
    print(f"  Avg escalation rate: {result.avg_escalation_rate:.3f}")
    print(f"  Escalation rate std: {result.escalation_rate_std:.4f}")
    print(f"  Determinism pass rate: {result.determinism_pass_rate*100:.1f}%")
    print()

    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.errors:
        print("ERRORS:")
        for e in result.errors:
            print(f"  - {e}")

    print()
    if result.is_healthy:
        print(">>> STATUS: HEALTHY - System passed stress test <<<")
    else:
        print(">>> STATUS: UNHEALTHY - See warnings/errors above <<<")


if __name__ == "__main__":
    main()
