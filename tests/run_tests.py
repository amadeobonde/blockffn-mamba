#!/usr/bin/env python3
"""
BlockFFN Adaptive Inference Test Runner

Staged test execution that enforces the testing philosophy:
1. Phase A (Correctness) MUST pass before Phase B
2. Phase B (Savings) should pass before claiming speedups
3. Phase C (Stress) reveals hidden failure modes

Usage:
    python tests/run_tests.py                    # Run all phases
    python tests/run_tests.py --phase A          # Run Phase A only
    python tests/run_tests.py --phase A B        # Run Phases A and B
    python tests/run_tests.py --skip-model-tests # Skip tests requiring model
    python tests/run_tests.py --device cuda      # Use GPU
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner(text: str, char: str = "=") -> None:
    """Print a banner."""
    print("\n" + char * 70)
    print(text)
    print(char * 70)


def run_phase_a(device: str, skip_model: bool = False) -> bool:
    """
    Run Phase A: Correctness & Safety tests.

    Returns:
        True if all tests passed
    """
    print_banner("PHASE A: CORRECTNESS & SAFETY")

    if skip_model:
        print("Skipping model-dependent tests (--skip-model-tests)")
        # Run only non-model tests
        from tests.test_a_correctness import CorrectnessTestSuite
        suite = CorrectnessTestSuite(device=device)

        # Run only decision maker tests (don't need model)
        print("\nRunning unit tests that don't require model loading...")

        results = []

        # A2.2: Yellow streak (no model needed)
        try:
            result = suite.test_a2_yellow_streak_detection()
            results.append(result)
            print(result)
        except Exception as e:
            print(f"[FAIL] A2.2: {e}")
            return False

        # A2.3: Cascade protection (no model needed)
        try:
            result = suite.test_a2_cascade_protection()
            results.append(result)
            print(result)
        except Exception as e:
            print(f"[FAIL] A2.3: {e}")
            return False

        return all(r.passed for r in results)

    else:
        from tests.test_a_correctness import CorrectnessTestSuite
        suite = CorrectnessTestSuite(device=device)
        return suite.run_all(stop_on_failure=True)


def run_phase_b(device: str, skip_model: bool = False) -> Dict[str, Any]:
    """
    Run Phase B: Compute Savings Quantification.

    Returns:
        Dict of results
    """
    print_banner("PHASE B: COMPUTE SAVINGS QUANTIFICATION")

    if skip_model:
        print("Skipping Phase B (requires model)")
        return {"skipped": True}

    from tests.test_b_savings import SavingsTestSuite
    suite = SavingsTestSuite(device=device)
    return suite.run_all()


def run_phase_c(device: str, skip_model: bool = False) -> bool:
    """
    Run Phase C: Stress Tests.

    Returns:
        True if all tests passed
    """
    print_banner("PHASE C: STRESS TESTS")

    if skip_model:
        print("Skipping model-dependent tests...")
        # Run only non-model tests
        from tests.test_c_stress import StressTestSuite
        suite = StressTestSuite(device=device)

        # A5.2: Escalation determinism (no model needed)
        try:
            result = suite.test_c5_escalation_determinism()
            print(result)
            return result.passed
        except Exception as e:
            print(f"[FAIL] C5.2: {e}")
            return False

    else:
        from tests.test_c_stress import StressTestSuite
        suite = StressTestSuite(device=device)
        return suite.run_all(stop_on_failure=False)


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save test results to JSON file."""
    results["timestamp"] = datetime.now().isoformat()

    # Convert non-serializable objects
    def convert(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        elif hasattr(obj, "name"):  # Enum
            return obj.name
        return str(obj)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="BlockFFN Adaptive Inference Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Testing Philosophy:
  1. Phase A (Correctness) MUST pass before Phase B
  2. Phase B (Savings) validates actual compute reduction
  3. Phase C (Stress) reveals hidden failure modes

If Phase A fails, any speedup numbers are MEANINGLESS.
        """,
    )

    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["A", "B", "C", "all"],
        default=["all"],
        help="Which phases to run (default: all)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to run tests on",
    )

    parser.add_argument(
        "--skip-model-tests",
        action="store_true",
        help="Skip tests that require loading the model (for quick CI)",
    )

    parser.add_argument(
        "--model",
        default="SparseLLM/BlockFFN-3B-SFT",
        help="Model to use for tests",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output file for test results (JSON)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Run later phases even if earlier phases fail",
    )

    args = parser.parse_args()

    # Determine which phases to run
    phases = set(args.phase)
    if "all" in phases:
        phases = {"A", "B", "C"}

    print_banner("BLOCKFFN ADAPTIVE INFERENCE TEST SUITE", "=")
    print(f"Device: {args.device}")
    print(f"Phases: {', '.join(sorted(phases))}")
    print(f"Skip model tests: {args.skip_model_tests}")

    results = {
        "device": args.device,
        "skip_model_tests": args.skip_model_tests,
        "phases_requested": list(phases),
    }

    # Run phases in order
    all_passed = True

    # Phase A
    if "A" in phases:
        phase_a_passed = run_phase_a(args.device, args.skip_model_tests)
        results["phase_a_passed"] = phase_a_passed

        if not phase_a_passed:
            all_passed = False
            if not args.force:
                print_banner("STOPPING: Phase A failed. Fix correctness issues first.", "*")
                print("\nPhase A failures mean the adaptive logic is broken.")
                print("Any speedup measurements would be meaningless.")
                print("\nUse --force to run later phases anyway (not recommended).")

                if args.output:
                    save_results(results, args.output)

                sys.exit(1)

    # Phase B
    if "B" in phases:
        if "A" not in phases:
            print("\nWarning: Running Phase B without Phase A.")
            print("Consider running Phase A first to verify correctness.\n")

        phase_b_results = run_phase_b(args.device, args.skip_model_tests)
        results["phase_b"] = phase_b_results

    # Phase C
    if "C" in phases:
        phase_c_passed = run_phase_c(args.device, args.skip_model_tests)
        results["phase_c_passed"] = phase_c_passed

        if not phase_c_passed:
            all_passed = False

    # Summary
    print_banner("TEST SUMMARY")

    if "A" in phases:
        status = "PASS" if results.get("phase_a_passed") else "FAIL"
        print(f"Phase A (Correctness): {status}")

    if "B" in phases:
        if results.get("phase_b", {}).get("skipped"):
            print("Phase B (Savings): SKIPPED")
        else:
            print("Phase B (Savings): COMPLETE (see metrics above)")

    if "C" in phases:
        status = "PASS" if results.get("phase_c_passed") else "FAIL"
        print(f"Phase C (Stress): {status}")

    print()

    if all_passed:
        print("All requested tests PASSED")
    else:
        print("Some tests FAILED - review output above")

    # Save results
    if args.output:
        save_results(results, args.output)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
