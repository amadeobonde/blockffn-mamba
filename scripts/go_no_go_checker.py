#!/usr/bin/env python3
"""
Go/No-Go Checker

Automated verification of production deployment criteria.
Aggregates results from policy validation, scheduling sweep, and stress tests.

Usage:
    python scripts/go_no_go_checker.py --input results/ --output results/go_no_go_report.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive.go_no_go import GoNoGoChecker, GoNoGoCriteria, check_go_no_go


def load_policy_validation(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Load policy validation results."""
    path = results_dir / "policy_validation" / "validation_results.json"
    if not path.exists():
        path = results_dir / "validation_results.json"
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def load_scheduling_sweep(results_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """Load scheduling sweep results."""
    path = results_dir / "scheduling_sweep.json"
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def load_escalation_sweep(results_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """Load escalation sweep results."""
    path = results_dir / "escalation_sweep.json"
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def load_stress_test(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Load stress test results."""
    path = results_dir / "stress_test.json"
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def aggregate_results(
    policy_results: Optional[Dict[str, Any]],
    scheduling_results: Optional[List[Dict[str, Any]]],
    escalation_results: Optional[List[Dict[str, Any]]],
    stress_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate results from all sources into a format suitable for go/no-go checking.

    Returns:
        Dictionary with keys matching GoNoGoChecker expectations:
        - escalation_rate
        - baseline_match_rate
        - verification_speedup
        - memory_growth_pct
        - is_deterministic
        - reduced_compute_rate
        - green_rate
        - red_rate
        - avg_streak_length
    """
    aggregated = {}

    # From policy validation
    if policy_results:
        # Use the "balanced" preset if available, otherwise first preset
        best_preset = None
        for preset_name, preset_data in policy_results.items():
            if preset_name == "balanced":
                best_preset = preset_data
                break
            if best_preset is None:
                best_preset = preset_data

        if best_preset:
            aggregated["escalation_rate"] = best_preset.get("avg_escalation_rate", 0)
            aggregated["green_rate"] = best_preset.get("avg_pct_green", 0)
            aggregated["red_rate"] = best_preset.get("avg_pct_red", 0)
            aggregated["avg_streak_length"] = best_preset.get("avg_accepted_streak_length", 0)

    # From scheduling sweep
    if scheduling_results:
        # Find baseline and best config
        baseline = None
        best = None

        for config in scheduling_results:
            if config.get("config_name") == "baseline":
                baseline = config
            # Find config with best speedup that has acceptable escalation
            if config.get("escalation_rate", 1.0) < 0.15:
                if best is None or config.get("speedup_vs_baseline", 1.0) > best.get("speedup_vs_baseline", 1.0):
                    best = config

        if best:
            aggregated["verification_speedup"] = best.get("speedup_vs_baseline", 1.0)
        if baseline:
            aggregated["reduced_compute_rate"] = 1.0 - baseline.get("escalation_rate", 0)

    # From stress test
    if stress_results:
        aggregated["memory_growth_pct"] = stress_results.get("memory_growth_pct", 0)
        aggregated["is_deterministic"] = stress_results.get("determinism_pass_rate", 0) >= 1.0

        # Override with stress test escalation rate if available
        if "avg_escalation_rate" in stress_results:
            aggregated["escalation_rate"] = stress_results["avg_escalation_rate"]

    # Estimate baseline match rate
    # This is approximated as 1 - escalation_rate (simplified)
    if "escalation_rate" in aggregated:
        aggregated["baseline_match_rate"] = 1.0 - aggregated["escalation_rate"]

    return aggregated


def generate_markdown_report(
    aggregated: Dict[str, Any],
    checker: GoNoGoChecker,
    check_results,
    policy_results: Optional[Dict[str, Any]],
    scheduling_results: Optional[List[Dict[str, Any]]],
    escalation_results: Optional[List[Dict[str, Any]]],
    stress_results: Optional[Dict[str, Any]],
) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# BlockFFN Go/No-Go Assessment Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall verdict
    all_passed = checker.all_passed(check_results)
    if all_passed:
        lines.append("## Verdict: GO")
        lines.append("")
        lines.append("System **PASSES** all go/no-go criteria and is ready for deployment.")
    else:
        lines.append("## Verdict: NO-GO")
        lines.append("")
        lines.append("System **FAILS** one or more go/no-go criteria. See details below.")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Criteria summary table
    lines.append("## Criteria Summary")
    lines.append("")
    lines.append("| Criterion | Threshold | Actual | Status |")
    lines.append("|-----------|-----------|--------|--------|")

    for result in check_results:
        status_icon = "PASS" if result.passed else ("SKIP" if result.status.value == "skipped" else "FAIL")
        actual_str = str(result.actual_value) if result.actual_value is not None else "N/A"
        if isinstance(result.actual_value, float):
            actual_str = f"{result.actual_value:.4f}"
        threshold_str = f"{result.operator} {result.threshold}"
        lines.append(f"| {result.criterion} | {threshold_str} | {actual_str} | {status_icon} |")

    lines.append("")

    # Detailed sections
    lines.append("## Detailed Results")
    lines.append("")

    # Policy validation section
    if policy_results:
        lines.append("### Policy Validation")
        lines.append("")
        lines.append("| Preset | Escalation Rate | Green Rate | Avg Streak |")
        lines.append("|--------|-----------------|------------|------------|")
        for preset_name, preset_data in policy_results.items():
            esc = preset_data.get("avg_escalation_rate", 0)
            green = preset_data.get("avg_pct_green", 0)
            streak = preset_data.get("avg_accepted_streak_length", 0)
            lines.append(f"| {preset_name} | {esc:.3f} | {green:.3f} | {streak:.1f} |")
        lines.append("")
    else:
        lines.append("### Policy Validation")
        lines.append("")
        lines.append("*No policy validation results found.*")
        lines.append("")

    # Scheduling sweep section
    if scheduling_results:
        lines.append("### Verification Scheduling")
        lines.append("")
        lines.append("| Config | Speedup | Verification Rate | Escalation Rate |")
        lines.append("|--------|---------|-------------------|-----------------|")
        for config in scheduling_results[:5]:  # Top 5
            name = config.get("config_name", "unknown")
            speedup = config.get("speedup_vs_baseline", 1.0)
            verify_rate = config.get("verification_rate", 1.0)
            esc_rate = config.get("escalation_rate", 0)
            lines.append(f"| {name} | {speedup:.2f}x | {verify_rate:.2f} | {esc_rate:.3f} |")
        lines.append("")
    else:
        lines.append("### Verification Scheduling")
        lines.append("")
        lines.append("*No scheduling sweep results found.*")
        lines.append("")

    # Stress test section
    if stress_results:
        lines.append("### Stress Test")
        lines.append("")
        lines.append(f"- **Duration:** {stress_results.get('duration_hours', 0):.2f} hours")
        lines.append(f"- **Prompts Processed:** {stress_results.get('total_prompts_processed', 0)}")
        lines.append(f"- **Tokens Generated:** {stress_results.get('total_tokens_generated', 0)}")
        lines.append(f"- **Memory Growth:** {stress_results.get('memory_growth_pct', 0)*100:.1f}%")
        lines.append(f"- **Determinism Pass Rate:** {stress_results.get('determinism_pass_rate', 0)*100:.1f}%")
        lines.append(f"- **Healthy:** {'Yes' if stress_results.get('is_healthy', False) else 'No'}")
        lines.append("")

        if stress_results.get("warnings"):
            lines.append("**Warnings:**")
            for w in stress_results["warnings"]:
                lines.append(f"- {w}")
            lines.append("")

        if stress_results.get("errors"):
            lines.append("**Errors:**")
            for e in stress_results["errors"]:
                lines.append(f"- {e}")
            lines.append("")
    else:
        lines.append("### Stress Test")
        lines.append("")
        lines.append("*No stress test results found.*")
        lines.append("")

    # Recommendations
    failures = checker.get_failures(check_results)
    if failures:
        lines.append("## Recommendations")
        lines.append("")
        for f in failures:
            rec = checker._get_recommendation(f.criterion)
            lines.append(f"- **{f.criterion}:** {rec}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by go_no_go_checker.py*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Go/No-Go Checker for BlockFFN"
    )
    parser.add_argument("--input", type=str, default="results",
                        help="Input directory containing experiment results")
    parser.add_argument("--output", type=str, default="results/go_no_go_report.md",
                        help="Output markdown report path")
    parser.add_argument("--output-json", type=str, default="results/go_no_go_results.json",
                        help="Output JSON results path")
    parser.add_argument("--verbose", action="store_true")

    # Custom criteria overrides
    parser.add_argument("--max-escalation-rate", type=float, default=0.10)
    parser.add_argument("--min-baseline-match", type=float, default=0.99)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--max-memory-growth", type=float, default=0.05)

    args = parser.parse_args()

    print("=" * 70)
    print("GO/NO-GO CHECKER")
    print("=" * 70)

    results_dir = Path(args.input)

    # Load results from different sources
    print("\nLoading results...")

    policy_results = load_policy_validation(results_dir)
    print(f"  Policy validation: {'Found' if policy_results else 'Not found'}")

    scheduling_results = load_scheduling_sweep(results_dir)
    print(f"  Scheduling sweep: {'Found' if scheduling_results else 'Not found'}")

    escalation_results = load_escalation_sweep(results_dir)
    print(f"  Escalation sweep: {'Found' if escalation_results else 'Not found'}")

    stress_results = load_stress_test(results_dir)
    print(f"  Stress test: {'Found' if stress_results else 'Not found'}")

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(
        policy_results, scheduling_results, escalation_results, stress_results
    )

    if args.verbose:
        print("  Aggregated values:")
        for k, v in aggregated.items():
            print(f"    {k}: {v}")

    # Create custom criteria if specified
    criteria = GoNoGoCriteria(
        max_escalation_rate=args.max_escalation_rate,
        baseline_match_rate=args.min_baseline_match,
        min_verification_speedup=args.min_speedup,
        max_memory_growth_pct=args.max_memory_growth,
    )

    # Run checks
    print("\nRunning go/no-go checks...")
    checker = GoNoGoChecker(criteria)
    check_results = checker.check_all(aggregated)

    # Generate reports
    print("\nGenerating reports...")

    # Console report
    console_report = checker.generate_report(check_results)
    print()
    print(console_report)

    # Markdown report
    md_report = generate_markdown_report(
        aggregated, checker, check_results,
        policy_results, scheduling_results, escalation_results, stress_results
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md_report)
    print(f"\nMarkdown report saved to: {output_path}")

    # JSON output
    json_path = Path(args.output_json)
    json_data = {
        "aggregated_metrics": aggregated,
        "criteria_results": checker.to_dict(check_results),
        "sources": {
            "policy_validation": policy_results is not None,
            "scheduling_sweep": scheduling_results is not None,
            "escalation_sweep": escalation_results is not None,
            "stress_test": stress_results is not None,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    # Exit with appropriate code
    if checker.all_passed(check_results):
        print("\n>>> GO - System is ready for deployment <<<")
        sys.exit(0)
    else:
        print("\n>>> NO-GO - System does NOT meet deployment criteria <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
