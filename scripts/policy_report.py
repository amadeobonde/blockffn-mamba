#!/usr/bin/env python3
"""
Policy Report Generator for BlockFFN Adaptive Inference

Generates formatted console reports and analysis from validation results.

Usage:
    python scripts/policy_report.py results.json
"""

import json
import argparse
from typing import Dict, Any


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print a formatted comparison report to console."""

    print("\n")
    print("=" * 80)
    print("POLICY VALIDATION REPORT")
    print("=" * 80)

    # Summary table
    print("\n" + "-" * 80)
    print("PRESET COMPARISON SUMMARY")
    print("-" * 80)

    header = f"{'Preset':<15} {'Tokens':>8} {'Small%':>8} {'Med%':>8} {'Full%':>8} {'Green%':>8} {'Yellow%':>8} {'Red%':>8}"
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        if hasattr(r, 'avg_pct_small_window'):
            # PresetResults object
            print(f"{name:<15} {r.total_tokens:>8} "
                  f"{r.avg_pct_small_window*100:>7.1f}% "
                  f"{r.avg_pct_medium_window*100:>7.1f}% "
                  f"{r.avg_pct_full_window*100:>7.1f}% "
                  f"{r.avg_pct_green*100:>7.1f}% "
                  f"{r.avg_pct_yellow*100:>7.1f}% "
                  f"{r.avg_pct_red*100:>7.1f}%")
        else:
            # Dict from JSON
            print(f"{name:<15} {r['total_tokens']:>8} "
                  f"{r['avg_pct_small_window']*100:>7.1f}% "
                  f"{r['avg_pct_medium_window']*100:>7.1f}% "
                  f"{r['avg_pct_full_window']*100:>7.1f}% "
                  f"{r['avg_pct_green']*100:>7.1f}% "
                  f"{r['avg_pct_yellow']*100:>7.1f}% "
                  f"{r['avg_pct_red']*100:>7.1f}%")

    # Escalation details
    print("\n" + "-" * 80)
    print("ESCALATION ANALYSIS")
    print("-" * 80)

    header2 = f"{'Preset':<15} {'Esc Rate':>10} {'Suffix Esc':>12} {'Avg Streak':>12} {'Max Streak':>12}"
    print(header2)
    print("-" * len(header2))

    for name, r in results.items():
        if hasattr(r, 'avg_escalation_rate'):
            print(f"{name:<15} "
                  f"{r.avg_escalation_rate*100:>9.1f}% "
                  f"{r.total_suffix_escalations:>12} "
                  f"{r.avg_accepted_streak_length:>12.1f} "
                  f"{r.max_accepted_streak_observed:>12}")
        else:
            print(f"{name:<15} "
                  f"{r['avg_escalation_rate']*100:>9.1f}% "
                  f"{r['total_suffix_escalations']:>12} "
                  f"{r['avg_accepted_streak_length']:>12.1f} "
                  f"{r['max_accepted_streak_observed']:>12}")

    # Confidence distribution
    print("\n" + "-" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("-" * 80)

    for name, r in results.items():
        hist = r.confidence_histogram if hasattr(r, 'confidence_histogram') else r['confidence_histogram']
        total = sum(hist.values()) if hist else 1

        print(f"\n{name}:")
        for bucket in ["0.00-0.50", "0.50-0.65", "0.65-0.75", "0.75-0.85", "0.85-0.95", "0.95-1.00"]:
            count = hist.get(bucket, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "*" * min(int(pct / 2), 40)  # Scale bar to max 40 chars
            print(f"  [{bucket}): {count:>6} ({pct:>5.1f}%) {bar}")

    # Performance
    print("\n" + "-" * 80)
    print("PERFORMANCE")
    print("-" * 80)

    header3 = f"{'Preset':<15} {'Avg Confidence':>15} {'Tokens/sec':>12}"
    print(header3)
    print("-" * len(header3))

    for name, r in results.items():
        if hasattr(r, 'avg_mean_confidence'):
            print(f"{name:<15} {r.avg_mean_confidence:>15.3f} {r.avg_tokens_per_second:>12.1f}")
        else:
            print(f"{name:<15} {r['avg_mean_confidence']:>15.3f} {r['avg_tokens_per_second']:>12.1f}")

    # Configuration comparison
    print("\n" + "-" * 80)
    print("CONFIGURATION COMPARISON")
    print("-" * 80)

    config_header = f"{'Preset':<15} {'tau_g':>8} {'tau_y':>8} {'small_w':>8} {'med_w':>8} {'stride':>8} {'cheap_gate':>12}"
    print(config_header)
    print("-" * len(config_header))

    for name, r in results.items():
        cfg = r.config_summary if hasattr(r, 'config_summary') else r['config_summary']
        print(f"{name:<15} "
              f"{cfg.get('tau_green', 'N/A'):>8} "
              f"{cfg.get('tau_yellow', 'N/A'):>8} "
              f"{cfg.get('small_window', 'N/A'):>8} "
              f"{cfg.get('medium_window', 'N/A'):>8} "
              f"{cfg.get('verification_stride', 1):>8} "
              f"{str(cfg.get('enable_cheap_gate', False)):>12}")

    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)

    # Find best preset for different use cases
    presets = list(results.keys())
    if presets:
        def get_val(p, key):
            r = results[p]
            if hasattr(r, key):
                return getattr(r, key)
            return r.get(key, 0)

        # Highest green rate (most stable)
        safest = max(presets, key=lambda p: get_val(p, 'avg_pct_green'))

        # Lowest escalation rate (most efficient)
        efficient = min(presets, key=lambda p: get_val(p, 'avg_escalation_rate'))

        # Highest throughput
        fastest = max(presets, key=lambda p: get_val(p, 'avg_tokens_per_second'))

        # Best balance (high green, low escalation, reasonable speed)
        def balance_score(p):
            green = get_val(p, 'avg_pct_green')
            esc = get_val(p, 'avg_escalation_rate')
            speed = get_val(p, 'avg_tokens_per_second')
            # Normalize and combine
            return green * 0.4 + (1 - esc) * 0.4 + (speed / 100) * 0.2

        best_balance = max(presets, key=balance_score)

        print(f"  For maximum stability (highest green rate): {safest}")
        print(f"  For minimum escalations: {efficient}")
        print(f"  For maximum throughput: {fastest}")
        print(f"  Best overall balance: {best_balance}")

    # Go/No-Go assessment
    print("\n" + "-" * 80)
    print("GO/NO-GO ASSESSMENT")
    print("-" * 80)

    go_criteria = {
        "Escalation rate < 10%": False,
        "Green rate >= 70%": False,
        "Avg streak >= 10": False,
    }

    for name, r in results.items():
        esc_rate = get_val(name, 'avg_escalation_rate')
        green_rate = get_val(name, 'avg_pct_green')
        avg_streak = get_val(name, 'avg_accepted_streak_length')

        if esc_rate < 0.10:
            go_criteria["Escalation rate < 10%"] = True
        if green_rate >= 0.70:
            go_criteria["Green rate >= 70%"] = True
        if avg_streak >= 10:
            go_criteria["Avg streak >= 10"] = True

    for criterion, passed in go_criteria.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[-]"
        print(f"  {symbol} {criterion}: {status}")

    all_pass = all(go_criteria.values())
    print(f"\n  Overall: {'GO for CUDA' if all_pass else 'Consider tuning before CUDA'}")

    print("\n" + "=" * 80)


def generate_json_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate a detailed JSON report."""
    report = {
        "summary": {},
        "presets": {},
    }

    for name, r in results.items():
        if hasattr(r, '__dict__'):
            report["presets"][name] = r.__dict__
        else:
            report["presets"][name] = r

    # Add cross-preset comparisons
    report["summary"]["total_presets"] = len(results)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def load_and_print_report(json_path: str) -> None:
    """Load results from JSON and print report."""
    with open(json_path, "r") as f:
        results = json.load(f)

    print_comparison_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate policy report from JSON results")
    parser.add_argument("input", help="Path to JSON results file")

    args = parser.parse_args()
    load_and_print_report(args.input)
