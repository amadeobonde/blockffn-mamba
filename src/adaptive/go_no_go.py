"""
Go/No-Go Criteria for Production Deployment

Centralized go/no-go configuration and checking utilities for
validating the BlockFFN adaptive inference system is ready for deployment.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Callable
from enum import Enum


class CriterionStatus(Enum):
    """Status of a go/no-go criterion."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class GoNoGoResult:
    """Result of a single go/no-go criterion check."""
    criterion: str
    status: CriterionStatus
    actual_value: Any
    threshold: Any
    operator: str  # "<=", ">=", "==", "<", ">"
    details: str

    @property
    def passed(self) -> bool:
        return self.status == CriterionStatus.PASSED


@dataclass
class GoNoGoCriteria:
    """
    Criteria that must be met for production deployment.

    All thresholds are based on the checklist from the project requirements.
    """

    # Correctness
    baseline_match_rate: float = 0.99  # 99% of greedy tokens match baseline

    # Efficiency
    max_escalation_rate: float = 0.10  # <10% escalations
    min_verification_speedup: float = 2.0  # 2x faster with scheduling vs no scheduling

    # Stability
    max_memory_growth_pct: float = 0.05  # <5% memory growth over long runs
    determinism_required: bool = True  # Outputs must be deterministic

    # Performance
    min_reduced_compute_rate: float = 0.70  # 70% tokens use reduced compute

    # Quality
    min_green_rate: float = 0.70  # 70% of tokens in green zone
    max_red_rate: float = 0.10  # <10% of tokens in red zone

    # Streaks
    min_avg_streak_length: float = 10.0  # Average accepted streak >= 10 tokens


class GoNoGoChecker:
    """
    Verifies system meets production criteria.

    Usage:
        checker = GoNoGoChecker()
        results = checker.check_all(experiment_results)
        report = checker.generate_report(results)

        if checker.all_passed(results):
            print("GO - System is ready for deployment")
        else:
            print("NO-GO - See report for failures")
    """

    def __init__(self, criteria: Optional[GoNoGoCriteria] = None):
        """
        Initialize checker with criteria.

        Args:
            criteria: Custom criteria, or use defaults if None
        """
        self.criteria = criteria or GoNoGoCriteria()

        # Define criterion checks as (name, operator, threshold_attr, result_key)
        self._checks: List[Tuple[str, str, str, str]] = [
            ("escalation_rate", "<=", "max_escalation_rate", "escalation_rate"),
            ("baseline_match_rate", ">=", "baseline_match_rate", "baseline_match_rate"),
            ("verification_speedup", ">=", "min_verification_speedup", "verification_speedup"),
            ("memory_growth_pct", "<=", "max_memory_growth_pct", "memory_growth_pct"),
            ("determinism", "==", "determinism_required", "is_deterministic"),
            ("reduced_compute_rate", ">=", "min_reduced_compute_rate", "reduced_compute_rate"),
            ("green_rate", ">=", "min_green_rate", "green_rate"),
            ("red_rate", "<=", "max_red_rate", "red_rate"),
            ("avg_streak_length", ">=", "min_avg_streak_length", "avg_streak_length"),
        ]

    def check_criterion(
        self,
        name: str,
        operator: str,
        threshold: Any,
        actual: Any,
    ) -> GoNoGoResult:
        """
        Check a single criterion.

        Args:
            name: Criterion name
            operator: Comparison operator ("<=", ">=", "==", "<", ">")
            threshold: Threshold value
            actual: Actual value from results

        Returns:
            GoNoGoResult with pass/fail status
        """
        if actual is None:
            return GoNoGoResult(
                criterion=name,
                status=CriterionStatus.SKIPPED,
                actual_value=actual,
                threshold=threshold,
                operator=operator,
                details=f"No data available for {name}",
            )

        # Perform comparison
        if operator == "<=":
            passed = actual <= threshold
        elif operator == ">=":
            passed = actual >= threshold
        elif operator == "==":
            passed = actual == threshold
        elif operator == "<":
            passed = actual < threshold
        elif operator == ">":
            passed = actual > threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")

        status = CriterionStatus.PASSED if passed else CriterionStatus.FAILED

        # Format actual value for display
        if isinstance(actual, float):
            actual_str = f"{actual:.4f}"
        elif isinstance(actual, bool):
            actual_str = "Yes" if actual else "No"
        else:
            actual_str = str(actual)

        if isinstance(threshold, float):
            threshold_str = f"{threshold:.4f}"
        elif isinstance(threshold, bool):
            threshold_str = "Yes" if threshold else "No"
        else:
            threshold_str = str(threshold)

        details = f"{actual_str} {operator} {threshold_str} -> {'PASS' if passed else 'FAIL'}"

        return GoNoGoResult(
            criterion=name,
            status=status,
            actual_value=actual,
            threshold=threshold,
            operator=operator,
            details=details,
        )

    def check_all(self, results: Dict[str, Any]) -> List[GoNoGoResult]:
        """
        Check all go/no-go criteria.

        Args:
            results: Dictionary containing experiment results with keys:
                - escalation_rate: float
                - baseline_match_rate: float
                - verification_speedup: float
                - memory_growth_pct: float
                - is_deterministic: bool
                - reduced_compute_rate: float
                - green_rate: float
                - red_rate: float
                - avg_streak_length: float

        Returns:
            List of GoNoGoResult for each criterion
        """
        check_results = []

        for name, operator, threshold_attr, result_key in self._checks:
            threshold = getattr(self.criteria, threshold_attr)
            actual = results.get(result_key)

            result = self.check_criterion(name, operator, threshold, actual)
            check_results.append(result)

        return check_results

    def all_passed(self, results: List[GoNoGoResult]) -> bool:
        """Check if all criteria passed."""
        return all(r.passed or r.status == CriterionStatus.SKIPPED for r in results)

    def count_failures(self, results: List[GoNoGoResult]) -> int:
        """Count number of failed criteria."""
        return sum(1 for r in results if r.status == CriterionStatus.FAILED)

    def get_failures(self, results: List[GoNoGoResult]) -> List[GoNoGoResult]:
        """Get list of failed criteria."""
        return [r for r in results if r.status == CriterionStatus.FAILED]

    def generate_report(self, results: List[GoNoGoResult]) -> str:
        """
        Generate a human-readable report.

        Args:
            results: List of GoNoGoResult from check_all

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("GO/NO-GO ASSESSMENT REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        passed = sum(1 for r in results if r.passed)
        failed = self.count_failures(results)
        skipped = sum(1 for r in results if r.status == CriterionStatus.SKIPPED)
        total = len(results)

        lines.append(f"Summary: {passed}/{total} criteria passed, {failed} failed, {skipped} skipped")
        lines.append("")

        # Overall verdict
        if self.all_passed(results):
            lines.append(">>> VERDICT: GO - System is ready for deployment <<<")
        else:
            lines.append(">>> VERDICT: NO-GO - System does NOT meet deployment criteria <<<")
        lines.append("")

        # Detailed results
        lines.append("-" * 70)
        lines.append("DETAILED RESULTS:")
        lines.append("-" * 70)
        lines.append("")

        for result in results:
            status_icon = {
                CriterionStatus.PASSED: "[PASS]",
                CriterionStatus.FAILED: "[FAIL]",
                CriterionStatus.SKIPPED: "[SKIP]",
                CriterionStatus.WARNING: "[WARN]",
            }[result.status]

            lines.append(f"{status_icon} {result.criterion}")
            lines.append(f"       {result.details}")
            lines.append("")

        # Failures section
        failures = self.get_failures(results)
        if failures:
            lines.append("-" * 70)
            lines.append("FAILURES REQUIRING ATTENTION:")
            lines.append("-" * 70)
            for f in failures:
                lines.append(f"  - {f.criterion}: {f.details}")
            lines.append("")

            # Recommendations
            lines.append("RECOMMENDATIONS:")
            for f in failures:
                rec = self._get_recommendation(f.criterion)
                lines.append(f"  - {f.criterion}: {rec}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _get_recommendation(self, criterion: str) -> str:
        """Get recommendation for a failed criterion."""
        recommendations = {
            "escalation_rate": "Adjust tau_green/tau_yellow thresholds or increase yellow_streak_limit",
            "baseline_match_rate": "Review escalation strategies, consider more aggressive suffix escalation",
            "verification_speedup": "Increase verification_stride or enable cheap_gate",
            "memory_growth_pct": "Check for memory leaks, ensure KV cache is properly managed",
            "determinism": "Disable sampling, use greedy decoding, check random seeds",
            "reduced_compute_rate": "Lower sparsity thresholds, reduce tau_green threshold",
            "green_rate": "Lower tau_green threshold or reduce window sizes",
            "red_rate": "Raise tau_yellow threshold or increase yellow_streak_limit",
            "avg_streak_length": "Lower tau_yellow threshold or use more conservative thresholds",
        }
        return recommendations.get(criterion, "Review system configuration")

    def to_dict(self, results: List[GoNoGoResult]) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "all_passed": self.all_passed(results),
            "passed_count": sum(1 for r in results if r.passed),
            "failed_count": self.count_failures(results),
            "criteria": [
                {
                    "name": r.criterion,
                    "status": r.status.value,
                    "actual": r.actual_value,
                    "threshold": r.threshold,
                    "operator": r.operator,
                    "passed": r.passed,
                }
                for r in results
            ],
        }


def check_go_no_go(results: Dict[str, Any], criteria: Optional[GoNoGoCriteria] = None) -> Tuple[bool, str]:
    """
    Convenience function to check go/no-go criteria.

    Args:
        results: Experiment results dictionary
        criteria: Custom criteria (optional)

    Returns:
        (all_passed, report_string)
    """
    checker = GoNoGoChecker(criteria)
    check_results = checker.check_all(results)
    report = checker.generate_report(check_results)
    return checker.all_passed(check_results), report
