#!/usr/bin/env python3
"""
Unit tests that don't require torch.

These test the pure Python logic components.
Can run in any Python environment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_validation():
    """Test configuration validation and presets."""
    from src.adaptive.config import AdaptiveInferenceConfig, PRESETS

    # Test default config
    config = AdaptiveInferenceConfig()
    assert 0.0 <= config.high_sparsity_threshold <= 1.0
    assert config.small_window > 0
    assert config.tau_green >= config.tau_yellow >= config.tau_red

    # Test presets exist and are valid
    for name, preset in PRESETS.items():
        assert isinstance(preset, AdaptiveInferenceConfig), f"Preset {name} invalid"

    # Test from_preset
    balanced = AdaptiveInferenceConfig.from_preset("balanced")
    assert balanced is not None

    # Test for_device
    cpu_config = config.for_device("cpu")
    assert cpu_config.device == "cpu"

    cuda_config = config.for_device("cuda")
    assert cuda_config.device == "cuda"

    # Test to_dict / from_dict
    d = config.to_dict()
    restored = AdaptiveInferenceConfig.from_dict(d)
    assert restored.high_sparsity_threshold == config.high_sparsity_threshold

    # Test target_layer_indices
    indices = config.get_target_layer_indices(30)
    assert len(indices) > 0
    assert all(0 <= i < 30 for i in indices)

    print("[PASS] Config validation")
    return True


def test_escalation_decision_logic():
    """Test escalation decision maker logic."""
    # Can't import EscalationDecisionMaker directly as it imports torch
    # So we test the config-level logic

    from src.adaptive.config import AdaptiveInferenceConfig

    config = AdaptiveInferenceConfig()

    # Test threshold ordering
    assert config.tau_green > config.tau_yellow > config.tau_red

    # Test streak limit
    assert config.yellow_streak_limit > 0

    # Test cascade threshold
    assert config.cascade_threshold > 0

    print("[PASS] Escalation decision logic")
    return True


def test_config_presets_consistency():
    """Test that all presets are internally consistent."""
    from src.adaptive.config import AdaptiveInferenceConfig, PRESETS

    for name, preset in PRESETS.items():
        # Validate constraints
        try:
            # This runs __post_init__ validation
            _ = AdaptiveInferenceConfig(
                high_sparsity_threshold=preset.high_sparsity_threshold,
                medium_sparsity_threshold=preset.medium_sparsity_threshold,
                small_window=preset.small_window,
                medium_window=preset.medium_window,
                tau_green=preset.tau_green,
                tau_yellow=preset.tau_yellow,
                tau_red=preset.tau_red,
            )
        except AssertionError as e:
            print(f"[FAIL] Preset {name} failed validation: {e}")
            return False

    print("[PASS] Config presets consistency")
    return True


def test_window_size_constraints():
    """Test window size constraint logic."""
    from src.adaptive.config import AdaptiveInferenceConfig

    config = AdaptiveInferenceConfig()

    # Small window < medium window
    assert config.small_window <= config.medium_window

    # Windows are positive
    assert config.small_window > 0
    assert config.medium_window > 0

    # Test constraint violation raises
    try:
        bad_config = AdaptiveInferenceConfig(
            small_window=100,
            medium_window=50,  # Smaller than small_window
        )
        print("[FAIL] Should have raised assertion for bad window sizes")
        return False
    except AssertionError:
        pass  # Expected

    print("[PASS] Window size constraints")
    return True


def test_threshold_constraints():
    """Test confidence threshold constraint logic."""
    from src.adaptive.config import AdaptiveInferenceConfig

    # Test valid thresholds
    config = AdaptiveInferenceConfig(
        tau_green=0.85,
        tau_yellow=0.65,
        tau_red=0.45,
    )
    assert config.tau_green == 0.85

    # Test invalid: green < yellow
    try:
        bad_config = AdaptiveInferenceConfig(
            tau_green=0.50,
            tau_yellow=0.70,
        )
        print("[FAIL] Should have raised for green < yellow")
        return False
    except AssertionError:
        pass  # Expected

    # Test invalid: yellow < red
    try:
        bad_config = AdaptiveInferenceConfig(
            tau_green=0.90,
            tau_yellow=0.40,
            tau_red=0.50,
        )
        print("[FAIL] Should have raised for yellow < red")
        return False
    except AssertionError:
        pass  # Expected

    print("[PASS] Threshold constraints")
    return True


def run_all():
    """Run all unit tests."""
    print("=" * 60)
    print("UNIT TESTS (No Torch Required)")
    print("=" * 60)

    tests = [
        test_config_validation,
        test_escalation_decision_logic,
        test_config_presets_consistency,
        test_window_size_constraints,
        test_threshold_constraints,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
