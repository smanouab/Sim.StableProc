"""
Tests for the utility functions.
"""

import numpy as np
import pytest
from stablesde.core.utils import check_stability, compute_moments, test_convergence


def test_check_stability():
    """Test that the stability check function works correctly."""
    # Create some test trajectories
    trajectories = [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, -1.0, -2.0, -3.0]),
        np.array([0.0, 10.0, 50.0, 100.0]),
        np.array([0.0, -10.0, -50.0, -100.0]),
    ]
    
    # Test with default threshold
    stability = check_stability(trajectories)
    assert stability['std_final'] == pytest.approx(58.02, abs=0.01)
    assert stability['max_val'] == 100.0
    assert stability['exploding_paths'] == 0  # No paths exceed the default threshold
    
    # Test with custom threshold
    stability = check_stability(trajectories, max_threshold=50.0)
    assert stability['exploding_paths'] == 2  # Two paths exceed the threshold of 50


def test_compute_moments():
    """Test that the moment computation function works correctly."""
    # Create some test trajectories
    trajectories = [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, -1.0, -2.0, -3.0]),
    ]
    
    # Test first moment (mean absolute value)
    moment1 = compute_moments(trajectories, p=1.0)
    assert moment1 == 3.0
    
    # Test second moment (mean squared value)
    moment2 = compute_moments(trajectories, p=2.0)
    assert moment2 == 9.0
    
    # Test with specific time index
    moment1_mid = compute_moments(trajectories, p=1.0, time_index=2)
    assert moment1_mid == 2.0


def test_test_convergence():
    """Test that the convergence test function works correctly."""
    # This is more of a smoke test to ensure the function runs without errors
    delta_values = [0.5, 0.6, 0.7]
    N = 10  # Small N for fast testing
    
    # Test with default parameters
    deltas, means = test_convergence(delta_values, N, alpha=1.9, n_trajectories=2)
    assert len(deltas) == len(delta_values)
    assert len(means) == len(delta_values)