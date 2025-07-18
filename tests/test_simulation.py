"""
Tests for the SDE simulation module.
"""

import numpy as np
import pytest
from stablesde.core.simulation import simulate_sde, simulate_sde_with_threshold, simulate_multiple_trajectories


def test_simulate_sde_shape():
    """Test that the simulated SDE has the correct shape."""
    N = 100
    t, X = simulate_sde(N, alpha=1.5)
    assert len(t) == N + 1
    assert len(X) == N + 1
    assert t[0] == 0.0
    assert t[-1] == 1.0  # Default T is 1.0


def test_simulate_sde_initial_value():
    """Test that the initial value is set correctly."""
    N = 100
    x0 = 2.0
    _, X = simulate_sde(N, alpha=1.5, x0=x0)
    assert X[0] == x0


def test_simulate_sde_custom_functions():
    """Test that custom drift and diffusion functions work."""
    N = 100
    drift_func = lambda x: -x  # Linear mean-reverting drift
    diffusion_func = lambda x: 1.0  # Constant diffusion
    
    _, X = simulate_sde(N, alpha=1.5, drift_func=drift_func, diffusion_func=diffusion_func)
    # No assertions on the values, just checking that it runs without errors


def test_simulate_sde_with_threshold():
    """Test that the SDE with threshold has the correct shape."""
    N = 100
    t, X = simulate_sde_with_threshold(N, alpha=1.5)
    assert len(t) == N + 1
    assert len(X) == N + 1


def test_simulate_sde_different_seeds():
    """Test that different seeds produce different trajectories."""
    N = 100
    _, X1 = simulate_sde(N, alpha=1.5, seed=42)
    _, X2 = simulate_sde(N, alpha=1.5, seed=43)
    assert not np.array_equal(X1, X2)


def test_simulate_multiple_trajectories():
    """Test that multiple trajectories are correctly simulated."""
    N = 100
    n_trajectories = 5
    t, trajectories = simulate_multiple_trajectories(n_trajectories, N, alpha=1.5)
    assert len(t) == N + 1
    assert len(trajectories) == n_trajectories
    assert all(len(traj) == N + 1 for traj in trajectories)