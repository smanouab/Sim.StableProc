"""
Tests for the stable process simulation module.
"""

import numpy as np
import pytest
from stablesde.core.processes import simulate_stable_process, get_stable_increments


def test_simulate_stable_process_shape():
    """Test that the simulated process has the correct shape."""
    N = 100
    process = simulate_stable_process(N, alpha=1.5)
    assert len(process) == N + 1
    assert process[0] == 0.0


def test_simulate_stable_process_seed():
    """Test that using the same seed produces the same process."""
    N = 100
    process1 = simulate_stable_process(N, alpha=1.5, seed=42)
    process2 = simulate_stable_process(N, alpha=1.5, seed=42)
    np.testing.assert_array_equal(process1, process2)


def test_simulate_stable_process_invalid_alpha():
    """Test that an invalid alpha value raises an error."""
    with pytest.raises(ValueError):
        simulate_stable_process(100, alpha=0.0)  # alpha <= 0
    with pytest.raises(ValueError):
        simulate_stable_process(100, alpha=2.5)  # alpha > 2


def test_simulate_stable_process_invalid_beta():
    """Test that an invalid beta value raises an error."""
    with pytest.raises(ValueError):
        simulate_stable_process(100, alpha=1.5, beta=-1.5)  # beta < -1
    with pytest.raises(ValueError):
        simulate_stable_process(100, alpha=1.5, beta=1.5)   # beta > 1


def test_simulate_stable_process_brownian_case():
    """Test that alpha=2 corresponds to Brownian motion."""
    # For alpha=2, the distribution is Gaussian
    N = 1000
    T = 1.0
    process = simulate_stable_process(N, alpha=2.0, sigma=1.0, T=T)
    
    # The standard deviation of the process at time T should be approximately sqrt(T)
    assert abs(np.std(process[-1:]) - np.sqrt(T)) < 0.2


def test_get_stable_increments():
    """Test that the increments function returns the correct output."""
    N = 100
    increments, dt = get_stable_increments(N, alpha=1.5, T=1.0, delta=1.0)
    assert len(increments) == N
    assert dt == 1.0 / N