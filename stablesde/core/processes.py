"""
Simulation of stable processes.

This module provides functions for simulating α-stable processes, which are
used as driving noise in stochastic differential equations.
"""

import numpy as np
from scipy.stats import levy_stable
from typing import Tuple, Optional


def simulate_stable_process(
    N: int,
    alpha: float,
    beta: float = 0.0,
    sigma: float = 1.0,
    mu: float = 0.0,
    T: float = 1.0,
    delta: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate a stable process Z_t on [0, T] with N increments.
    
    Parameters
    ----------
    N : int
        Number of increments.
    alpha : float
        Stability index, must be in (0, 2]. When alpha=2, the process is Gaussian.
    beta : float, optional
        Skewness parameter, must be in [-1, 1]. Default is 0.0 (symmetric).
    sigma : float, optional
        Scale parameter, must be positive. Default is 1.0.
    mu : float, optional
        Location parameter. Default is 0.0.
    T : float, optional
        Time horizon. Default is 1.0.
    delta : float, optional
        Discretization parameter affecting time step: dt = T / N^delta. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    np.ndarray
        Array of shape (N+1,) containing the simulated stable process values
        at times t = 0, T/N^delta, 2T/N^delta, ..., T.
        
    Notes
    -----
    The increments of the process are sampled from an α-stable distribution
    with parameters (alpha, beta, mu*dt, sigma*dt^(1/alpha)).
    """
    if seed is not None:
        np.random.seed(seed)
    
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1, 1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
        
    dt = T / (N ** delta)
    increments = levy_stable.rvs(
        alpha, beta,
        loc=mu * dt,
        scale=sigma * dt**(1/alpha),
        size=N
    )
    
    # Return the process with initial value 0
    return np.concatenate([[0.0], np.cumsum(increments)])


def get_stable_increments(
    N: int,
    alpha: float,
    beta: float = 0.0,
    sigma: float = 1.0,
    mu: float = 0.0,
    T: float = 1.0,
    delta: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Generate increments of a stable process and the corresponding time step.
    
    Parameters
    ----------
    N : int
        Number of increments.
    alpha : float
        Stability index, must be in (0, 2]. When alpha=2, the process is Gaussian.
    beta : float, optional
        Skewness parameter, must be in [-1, 1]. Default is 0.0 (symmetric).
    sigma : float, optional
        Scale parameter, must be positive. Default is 1.0.
    mu : float, optional
        Location parameter. Default is 0.0.
    T : float, optional
        Time horizon. Default is 1.0.
    delta : float, optional
        Discretization parameter affecting time step: dt = T / N^delta. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing:
        - An array of shape (N,) containing the increments of the stable process
        - The time step dt
        
    Notes
    -----
    This function is useful when you only need the increments of the process
    rather than the cumulative process values.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1, 1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
        
    dt = T / (N ** delta)
    increments = levy_stable.rvs(
        alpha, beta,
        loc=mu * dt,
        scale=sigma * dt**(1/alpha),
        size=N
    )
    
    return increments, dt