"""
Utility functions for stability analysis and convergence testing.

This module provides functions for analyzing the stability and convergence
of simulated stochastic differential equations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union


def check_stability(
    trajectories: List[np.ndarray],
    max_threshold: float = 100.0
) -> Dict[str, float]:
    """
    Check the stability of simulated trajectories.
    
    Parameters
    ----------
    trajectories : List[np.ndarray]
        List of simulated trajectories.
    max_threshold : float, optional
        Threshold above which a trajectory is considered "exploding". Default is 100.0.
        
    Returns
    -------
    Dict[str, float]
        A dictionary containing stability metrics:
        - 'std_final': Standard deviation of final values
        - 'max_val': Maximum absolute value across all trajectories
        - 'exploding_paths': Number of trajectories exceeding the max_threshold
        - 'mean_final': Mean of final values
        
    Notes
    -----
    This function is useful for assessing the numerical stability of a simulation
    method under different parameter settings.
    """
    final_values = np.array([traj[-1] for traj in trajectories])
    max_values = np.array([np.max(np.abs(traj)) for traj in trajectories])
    
    return {
        'std_final': np.std(final_values),
        'mean_final': np.mean(final_values),
        'max_val': np.max(max_values),
        'exploding_paths': np.sum(max_values > max_threshold),
    }


def compute_moments(
    trajectories: List[np.ndarray],
    p: float = 1.0,
    time_index: Optional[int] = None
) -> float:
    """
    Compute the p-th absolute moment of the simulated trajectories.
    
    Parameters
    ----------
    trajectories : List[np.ndarray]
        List of simulated trajectories.
    p : float, optional
        Order of the moment to compute. Default is 1.0 (mean absolute value).
    time_index : int, optional
        Index of the time point to use. If None, the final time point is used. Default is None.
        
    Returns
    -------
    float
        The p-th absolute moment.
        
    Notes
    -----
    For stable processes, moments of order p >= alpha do not exist.
    This function can be used to verify this property empirically.
    """
    if time_index is None:
        time_index = -1  # Use the final time point
    
    values = np.array([traj[time_index] for traj in trajectories])
    return np.mean(np.abs(values) ** p)


def test_convergence(
    delta_values: List[float],
    N: int,
    alpha: float,
    beta: float = 0.0,
    sigma: float = 1.0,
    mu: float = 0.0,
    T: float = 1.0,
    x0: float = 0.0,
    drift_func: Optional[Callable[[float], float]] = None,
    diffusion_func: Optional[Callable[[float], float]] = None,
    theta: float = 0.0,
    lam: float = 0.0,
    rho: float = 1.0,
    n_trajectories: int = 100,
    use_threshold: bool = True,
    epsilon_factor: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Test the convergence of the simulation method as the discretization parameter delta varies.
    
    Parameters
    ----------
    delta_values : List[float]
        List of delta values to test.
    N : int
        Number of time steps.
    alpha : float
        Stability index of the driving stable process, must be in (0, 2].
    beta : float, optional
        Skewness parameter of the driving stable process, must be in [-1, 1]. Default is 0.0.
    sigma : float, optional
        Scale parameter of the driving stable process, must be positive. Default is 1.0.
    mu : float, optional
        Location parameter of the driving stable process. Default is 0.0.
    T : float, optional
        Time horizon. Default is 1.0.
    x0 : float, optional
        Initial value of the process. Default is 0.0.
    drift_func : Callable[[float], float], optional
        Drift function a(X_t). If None, a default drift function is used. Default is None.
    diffusion_func : Callable[[float], float], optional
        Diffusion function b(X_t). If None, a default diffusion function is used. Default is None.
    theta : float, optional
        Parameter for the default drift function. Default is 0.0.
    lam : float, optional
        Parameter for the default drift function. Default is 0.0.
    rho : float, optional
        Parameter for the default diffusion function. Default is 1.0.
    n_trajectories : int, optional
        Number of trajectories to simulate for each delta value. Default is 100.
    use_threshold : bool, optional
        Whether to use the threshold mechanism for small jumps. Default is True.
    epsilon_factor : float, optional
        Factor to adjust the threshold epsilon. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[List[float], List[float]]
        A tuple containing:
        - List of delta values
        - List of corresponding mean final values
        
    Notes
    -----
    This function is useful for assessing the convergence of the simulation method
    as the discretization parameter delta varies. The mean final value should
    converge as delta increases.
    """
    from stablesde.core.simulation import simulate_multiple_trajectories
    
    if seed is not None:
        np.random.seed(seed)
    
    mean_final_values = []
    
    for delta in delta_values:
        _, trajectories = simulate_multiple_trajectories(
            n_trajectories, N, alpha, beta, sigma, mu, T, x0,
            drift_func, diffusion_func, theta, lam, rho, delta,
            use_threshold, epsilon_factor
        )
        
        # Compute the mean final value
        final_values = np.array([traj[-1] for traj in trajectories])
        mean_final = np.mean(final_values)
        mean_final_values.append(mean_final)
    
    return delta_values, mean_final_values