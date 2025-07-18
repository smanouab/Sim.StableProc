"""
Simulation of stochastic differential equations driven by stable processes.

This module provides functions for simulating SDEs of the form:
dX_t = a(X_t) dt + b(X_t) dZ_t
where Z_t is an α-stable process.
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple, List

from stablesde.core.processes import simulate_stable_process, get_stable_increments


def simulate_sde(
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
    delta: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a stochastic differential equation driven by an α-stable process.
    
    The SDE is of the form:
    dX_t = a(X_t) dt + b(X_t) dZ_t
    
    where a(X_t) is the drift function and b(X_t) is the diffusion function.
    
    Parameters
    ----------
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
    delta : float, optional
        Discretization parameter affecting time step: dt = T / N^delta. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - An array of shape (N+1,) containing the time points
        - An array of shape (N+1,) containing the simulated process values
        
    Notes
    -----
    If drift_func is None, the default drift function is:
    a(X_t) = theta * (lam - X_t)
    
    If diffusion_func is None, the default diffusion function is:
    b(X_t) = rho * |X_t|^(1/alpha)
    """
    # Set default functions if not provided
    if drift_func is None:
        drift_func = lambda x: theta * (lam - x)
    if diffusion_func is None:
        diffusion_func = lambda x: rho * (np.abs(x)**(1/alpha))
    
    # Generate the driving stable process
    Z = simulate_stable_process(N, alpha, beta, sigma, mu, T, delta, seed)
    
    # Time points
    dt = T / (N ** delta)
    t = np.linspace(0, T, N + 1)
    
    # Initialize the process
    X = np.empty(N + 1)
    X[0] = x0
    
    # Euler-Maruyama discretization
    for k in range(1, N + 1):
        X_prev = X[k - 1]
        dZ = Z[k] - Z[k - 1]
        drift = drift_func(X_prev) * dt
        diffusion = diffusion_func(X_prev) * dZ
        X[k] = X_prev + drift + diffusion
    
    return t, X


def simulate_sde_with_threshold(
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
    delta: float = 1.0,
    epsilon_factor: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a stochastic differential equation driven by an α-stable process,
    with a threshold mechanism for small jumps.
    
    The SDE is of the form:
    dX_t = a(X_t) dt + b(X_t) dZ_t
    
    where a(X_t) is the drift function and b(X_t) is the diffusion function.
    Small jumps (those with magnitude below epsilon) are ignored.
    
    Parameters
    ----------
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
    delta : float, optional
        Discretization parameter affecting time step: dt = T / N^delta. Default is 1.0.
    epsilon_factor : float, optional
        Factor to adjust the threshold epsilon. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - An array of shape (N+1,) containing the time points
        - An array of shape (N+1,) containing the simulated process values
        
    Notes
    -----
    The threshold epsilon is calculated as:
    epsilon = epsilon_factor * dt^(1/alpha)
    
    Jumps with magnitude less than epsilon are ignored, which can help improve
    numerical stability.
    """
    # Set default functions if not provided
    if drift_func is None:
        drift_func = lambda x: theta * (lam - x)
    if diffusion_func is None:
        diffusion_func = lambda x: rho * (np.abs(x)**(1/alpha))
    
    # Generate the driving stable process
    Z = simulate_stable_process(N, alpha, beta, sigma, mu, T, delta, seed)
    
    # Time points
    dt = T / (N ** delta)
    t = np.linspace(0, T, N + 1)
    
    # Calculate the threshold epsilon
    epsilon = epsilon_factor * dt ** (1 / alpha)
    
    # Initialize the process
    X = np.empty(N + 1)
    X[0] = x0
    
    # Euler-Maruyama discretization with threshold
    for k in range(1, N + 1):
        X_prev = X[k - 1]
        dZ = Z[k] - Z[k - 1]
        drift = drift_func(X_prev) * dt
        
        # Apply threshold for small jumps
        if abs(dZ) > epsilon:
            diffusion = diffusion_func(X_prev) * dZ
        else:
            diffusion = 0.0
        
        X[k] = X_prev + drift + diffusion
    
    return t, X


def simulate_multiple_trajectories(
    n_trajectories: int,
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
    delta: float = 1.0,
    use_threshold: bool = True,
    epsilon_factor: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Simulate multiple trajectories of a stochastic differential equation driven by an α-stable process.
    
    Parameters
    ----------
    n_trajectories : int
        Number of trajectories to simulate.
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
    delta : float, optional
        Discretization parameter affecting time step: dt = T / N^delta. Default is 1.0.
    use_threshold : bool, optional
        Whether to use the threshold mechanism for small jumps. Default is True.
    epsilon_factor : float, optional
        Factor to adjust the threshold epsilon. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        A tuple containing:
        - An array of shape (N+1,) containing the time points
        - A list of n_trajectories arrays, each of shape (N+1,), containing the simulated process values
        
    Notes
    -----
    If seed is provided, each trajectory will have a different seed to ensure independence.
    """
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.randint(0, 2**32, size=n_trajectories)
    else:
        seeds = [None] * n_trajectories
    
    # Set default functions if not provided
    if drift_func is None:
        drift_func = lambda x: theta * (lam - x)
    if diffusion_func is None:
        diffusion_func = lambda x: rho * (np.abs(x)**(1/alpha))
    
    # Time points
    dt = T / (N ** delta)
    t = np.linspace(0, T, N + 1)
    
    # Simulate trajectories
    trajectories = []
    for i in range(n_trajectories):
        if use_threshold:
            _, X = simulate_sde_with_threshold(
                N, alpha, beta, sigma, mu, T, x0, drift_func, diffusion_func,
                theta, lam, rho, delta, epsilon_factor, seeds[i]
            )
        else:
            _, X = simulate_sde(
                N, alpha, beta, sigma, mu, T, x0, drift_func, diffusion_func,
                theta, lam, rho, delta, seeds[i]
            )
        trajectories.append(X)
    
    return t, trajectories