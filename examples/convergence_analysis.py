"""
Convergence analysis for the StableSDE package.

This script demonstrates how to analyze the convergence and stability
of the simulation methods as the discretization parameter delta varies.
"""

import numpy as np
import matplotlib.pyplot as plt

from stablesde.core.simulation import simulate_multiple_trajectories
from stablesde.core.utils import check_stability, test_convergence
from stablesde.visualization.plots import plot_stability_analysis, plot_convergence_test


def analyze_stability():
    """Analyze the stability of the simulation method for different delta values."""
    # Parameters
    params = {
        'alpha': 1.6,    # stability index
        'beta': 0.0,     # skewness
        'sigma': 1.0,    # scale
        'mu': 0.0,       # location
        'T': 1.0,        # time horizon
        'N': 5000,       # number of steps
        'x0': 1.0,       # initial value
        'theta': 0.7,    # drift parameter
        'lam': 1.0,      # mean-reversion level
        'rho': 1.0,      # diffusion parameter
    }
    
    # Delta values to test
    delta_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_trajectories = 10
    
    # Simulate and analyze for each delta
    stability_data = []
    for delta in delta_values:
        print(f"Analyzing stability for δ = {delta}...")
        _, trajectories = simulate_multiple_trajectories(
            n_trajectories, params['N'], params['alpha'], params['beta'],
            params['sigma'], params['mu'], params['T'], params['x0'],
            theta=params['theta'], lam=params['lam'], rho=params['rho'],
            delta=delta, use_threshold=True, seed=42
        )
        
        # Check stability
        stability = check_stability(trajectories, max_threshold=50.0)
        print(f"  δ = {delta:.1f} | std_final = {stability['std_final']:.4f} "
              f"| max_val = {stability['max_val']:.2f} "
              f"| exploding_paths = {stability['exploding_paths']}")
        
        stability_data.append(stability)
    
    # Plot stability analysis
    plot_stability_analysis(delta_values, stability_data)
    plt.show()


def analyze_convergence():
    """Analyze the convergence of the simulation method as delta varies."""
    # Parameters
    params = {
        'alpha': 1.6,    # stability index
        'beta': 0.0,     # skewness
        'sigma': 1.0,    # scale
        'mu': 0.0,       # location
        'T': 1.0,        # time horizon
        'N': 5000,       # number of steps
        'x0': 1.0,       # initial value
        'theta': 0.7,    # drift parameter
        'lam': 1.0,      # mean-reversion level
        'rho': 1.0,      # diffusion parameter
    }
    
    # Delta values to test
    delta_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Test convergence
    print("Testing convergence...")
    delta_values, mean_values = test_convergence(
        delta_values, params['N'], params['alpha'], params['beta'],
        params['sigma'], params['mu'], params['T'], params['x0'],
        theta=params['theta'], lam=params['lam'], rho=params['rho'],
        n_trajectories=20, use_threshold=True, seed=42
    )
    
    # Print results
    for delta, mean in zip(delta_values, mean_values):
        print(f"  δ = {delta:.1f} | mean_final = {mean:.6f}")
    
    # Plot convergence test
    plot_convergence_test(delta_values, mean_values)
    plt.show()


def analyze_moment_existence():
    """Analyze the existence of moments for different alpha values."""
    # Parameters
    base_params = {
        'beta': 0.0,     # skewness
        'sigma': 1.0,    # scale
        'mu': 0.0,       # location
        'T': 1.0,        # time horizon
        'N': 5000,       # number of steps
        'x0': 1.0,       # initial value
        'theta': 0.7,    # drift parameter
        'lam': 1.0,      # mean-reversion level
        'rho': 1.0,      # diffusion parameter
        'delta': 0.7     # discretization parameter
    }
    
    # Alpha values to test
    alpha_values = [1.2, 1.5, 1.8, 2.0]
    
    # Moment orders to test
    p_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    # Number of trajectories
    n_trajectories = 50
    
    # Initialize results
    results = np.zeros((len(alpha_values), len(p_values)))
    
    # Simulate and compute moments for each alpha
    for i, alpha in enumerate(alpha_values):
        print(f"Analyzing moments for α = {alpha}...")
        _, trajectories = simulate_multiple_trajectories(
            n_trajectories, base_params['N'], alpha, base_params['beta'],
            base_params['sigma'], base_params['mu'], base_params['T'], base_params['x0'],
            theta=base_params['theta'], lam=base_params['lam'], rho=base_params['rho'],
            delta=base_params['delta'], use_threshold=True, seed=42
        )
        
        # Compute moments
        for j, p in enumerate(p_values):
            # Extract final values
            final_values = np.array([traj[-1] for traj in trajectories])
            # Compute moment
            moment_p = np.mean(np.abs(final_values) ** p)
            results[i, j] = moment_p
            print(f"  p = {p:.1f} | moment = {moment_p:.6f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red', 'purple']
    for i, alpha in enumerate(alpha_values):
        ax.plot(p_values, results[i], 'o-', color=colors[i], label=f"α = {alpha}")
    
    ax.set_title("Existence of Moments for Different α Values")
    ax.set_xlabel("Moment Order (p)")
    ax.set_ylabel("E[|X|^p]")
    ax.set_yscale('log')  # Use log scale for better visualization
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_stability()
    analyze_convergence()
    analyze_moment_existence()