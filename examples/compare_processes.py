"""
Comparison of Brownian motion and α-stable processes.

This script demonstrates the differences between Brownian motion (a Gaussian process)
and α-stable processes with α < 2, which exhibit heavy tails and jumps.
"""

import numpy as np
import matplotlib.pyplot as plt

from stablesde.core.processes import simulate_stable_process
from stablesde.visualization.plots import plot_process_comparison


def compare_brownian_and_stable():
    """Compare Brownian motion and α-stable processes."""
    # Parameters
    N = 1000
    T = 1.0
    
    # Generate Brownian motion (α = 2)
    print("Generating Brownian motion...")
    brownian = simulate_stable_process(N, alpha=2.0, beta=0.0, T=T, seed=42)
    
    # Generate stable processes with different α values
    alpha_values = [1.9, 1.5, 1.0]
    stable_processes = []
    
    for alpha in alpha_values:
        print(f"Generating stable process with α = {alpha}...")
        stable = simulate_stable_process(N, alpha=alpha, beta=0.0, T=T, seed=42)
        stable_processes.append(stable)
    
    # Plot comparisons
    t = np.linspace(0, T, N + 1)
    
    for i, alpha in enumerate(alpha_values):
        fig, axs = plot_process_comparison(
            t, brownian, stable_processes[i], alpha,
            title=f"Comparison: Brownian Motion vs α-Stable Process (α = {alpha})"
        )
        plt.show()


def show_stable_process_jumps():
    """Show the jumps in α-stable processes with different α values."""
    # Parameters
    N = 10000
    T = 1.0
    
    # Alpha values to compare
    alpha_values = [1.9, 1.7, 1.5, 1.3, 1.1]
    
    # Generate stable processes
    stable_processes = []
    for alpha in alpha_values:
        print(f"Generating stable process with α = {alpha}...")
        stable = simulate_stable_process(N, alpha=alpha, beta=0.0, T=T, seed=42)
        stable_processes.append(stable)
    
    # Plot processes
    t = np.linspace(0, T, N + 1)
    fig, axs = plt.subplots(len(alpha_values), 1, figsize=(10, 12), sharex=True)
    
    for i, (alpha, process) in enumerate(zip(alpha_values, stable_processes)):
        axs[i].plot(t, process, color=plt.cm.viridis(i / len(alpha_values)))
        axs[i].set_title(f"α-Stable Process (α = {alpha})")
        axs[i].grid(True)
        axs[i].set_ylabel("Value")
    
    axs[-1].set_xlabel("Time")
    fig.suptitle("α-Stable Processes with Different α Values")
    plt.tight_layout()
    plt.show()
    
    # Plot a zoom window to see the jumps more clearly
    t_start = 0.4
    t_end = 0.5
    mask = (t >= t_start) & (t <= t_end)
    
    fig, axs = plt.subplots(len(alpha_values), 1, figsize=(10, 12), sharex=True)
    
    for i, (alpha, process) in enumerate(zip(alpha_values, stable_processes)):
        axs[i].plot(t[mask], process[mask], color=plt.cm.viridis(i / len(alpha_values)))
        axs[i].set_title(f"α-Stable Process (α = {alpha})")
        axs[i].grid(True)
        axs[i].set_ylabel("Value")
    
    axs[-1].set_xlabel("Time")
    fig.suptitle(f"Zoom: t ∈ [{t_start}, {t_end}]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_brownian_and_stable()
    show_stable_process_jumps()