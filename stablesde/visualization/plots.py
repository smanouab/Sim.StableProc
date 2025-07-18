"""
Plotting functions for visualizing simulation results.

This module provides functions for creating various plots to visualize
the simulated trajectories and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union


def plot_trajectories(
    trajectories: List[np.ndarray],
    T: float = 1.0,
    alpha: Optional[float] = None,
    delta: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    alpha_opacity: float = 0.8,
    linewidth: float = 0.8,
    show_legend: bool = True,
    ylim: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple trajectories of a simulated stochastic differential equation.
    
    Parameters
    ----------
    trajectories : List[np.ndarray]
        List of simulated trajectories.
    T : float, optional
        Time horizon. Default is 1.0.
    alpha : float, optional
        Stability index of the driving stable process. If provided, included in the title. Default is None.
    delta : float, optional
        Discretization parameter. If provided, included in the title. Default is None.
    title : str, optional
        Custom title for the plot. If None, a default title is generated. Default is None.
    figsize : Tuple[int, int], optional
        Figure size. Default is (10, 6).
    colors : List[str], optional
        List of colors for the trajectories. If None, a colormap is used. Default is None.
    alpha_opacity : float, optional
        Opacity of the trajectories. Default is 0.8.
    linewidth : float, optional
        Line width for the trajectories. Default is 0.8.
    show_legend : bool, optional
        Whether to show the legend. Default is True.
    ylim : Tuple[float, float], optional
        Y-axis limits. If None, automatic limits are used. Default is None.
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
        
    Notes
    -----
    This function creates a figure showing multiple trajectories of a simulated
    stochastic differential equation.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Time points
    n_trajectories = len(trajectories)
    n_points = len(trajectories[0])
    t = np.linspace(0, T, n_points)
    
    # Colors
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    # Plot trajectories
    for i, traj in enumerate(trajectories):
        label = f"Trajectory {i+1}" if i == 0 else None if i >= 5 else f"Trajectory {i+1}"
        ax.plot(t, traj, color=colors[i], lw=linewidth, alpha=alpha_opacity, label=label)
    
    # Title
    if title is None:
        if alpha is not None and delta is not None:
            title = f"SDE Trajectories (α = {alpha:.2f}, δ = {delta:.1f})"
        elif alpha is not None:
            title = f"SDE Trajectories (α = {alpha:.2f})"
        elif delta is not None:
            title = f"SDE Trajectories (δ = {delta:.1f})"
        else:
            title = "SDE Trajectories"
    ax.set_title(title)
    
    # Axes
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    
    # Legend
    if show_legend:
        ax.legend(loc="best")
    
    # Layout
    fig.tight_layout()
    
    return fig, ax


def plot_stability_analysis(
    delta_values: List[float],
    stability_data: List[Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot stability analysis results.
    
    Parameters
    ----------
    delta_values : List[float]
        List of delta values used in the analysis.
    stability_data : List[Dict[str, float]]
        List of dictionaries containing stability metrics for each delta value.
    figsize : Tuple[int, int], optional
        Figure size. Default is (10, 6).
    title : str, optional
        Custom title for the plot. If None, a default title is used. Default is None.
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
        
    Notes
    -----
    This function creates a figure showing the stability metrics as a function of delta.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Extract metrics
    stds = [data['std_final'] for data in stability_data]
    exploding = [data['exploding_paths'] for data in stability_data]
    
    # First axis: standard deviation
    color = 'tab:blue'
    ax1.set_xlabel('Delta')
    ax1.set_ylabel('Standard Deviation', color=color)
    ax1.plot(delta_values, stds, 'o-', color=color, label='Std Dev')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Second axis: exploding paths
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Exploding Paths', color=color)
    ax2.plot(delta_values, exploding, 's--', color=color, label='Exploding Paths')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Title
    if title is None:
        title = "Stability Analysis vs Delta"
    fig.suptitle(title)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Grid
    ax1.grid(True)
    
    # Layout
    fig.tight_layout()
    
    return fig, ax1


def plot_convergence_test(
    delta_values: List[float],
    mean_values: List[float],
    figsize: Tuple[int, int] = (8, 5),
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot convergence test results.
    
    Parameters
    ----------
    delta_values : List[float]
        List of delta values used in the test.
    mean_values : List[float]
        List of mean values for each delta value.
    figsize : Tuple[int, int], optional
        Figure size. Default is (8, 5).
    title : str, optional
        Custom title for the plot. If None, a default title is used. Default is None.
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
        
    Notes
    -----
    This function creates a figure showing the convergence of the mean value
    as the discretization parameter delta varies.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the convergence
    ax.plot(delta_values, mean_values, 'o-', color='darkblue')
    
    # Title
    if title is None:
        title = "Convergence Test: Mean Final Value vs Delta"
    ax.set_title(title)
    
    # Axes
    ax.set_xlabel("Delta")
    ax.set_ylabel("Mean Final Value")
    ax.grid(True)
    
    # Layout
    fig.tight_layout()
    
    return fig, ax


def plot_process_comparison(
    t: np.ndarray,
    brownian: np.ndarray,
    stable: np.ndarray,
    alpha: float,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a comparison between a Brownian motion and an α-stable process.
    
    Parameters
    ----------
    t : np.ndarray
        Time points.
    brownian : np.ndarray
        Simulated Brownian motion trajectory.
    stable : np.ndarray
        Simulated α-stable process trajectory.
    alpha : float
        Stability index of the stable process.
    figsize : Tuple[int, int], optional
        Figure size. Default is (12, 8).
    title : str, optional
        Custom title for the plot. If None, a default title is used. Default is None.
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
        
    Notes
    -----
    This function creates a figure comparing a Brownian motion (α = 2) with
    an α-stable process (α < 2).
    """
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot Brownian motion
    axs[0].plot(t, brownian, color='blue')
    axs[0].set_title("Brownian Motion (α = 2)")
    axs[0].grid(True)
    axs[0].set_ylabel("Value")
    
    # Plot stable process
    axs[1].plot(t, stable, color='red')
    axs[1].set_title(f"α-Stable Process (α = {alpha:.2f})")
    axs[1].grid(True)
    axs[1].set_ylabel("Value")
    axs[1].set_xlabel("Time")
    
    # Main title
    if title is None:
        title = "Comparison: Brownian Motion vs α-Stable Process"
    fig.suptitle(title)
    
    # Layout
    fig.tight_layout()
    
    return fig, axs


def plot_zoom_window(
    t: np.ndarray,
    trajectories: List[np.ndarray],
    t_start: float,
    t_end: float,
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a zoomed window of the trajectories.
    
    Parameters
    ----------
    t : np.ndarray
        Time points.
    trajectories : List[np.ndarray]
        List of simulated trajectories.
    t_start : float
        Start time of the zoom window.
    t_end : float
        End time of the zoom window.
    figsize : Tuple[int, int], optional
        Figure size. Default is (10, 5).
    title : str, optional
        Custom title for the plot. If None, a default title is generated. Default is None.
    colors : List[str], optional
        List of colors for the trajectories. If None, a colormap is used. Default is None.
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
        
    Notes
    -----
    This function creates a figure showing a zoomed window of the trajectories,
    which can be useful for examining the behavior of jumps in the process.
    """
    # Create mask for the zoom window
    mask = (t >= t_start) & (t <= t_end)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    n_trajectories = len(trajectories)
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    # Plot trajectories in the zoom window
    for i, traj in enumerate(trajectories):
        ax.plot(t[mask], traj[mask], color=colors[i], lw=0.9)
    
    # Title
    if title is None:
        title = f"Zoom Window: t ∈ [{t_start}, {t_end}]"
    ax.set_title(title)
    
    # Axes
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.grid(True)
    
    # Layout
    fig.tight_layout()
    
    return fig, ax