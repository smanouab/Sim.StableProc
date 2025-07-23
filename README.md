# 🌋 Python-Package-for-Stable-Driven-SDE-Simulation
 
This Python package Sim.StableProc  provides a simuation of weak solution for stable driven stochastiques differential equations. Under suitable conditions we ensure stability of the Euler-Maruyama scheme.

 🌋 For more details, see :
 
 ## Solym M. Manou-Abi (2025) . Weak approximation for stable-driven stochastic differential equations.

# 📋 Description


🚀 Local Installation (Developers) and Prerequisites

    Python 3.8 ou plus récent
    pip (gestionnaire de packages Python)

Installation

# Clone the repository
git clone https://github.com/YOUR-USERNAME/Sim.StableProc.git
cd Sim.StableProc
# 📦 Install the dependencies
pip install -r requirements.txt

# 📦 Module Overview

🚀 🌀 processes.py — Simulation of α-Stable Processes

This module provides core functionality for simulating α-stable Lévy processes, which are frequently used as driving noise in stochastic differential equations (SDEs).
It includes:

    Generation of strictly stable or symmetric α-stable increments

    Construction of sample paths of Lévy processes

    Support for varying stability index α, skewness β, and scaling

🚀 🔁 simulation.py — Simulation of SDEs Driven by Stable Processes

This module implements numerical schemes to simulate stable driven stochastic differential equations of the form:
It allows flexible specification of Drift functions and  Diffusion functions.

    Time discretization and sample path resolution

🚀 📊 plots.py — Visualization Utilities

This module contains plotting utilities to visualize simulation results:

   🌀 Time series plots of process trajectories

    🌀 Comparative plots across parameter settings (e.g. different α or β)

   🌀  Stylized plots suitable for academic or professional presentation
