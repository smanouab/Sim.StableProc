# 🌋 Python-Package-for-Stable-Driven-SDE-Simulation
 
This forthcoming Python package Sim.StableProc  provides a simuation of weak solution for stable driven stochastiques differential equations. Under suitable conditions we ensure stability of the Euler-Maruyama scheme.

For more details, see Solym M. Manou-Abi. Weak approximation for stable-driven stochastic differential equations. Preprint 2025

📋 Description


🚀 Installation Locale (Développeurs)
Prérequis

    Python 3.8 ou plus récent
    pip (gestionnaire de packages Python)

Installation

# Cloner le repository
git clone https://github.com/YOUR-USERNAME/Sim.StableProc---Package-for-Stable-Driven-SDE-Simulation .git
cd Sim.StableProc---Package-for-Stable-Driven-SDE-Simulation 
# Installer les dépendances
pip install -r requirements.txt

📦 Module Overview
🌀 processes.py — Simulation of α-Stable Processes

This module provides core functionality for simulating α-stable Lévy processes, which are frequently used as driving noise in stochastic differential equations (SDEs).
It includes:

    Generation of strictly stable or symmetric α-stable increments

    Construction of sample paths of Lévy processes

    Support for varying stability index α, skewness β, and scaling

🔁 simulation.py — Simulation of SDEs Driven by Stable Processes

This module implements numerical schemes to simulate stochastic differential equations of the form:
dXt=a(Xt) dt+b(Xt) dZt
dXt​=a(Xt​)dt+b(Xt​)dZt​

where ZtZt​ is a simulated α-stable process.
It allows flexible specification of:

    Drift functions a(Xt)a(Xt​)

    Diffusion functions b(Xt)b(Xt​)

    Time discretization and sample path resolution

📊 plots.py — Visualization Utilities

This module contains plotting utilities to visualize simulation results:

    Time series plots of process trajectories

    Comparative plots across parameter settings (e.g. different α or β)

    Stylized plots suitable for academic or professional presentation
