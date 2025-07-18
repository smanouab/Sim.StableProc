"""
StableSDE: A package for simulating stochastic differential equations driven by stable processes.

This package provides tools for simulating and analyzing stochastic differential equations
driven by α-stable processes, which are particularly useful for modeling systems with
heavy-tailed noise distributions.
"""

__version__ = "0.1.0"
__author__ = "Author"
__email__ = "author@example.com"

from stablesde.core.processes import simulate_stable_process
from stablesde.core.simulation import simulate_sde, simulate_sde_with_threshold