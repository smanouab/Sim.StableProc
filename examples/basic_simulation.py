"""
Example de simulation simple utilisant uniquement la bibliothèque standard Python.
"""

import random
import math
from typing import List, Tuple

def simulate_random_walk(steps: int, drift: float = 0.0) -> List[float]:
    """Simule une marche aléatoire simple."""
    path = [1.0]  # Commence à 1.0
    for _ in range(steps):
        # Utilise la distribution uniforme comme approximation
        step = random.uniform(-1, 1) + drift
        path.append(path[-1] + step)
    return path

def plot_text(values: List[float], width: int = 60, height: int = 20) -> None:
    """Crée une visualisation ASCII simple d'une série temporelle."""
    if not values:
        return
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
    
    print("\nSimulation Results:")
    print("-" * width)
    
    for i in range(height-1, -1, -1):
        line = ""
        for val in values:
            normalized = (val - min_val) / range_val
            pos = int(normalized * (height-1))
            line += "*" if pos == i else " "
        print(f"{line}")
    
    print("-" * width)
    print(f"Min: {min_val:.2f}, Max: {max_val:.2f}")

def run_basic_simulation():
    """Exécute une simulation de base."""
    # Paramètres
    steps = 100
    n_paths = 3
    
    print("Simulation d'une marche aléatoire simple...")
    
    # Simule plusieurs chemins
    paths = []
    for i in range(n_paths):
        path = simulate_random_walk(steps, drift=0.01)
        paths.append(path)
        print(f"\nChemin {i+1}:")
        plot_text(path)

if __name__ == "__main__":
    run_basic_simulation()