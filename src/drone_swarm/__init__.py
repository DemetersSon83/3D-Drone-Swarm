"""Drone swarm boids simulation package.

The package intentionally keeps Mesa-dependent imports out of ``__init__`` so
MDP schemas, physics helpers, and IO utilities remain importable in lightweight
analysis environments.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
