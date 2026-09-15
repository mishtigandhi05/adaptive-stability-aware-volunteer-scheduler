"""
Simulation environment, synthetic device/workload generator, and metrics aggregation.
"""

from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector, MetricSummary, compute_95_confidence_interval

__all__ = [
    "generate_heterogeneous_population",
    "generate_workload",
    "SimulationEnvironment",
    "MetricCollector",
    "MetricSummary",
    "compute_95_confidence_interval",
]
