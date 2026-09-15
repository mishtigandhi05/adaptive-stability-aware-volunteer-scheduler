"""
Metrics collection and statistical confidence interval calculation.
Paper Section 13 & 14.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class MetricSummary:
    """Stores mean and 95% confidence interval bounds for paper reported metrics."""
    mean: float
    ci95: float
    std: float
    n_samples: int

    def __str__(self) -> str:
        return f"{self.mean:.2f} ± {self.ci95:.2f}"


def compute_95_confidence_interval(data: List[float]) -> MetricSummary:
    """
    Compute mean and 95% confidence interval using paper formula (Section 13):
    bar_x +/- 1.96 * (s / sqrt(n))
    """
    if not data:
        return MetricSummary(0.0, 0.0, 0.0, 0)
        
    arr = np.array(data, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    
    if n <= 1:
        return MetricSummary(mean_val, 0.0, 0.0, n)
        
    std_val = float(np.std(arr, ddof=1))
    ci95_val = 1.96 * (std_val / np.sqrt(n))
    return MetricSummary(mean_val, float(ci95_val), std_val, n)


class MetricCollector:
    """Aggregates metrics across simulation seeds."""
    def __init__(self):
        self.deadline_completion_rates: List[float] = []
        self.wasted_cpu_times: List[float] = []
        self.cloud_inst_secs: List[float] = []
        self.checkpoint_storages: List[float] = []
        self.replica_launches: List[float] = []
        self.replica_hours: List[float] = []  # Duration-based replica-hours/task

    def add_run_result(
        self,
        deadline_rate: float,
        wasted_cpu: float,
        cloud_inst_sec: float,
        checkpoint_storage: float,
        avg_replicas: float,
        avg_replica_hours: float = 0.0
    ):
        self.deadline_completion_rates.append(deadline_rate)
        self.wasted_cpu_times.append(wasted_cpu)
        self.cloud_inst_secs.append(cloud_inst_sec)
        self.checkpoint_storages.append(checkpoint_storage)
        self.replica_launches.append(avg_replicas)
        self.replica_hours.append(avg_replica_hours)

    def summarize(self) -> Dict[str, MetricSummary]:
        return {
            "deadline_pct": compute_95_confidence_interval(self.deadline_completion_rates),
            "wasted_cpu": compute_95_confidence_interval(self.wasted_cpu_times),
            "cloud_inst_sec": compute_95_confidence_interval(self.cloud_inst_secs),
            "checkpoint_storage": compute_95_confidence_interval(self.checkpoint_storages),
            "replica_launches": compute_95_confidence_interval(self.replica_launches),
            "replica_hours": compute_95_confidence_interval(self.replica_hours),
        }
