"""
Configuration parameters for adaptive stability-aware scheduling simulation.
Distinguishes between paper-specified parameters and implementation assumptions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class SimConfig:
    """Simulation time & scale configuration."""
    # Paper-specified parameters (Section 13)
    step_minutes: float = 5.0  # 1 step = 5 minutes
    history_weeks: int = 4
    eval_weeks: int = 1
    num_tasks: int = 150
    default_num_devices: int = 30
    deadline_factor_range: Tuple[float, float] = (1.6, 3.0)
    
    # Paper-specified Monte Carlo seeds (Section 13, 16)
    main_seeds: int = 300
    correlated_seeds: int = 500
    
    # Computed properties
    @property
    def history_steps(self) -> int:
        # 4 weeks = 28 days * 24 hours * 12 steps/hour = 8064 steps
        return self.history_weeks * 7 * 24 * 12

    @property
    def eval_steps(self) -> int:
        # 1 week = 7 days * 24 hours * 12 steps/hour = 2016 steps
        return self.eval_weeks * 7 * 24 * 12


@dataclass
class ExecutionThresholds:
    """Paper-specified execution tier thresholds (Section 7, 17)."""
    R1: float = 0.05  # L0 -> L1 (Checkpointed)
    R2: float = 0.20  # L1 -> L2 (Volunteer Hedge)
    R3: float = 0.50  # L2 -> L3 (Cloud Fallback)


@dataclass
class StabilityWeights:
    """Weights for Stability Score SS_i = wa*A + wu*U + wn*N + wh*H (Section 3)."""
    # Implementation assumptions for unspecified weights (summing to 1.0)
    wa: float = 0.35  # Availability weight
    wu: float = 0.25  # Utilization / Capability weight
    wn: float = 0.15  # Network weight
    wh: float = 0.25  # Historical stability weight
    alpha: float = 0.30  # Exponential smoothing factor X_new = alpha*X_recent + (1-alpha)*X_history


@dataclass
class SuitabilityWeights:
    """Weights for Task-Device Suitability S_ij (Section 6)."""
    # Implementation assumptions for unspecified weights
    wc: float = 0.25  # Compute suitability
    wm: float = 0.20  # Memory suitability
    wn: float = 0.15  # Network suitability
    ws: float = 0.15  # Stability score factor
    wp: float = 0.15  # Survival probability factor
    wd: float = 0.10  # Deadline suitability
    wk: float = 0.05  # Cost penalty factor


@dataclass
class HazardConfig:
    """State-aware hazard multipliers m_k (Section 5)."""
    # Implementation assumptions for hazard multipliers
    m_battery: float = 2.50       # Multiplier when laptop is on battery power
    m_low_battery: float = 4.00   # Multiplier when battery level < 20%
    m_network_unstable: float = 1.80 # Multiplier when network fluctuates


@dataclass
class CoAvailabilityConfig:
    """Co-availability shrinkage estimator parameters (Section 10)."""
    # Implementation assumptions for shrinkage estimation
    alpha_s: float = 50.0      # Shrinkage weight parameter
    lambda_cost: float = 0.01  # Candidate hedge cost penalty factor


@dataclass
class WorkloadInstabilityParams:
    """MTBF parameters (in 5-min steps) for synthetic availability generation."""
    # Implementation assumptions matching Table 1 qualitative profiles
    stable_mtbf_steps: float = 650.0    # ~54 hours continuous uptime MTBF
    moderate_mtbf_steps: float = 200.0  # ~16.6 hours continuous uptime MTBF
    high_mtbf_steps: float = 75.0       # ~6.25 hours continuous uptime MTBF
    
    mean_repair_steps: float = 12.0     # ~1 hour mean time to repair/reconnect


def get_default_config() -> Dict:
    """Get complete system default configuration."""
    return {
        "sim": SimConfig(),
        "thresholds": ExecutionThresholds(),
        "stability": StabilityWeights(),
        "suitability": SuitabilityWeights(),
        "hazard": HazardConfig(),
        "co_avail": CoAvailabilityConfig(),
        "instability": WorkloadInstabilityParams(),
    }
