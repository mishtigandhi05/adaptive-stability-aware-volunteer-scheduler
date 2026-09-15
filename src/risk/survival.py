"""
Conditional survival probability and interruption risk model.
Paper Section 4 & 5.
"""

import math
from src.models.device import VolunteerDevice
from src.risk.hazard import calculate_adjusted_hazard
from src.config import HazardConfig


def calculate_conditional_survival(
    device: VolunteerDevice,
    remaining_steps: float,
    config: HazardConfig = HazardConfig()
) -> float:
    """
    Calculate conditional survival probability for remaining execution period tau (Section 4 & 5):
    P_i_adj(tau) = exp( - int_{a_i}^{a_i + tau} lambda_i_adj(u) du )
    
    Implementation Assumption:
    Documented as a simplified exponential conditional-survival approximation:
    P(T >= a_i + tau | T >= a_i) = exp(-adj_hazard * tau)
    """
    if remaining_steps <= 0:
        return 1.0
        
    baseline_hazard = 1.0 / max(1.0, device.baseline_mtbf)
    adj_hazard = calculate_adjusted_hazard(device, baseline_hazard, config)
    
    prob = math.exp(-adj_hazard * remaining_steps)
    return max(0.0, min(1.0, float(prob)))


def calculate_interruption_risk(
    device: VolunteerDevice,
    remaining_steps: float,
    config: HazardConfig = HazardConfig()
) -> float:
    """
    Calculate interruption risk (Section 4 & 5):
    R_i(tau) = 1 - P_i_adj(tau), bounded in [0.0, 1.0].
    """
    p_survival = calculate_conditional_survival(device, remaining_steps, config)
    return max(0.0, min(1.0, 1.0 - p_survival))
