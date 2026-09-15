"""
State-aware hazard adjustment module.
Paper Section 5.
"""

from typing import List
from src.config import HazardConfig
from src.models.device import VolunteerDevice


def calculate_adjusted_hazard(
    device: VolunteerDevice,
    baseline_hazard: float,
    config: HazardConfig = HazardConfig()
) -> float:
    """
    Calculate adjusted hazard rate (Section 5):
    lambda_i_adj(t) = lambda_i(t) * prod_{k in K_i} m_k
    """
    active_factors = device.get_hazard_factors()
    multiplier = 1.0
    
    for factor in active_factors:
        if factor == "battery":
            multiplier *= config.m_battery
        elif factor == "low_battery":
            multiplier *= config.m_low_battery
        elif factor == "network_unstable":
            multiplier *= config.m_network_unstable
            
    return baseline_hazard * multiplier
