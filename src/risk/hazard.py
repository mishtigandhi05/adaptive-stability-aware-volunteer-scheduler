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
    Calculate adjusted hazard rate (Section 5 & simplified exponential model):
    adjusted_hazard = baseline_hazard * age_factor * prod_{k in K_i} m_k
    
    Implementation Assumption:
    Documented as a simplified exponential conditional-survival approximation.
    State factors include:
    - baseline hazard (1 / MTBF)
    - session age factor (1.0 at age 0, increasing as session ages)
    - power state: AC (1.0) vs Battery (m_battery)
    - battery state: low battery level < 20% (m_low_battery)
    - network state: fluctuation or low uplink (m_network_unstable)
    """
    # Age factor: older sessions gradually experience higher interruption hazard
    age_factor = 1.0 + min(1.0, max(0.0, device.current_session_age) / max(1.0, device.baseline_mtbf * 2.0))
    
    multiplier = age_factor
    active_factors = device.get_hazard_factors()
    
    for factor in active_factors:
        if factor == "battery":
            multiplier *= config.m_battery
        elif factor == "low_battery":
            multiplier *= config.m_low_battery
        elif factor == "network_unstable":
            multiplier *= config.m_network_unstable
            
    return baseline_hazard * multiplier
