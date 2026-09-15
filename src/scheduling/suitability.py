"""
Task-Device Suitability model (Paper Section 6).
Evaluates basic hardware filtering and computes suitability score S_ij.
"""

from typing import List, Optional
from src.config import SuitabilityWeights
from src.models.device import VolunteerDevice
from src.models.task import Task
from src.risk.survival import calculate_conditional_survival


def filter_capable_devices(devices: List[VolunteerDevice], task: Task) -> List[VolunteerDevice]:
    """Filter devices satisfying basic hardware requirements (Section 6)."""
    return [
        d for d in devices
        if d.is_available
        and d.cores >= task.req_cores
        and d.ram_gb >= task.req_ram_gb
        and d.uplink_mbps >= task.req_uplink_mbps
    ]


def calculate_suitability_score(
    device: VolunteerDevice,
    task: Task,
    weights: SuitabilityWeights = SuitabilityWeights()
) -> float:
    """
    Calculate Task-Device Suitability score S_ij (Section 6):
    S_ij = wc*C_ij + wm*M_ij + wn*N_ij + ws*SS_i + wp*P_i(tau_j) + wd*D_ij - wk*K_ij
    """
    # Normalized capability factors
    c_ij = min(1.0, device.cores / max(1.0, task.req_cores))
    m_ij = min(1.0, device.ram_gb / max(1.0, task.req_ram_gb))
    n_ij = min(1.0, device.uplink_mbps / max(1.0, task.req_uplink_mbps))
    
    # Device stability score SS_i
    ss_i = device.stability_score
    
    # Conditional survival probability P_i(tau_j) for remaining task steps
    remaining_fraction = max(0.0, 1.0 - task.completed_progress)
    remaining_steps = remaining_fraction * task.expected_duration_steps
    p_survival = calculate_conditional_survival(device, remaining_steps)
    
    # Deadline suitability (speed ratio)
    d_ij = min(1.0, task.deadline_step / max(1.0, remaining_steps))
    
    # Estimated transfer/checkpoint cost penalty
    k_ij = 0.1 if device.uplink_mbps < 10.0 else 0.02
    
    score = (
        weights.wc * c_ij +
        weights.wm * m_ij +
        weights.wn * n_ij +
        weights.ws * ss_i +
        weights.wp * p_survival +
        weights.wd * d_ij -
        weights.wk * k_ij
    )
    return float(score)
