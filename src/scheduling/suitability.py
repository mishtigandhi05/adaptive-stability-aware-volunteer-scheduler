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
    weights: SuitabilityWeights = SuitabilityWeights(),
    current_step: Optional[float] = None
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
    # Authoritative calculation incorporating device CPU speed factor
    speed_factor = device.cpu_speed_factor if hasattr(device, "cpu_speed_factor") else 1.0
    remaining_steps = task.get_remaining_steps(speed_factor)
    p_survival = calculate_conditional_survival(device, remaining_steps)
    
    # Deadline suitability based on remaining slack (Fix Item 1):
    # remaining_time_to_deadline = max(0, deadline_step - current_step)
    # deadline_ratio = remaining_time_to_deadline / remaining_steps
    # d_ij = clamp(deadline_ratio, 0, 1)
    step_now = current_step if current_step is not None else task.arrival_step
    remaining_time_to_deadline = max(0.0, task.deadline_step - step_now)
    deadline_ratio = remaining_time_to_deadline / max(1.0, remaining_steps)
    d_ij = max(0.0, min(1.0, deadline_ratio))
    
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
