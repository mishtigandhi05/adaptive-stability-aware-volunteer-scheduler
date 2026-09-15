"""
Co-Availability-Aware Hedge Selection module implementing Algorithm 2 from Section 10 of the paper.
Uses historical availability bitmaps, empirical joint failure probabilities, independence bounds, and shrinkage estimation.
"""

from typing import List, Optional
import numpy as np
from src.config import CoAvailabilityConfig
from src.models.device import VolunteerDevice
from src.models.task import Task
from src.scheduling.suitability import filter_capable_devices


def calculate_empirical_joint_failure(devices: List[VolunteerDevice]) -> float:
    """
    Calculate empirical joint failure probability F_emp(R) (Section 10):
    F_emp(R) = (1 / N) * sum_{w=1}^N prod_{i in R} (1 - x_i(w))
    """
    if not devices:
        return 1.0
        
    bitmaps = [np.array(d.history_bitmap, dtype=int) for d in devices if len(d.history_bitmap) > 0]
    if not bitmaps:
        return 1.0
        
    min_len = min(len(b) for b in bitmaps)
    if min_len == 0:
        return 1.0
        
    # Stack bitmaps up to min_len
    stacked = np.column_stack([b[:min_len] for b in bitmaps])
    
    # prod_{i in R} (1 - x_i(w)): 1 when ALL devices in set are 0 (unavailable) at slot w
    unavail_all = np.prod(1 - stacked, axis=1)
    f_emp = float(np.mean(unavail_all))
    return f_emp


def calculate_independent_joint_failure(devices: List[VolunteerDevice]) -> float:
    """
    Calculate independence joint failure probability F_ind(R) (Section 10):
    F_ind(R) = prod_{i in R} (1 - p_i)
    """
    if not devices:
        return 1.0
        
    f_ind = 1.0
    for d in devices:
        # p_i is individual availability score
        p_i = max(0.01, min(0.99, d.recent_availability))
        f_ind *= (1.0 - p_i)
    return float(f_ind)


def calculate_shrinkage_joint_failure(
    devices: List[VolunteerDevice],
    config: CoAvailabilityConfig = CoAvailabilityConfig()
) -> float:
    """
    Calculate shrinkage joint failure estimate F(R) (Section 10):
    F(R) = (N * F_emp(R) + alpha_s * F_ind(R)) / (N + alpha_s)
    """
    if not devices:
        return 1.0
        
    N = min((len(d.history_bitmap) for d in devices if len(d.history_bitmap) > 0), default=8064)
    f_emp = calculate_empirical_joint_failure(devices)
    f_ind = calculate_independent_joint_failure(devices)
    
    f_shrink = (N * f_emp + config.alpha_s * f_ind) / (N + config.alpha_s)
    return float(f_shrink)


def select_co_availability_hedge(
    primary_set: List[VolunteerDevice],
    candidate_devices: List[VolunteerDevice],
    task: Task,
    config: CoAvailabilityConfig = CoAvailabilityConfig()
) -> Optional[VolunteerDevice]:
    """
    Implements Algorithm 2: Co-Availability-Aware Hedge Selection (Paper Section 10 & Algorithm 2).
    """
    # 1. Remove candidates that fail capability requirements
    capable_candidates = filter_capable_devices(candidate_devices, task)
    
    # Filter out devices already in primary set
    primary_ids = {d.device_id for d in primary_set}
    candidates = [d for d in capable_candidates if d.device_id not in primary_ids]
    
    if not candidates:
        return None
        
    best_candidate = None
    min_objective = float("inf")
    
    # 2. Loop over candidates k in C
    for candidate in candidates:
        candidate_set = primary_set + [candidate]
        
        # Estimate F(R u {k})
        f_joint = calculate_shrinkage_joint_failure(candidate_set, config)
        
        # Candidate cost (network transfer / uplink cost)
        cost_k = 1.0 / max(1.0, candidate.uplink_mbps)
        
        # Combined objective: F(R u {k}) + lambda * Cost_k
        objective = f_joint + config.lambda_cost * cost_k
        
        if objective < min_objective:
            min_objective = objective
            best_candidate = candidate
            
    return best_candidate
