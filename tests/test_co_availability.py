"""
Unit tests for co-availability bitmap metrics, shrinkage estimation, and Algorithm 2 hedge selection.
"""

import pytest
import numpy as np
from src.models.device import VolunteerDevice, DeviceType
from src.models.task import Task
from src.scheduling.co_availability import (
    calculate_empirical_joint_failure,
    calculate_independent_joint_failure,
    calculate_shrinkage_joint_failure,
    select_co_availability_hedge,
)


def test_empirical_joint_failure():
    d1 = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[1, 1, 0, 0, 1])
    d2 = VolunteerDevice(2, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[1, 0, 0, 1, 1])
    
    # Both are 0 at slot index 2 (slot 3). So 1 out of 5 slots has both 0 -> 0.20
    f_emp = calculate_empirical_joint_failure([d1, d2])
    assert abs(f_emp - 0.20) < 1e-5


def test_shrinkage_joint_failure():
    d1 = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[1, 1, 0, 1])
    d2 = VolunteerDevice(2, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[1, 0, 1, 1])
    f_shrink = calculate_shrinkage_joint_failure([d1, d2])
    assert 0.0 <= f_shrink <= 1.0


def test_algorithm_2_hedge_selection():
    d1 = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[1, 1, 1, 0])
    candidate1 = VolunteerDevice(2, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[0, 1, 1, 1])
    candidate2 = VolunteerDevice(3, DeviceType.DESKTOP, 4, 8.0, 10.0, history_bitmap=[1, 1, 1, 0])
    
    task = Task(task_id=1, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=5.0)
    
    # Candidate 1 is complementary to d1 (d1 is 0 when c1 is 1)
    best = select_co_availability_hedge(
        primary_set=[d1],
        candidate_devices=[candidate1, candidate2],
        task=task
    )
    assert best is not None
    assert best.device_id == 2  # candidate1 selected due to co-availability independence
