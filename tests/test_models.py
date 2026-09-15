"""
Unit tests for domain models: Device, Task, Checkpoint, and Stability.
"""

import pytest
import numpy as np
from src.models.device import VolunteerDevice, DeviceType, PowerState
from src.models.task import Task, TaskState
from src.models.checkpoint import Checkpoint
from src.models.stability import calculate_stability_score, update_exponential_moving_average


def test_device_creation_and_power_state():
    dev = VolunteerDevice(
        device_id=1,
        device_type=DeviceType.LAPTOP,
        cores=4,
        ram_gb=8.0,
        uplink_mbps=10.0,
        power_state=PowerState.BATTERY,
        battery_level=0.15
    )
    assert dev.device_type == DeviceType.LAPTOP
    assert dev.cores == 4
    factors = dev.get_hazard_factors()
    assert "battery" in factors
    assert "low_battery" in factors


def test_task_deadline_and_checkpoint():
    task = Task(
        task_id=10,
        req_cores=2,
        expected_duration_steps=20.0,
        deadline_factor=2.0,
        arrival_step=5.0
    )
    assert task.deadline_step == 45.0  # 5 + (20 * 2)
    
    ckpt = task.add_checkpoint(current_step=15.0)
    assert ckpt.seq_number == 1
    assert task.checkpoint_storage_mb == 100.0


def test_stability_score_computation():
    ss = calculate_stability_score(
        availability=1.0,
        utilization=0.8,
        network=0.9,
        history_score=0.95
    )
    assert 0.0 <= ss <= 1.0
    
    updated = update_exponential_moving_average(0.8, 0.5, alpha=0.3)
    assert abs(updated - (0.3 * 0.8 + 0.7 * 0.5)) < 1e-6
