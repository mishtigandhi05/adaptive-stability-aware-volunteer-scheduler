"""
Unit tests for hazard adjustment, conditional survival, and Algorithm 1 Live Risk Controller.
"""

import pytest
from src.models.device import VolunteerDevice, DeviceType, PowerState
from src.models.task import Task, TaskState, ReplicaInstance
from src.risk.hazard import calculate_adjusted_hazard
from src.risk.survival import calculate_conditional_survival, calculate_interruption_risk
from src.risk.live_controller import evaluate_and_update_task_risk
from src.config import ExecutionThresholds, HazardConfig


def test_hazard_multipliers():
    dev = VolunteerDevice(
        device_id=1,
        device_type=DeviceType.LAPTOP,
        cores=2,
        ram_gb=4.0,
        uplink_mbps=1.0,
        power_state=PowerState.BATTERY,
        battery_level=0.10
    )
    base_h = 0.01
    cfg = HazardConfig(m_battery=2.0, m_low_battery=3.0, m_network_unstable=1.5)
    adj_h = calculate_adjusted_hazard(dev, base_h, cfg)
    # Expected multiplier = 2.0 * 3.0 * 1.5 = 9.0
    assert abs(adj_h - (0.01 * 9.0)) < 1e-5


def test_conditional_survival_and_risk():
    dev = VolunteerDevice(
        device_id=2,
        device_type=DeviceType.DESKTOP,
        cores=4,
        ram_gb=8.0,
        uplink_mbps=100.0,
        baseline_mtbf=100.0
    )
    p_surv = calculate_conditional_survival(dev, remaining_steps=10.0)
    risk = calculate_interruption_risk(dev, remaining_steps=10.0)
    assert 0.0 <= p_surv <= 1.0
    assert abs((p_surv + risk) - 1.0) < 1e-6


def test_algorithm_1_live_risk_control_transitions():
    dev = VolunteerDevice(
        device_id=3,
        device_type=DeviceType.DESKTOP,
        cores=4,
        ram_gb=8.0,
        uplink_mbps=10.0,
        baseline_mtbf=50.0
    )
    task = Task(task_id=1, expected_duration_steps=40.0)
    task.state = TaskState.RUNNING
    primary_rep = ReplicaInstance(replica_id="primary", device_id=3, is_cloud=False, active=True)
    task.replicas.append(primary_rep)
    
    thresholds = ExecutionThresholds(R1=0.05, R2=0.20, R3=0.50)
    
    # Evaluate risk transition
    level, new_rep = evaluate_and_update_task_risk(
        task=task,
        primary_device=dev,
        available_devices=[dev],
        current_step=10.0,
        thresholds=thresholds
    )
    assert level in [0, 1, 2, 3]
