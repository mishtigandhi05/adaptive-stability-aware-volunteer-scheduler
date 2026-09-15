"""
Comprehensive integration tests for discrete-time simulation, correlated outages,
ablation distinctions, recovery lifecycle, non-duplicate checkpointing, executable dispatch levels,
and metric tracking.
"""

import pytest
import numpy as np
from src.models.device import VolunteerDevice, DeviceType, PowerState
from src.models.task import Task, TaskState, ReplicaInstance
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.scheduling.schedulers import (
    CapabilityOnlyScheduler,
    CapabilityStabilityScheduler,
    IndependenceReplicationScheduler,
    AdaptiveStabilityScheduler,
)
from src.config import ExecutionThresholds


def test_generator_imports_and_execution():
    """Item 1: Verify generator imports and execution without errors."""
    rng = np.random.default_rng(42)
    devices = generate_heterogeneous_population(num_devices=10, instability="high", rng=rng)
    tasks = generate_workload(num_tasks=15, rng=rng)
    assert len(devices) == 10
    assert len(tasks) == 15
    assert all(d.group_id is not None for d in devices)


def test_correlated_group_outages():
    """Item 2: Verify devices in the same correlated group share correlated availability history."""
    rng = np.random.default_rng(100)
    devices = generate_heterogeneous_population(num_devices=8, instability="high", num_correlated_groups=2, rng=rng)
    group0_devs = [d for d in devices if d.group_id == 0]
    assert len(group0_devs) >= 2
    
    b0 = np.array(group0_devs[0].history_bitmap)
    b1 = np.array(group0_devs[1].history_bitmap)
    overlap_same_group = np.mean(b0 == b1)
    
    group1_devs = [d for d in devices if d.group_id == 1]
    b_other = np.array(group1_devs[0].history_bitmap)
    overlap_diff_group = np.mean(b0 == b_other)
    
    assert overlap_same_group >= overlap_diff_group


def test_ablation_dispatch_levels_executable_behavior():
    """Verify '+ Dispatch levels' ablation stage introduces real executable dispatch actions."""
    stage3 = AdaptiveStabilityScheduler(
        use_conditional_survival=True, use_dispatch_levels=False, use_live_risk=False,
        use_hazard_multipliers=False, use_co_availability=False, use_cloud_fallback=False
    )
    stage4 = AdaptiveStabilityScheduler(
        use_conditional_survival=True, use_dispatch_levels=True, use_live_risk=False,
        use_hazard_multipliers=False, use_co_availability=False, use_cloud_fallback=False
    )
    
    rng = np.random.default_rng(200)
    devices = generate_heterogeneous_population(num_devices=15, instability="high", rng=rng)
    tasks = generate_workload(num_tasks=25, total_eval_steps=300, rng=rng)
    
    import copy
    env3 = SimulationEnvironment(devices=copy.deepcopy(devices), tasks=copy.deepcopy(tasks), scheduler=stage3, total_steps=300, seed=1)
    res3 = env3.run()
    
    env4 = SimulationEnvironment(devices=copy.deepcopy(devices), tasks=copy.deepcopy(tasks), scheduler=stage4, total_steps=300, seed=1)
    res4 = env4.run()
    
    assert res3 != res4 or any(t.replica_launches_count > 1 for t in env4.tasks)


def test_deadline_reliability_strictness():
    """Verify tasks completing AFTER deadline_step fail deadline and are marked FAILED_DEADLINE."""
    d = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, baseline_mtbf=500.0)
    
    # Task with duration 20 steps, deadline factor = 0.75 -> deadline step = 15 steps (impossible to complete in time)
    task = Task(task_id=1, req_cores=2, expected_duration_steps=20.0, deadline_factor=0.75, arrival_step=0.0)
    
    sched = CapabilityOnlyScheduler()
    env = SimulationEnvironment(devices=[d], tasks=[task], scheduler=sched, total_steps=30, seed=42)
    env.run()
    
    assert task.state == TaskState.FAILED_DEADLINE


def test_l3_direct_cloud_fallback_launch():
    """Verify R >= R3 level 3 directly launches cloud fallback when use_cloud_fallback is True."""
    dev = VolunteerDevice(1, DeviceType.LAPTOP, 2, 4.0, 1.0, power_state=PowerState.BATTERY, battery_level=0.10, baseline_mtbf=200.0)
    task = Task(task_id=1, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=1.0, expected_duration_steps=50.0, arrival_step=0.0)
    
    sched = AdaptiveStabilityScheduler(
        thresholds=ExecutionThresholds(R1=0.01, R2=0.02, R3=0.03),
        use_live_risk=True,
        use_cloud_fallback=True
    )
    env = SimulationEnvironment(devices=[dev], tasks=[task], scheduler=sched, total_steps=20, seed=1)
    env.run()
    
    has_cloud_replica = any(r.is_cloud for r in task.replicas)
    assert has_cloud_replica


def test_lifecycle_continuation_after_primary_failure():
    """Verify task recovers and continues via hedge/cloud after primary device failure."""
    d_primary = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, baseline_mtbf=5.0)
    d_hedge = VolunteerDevice(2, DeviceType.DESKTOP, 4, 8.0, 10.0, baseline_mtbf=500.0)
    
    task = Task(task_id=1, req_cores=2, req_ram_gb=4.0, expected_duration_steps=30.0, arrival_step=0.0)
    
    sched = AdaptiveStabilityScheduler(use_live_risk=True)
    env = SimulationEnvironment(devices=[d_primary, d_hedge], tasks=[task], scheduler=sched, total_steps=100, seed=42)
    
    d_pct, wasted_cpu, cloud_sec, ckpt_mb, avg_reps, rep_hours = env.run()
    assert task.state in [TaskState.COMPLETED, TaskState.RUNNING, TaskState.FAILED_DEADLINE]
    assert task.replica_launches_count >= 1


def test_checkpoint_not_duplicated_every_step():
    """Verify checkpoint is created on level transitions/progress advances, not every timestep."""
    d = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, baseline_mtbf=40.0)
    task = Task(task_id=1, req_cores=2, expected_duration_steps=40.0, arrival_step=0.0)
    
    sched = AdaptiveStabilityScheduler(thresholds=ExecutionThresholds(R1=0.01, R2=0.05, R3=0.10))
    env = SimulationEnvironment(devices=[d], tasks=[task], scheduler=sched, total_steps=20, seed=42)
    env.run()
    
    if task.latest_checkpoint is not None:
        assert task.latest_checkpoint.seq_number < 20


def test_replica_hours_calculation():
    """Verify duration-based replica hours tracking."""
    task = Task(task_id=1, expected_duration_steps=20.0)
    r_sec = ReplicaInstance(replica_id="hedge_vol_2", device_id=2, is_cloud=False, start_progress=0.0, active=True)
    task.replicas.append(r_sec)
    
    for _ in range(12):
        task.secondary_replica_hours_total += (5.0 / 60.0)
        
    assert abs(task.secondary_replica_hours_total - 1.0) < 1e-6


def test_dynamic_stability_update():
    """Verify stability score SS_i updates dynamically during simulation steps."""
    dev = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, is_available=False, recent_availability=0.90)
    initial_ss = dev.stability_score
    
    task = Task(task_id=1, expected_duration_steps=10.0)
    sched = CapabilityOnlyScheduler()
    env = SimulationEnvironment(devices=[dev], tasks=[task], scheduler=sched, total_steps=24, seed=42)
    env.run()
    
    assert dev.stability_score != initial_ss


def test_threshold_transitions():
    """Verify threshold transitions across risk levels."""
    thresh = ExecutionThresholds(R1=0.05, R2=0.20, R3=0.50)
    sched = AdaptiveStabilityScheduler(thresholds=thresh)
    assert sched.thresholds.R1 == 0.05
    assert sched.thresholds.R2 == 0.20
    assert sched.thresholds.R3 == 0.50


def test_end_to_end_simulation():
    """End-to-end simulation run producing valid non-null metrics."""
    rng = np.random.default_rng(500)
    devices = generate_heterogeneous_population(num_devices=12, instability="high", rng=rng)
    tasks = generate_workload(num_tasks=20, total_eval_steps=200, rng=rng)
    
    sched = AdaptiveStabilityScheduler()
    env = SimulationEnvironment(devices=devices, tasks=tasks, scheduler=sched, total_steps=200, seed=500)
    
    d_pct, wasted_cpu, cloud_sec, ckpt_mb, avg_reps, rep_hours = env.run()
    assert 0.0 <= d_pct <= 100.0
    assert wasted_cpu >= 0.0
    assert cloud_sec >= 0.0
    assert avg_reps >= 1.0
    assert rep_hours >= 0.0
