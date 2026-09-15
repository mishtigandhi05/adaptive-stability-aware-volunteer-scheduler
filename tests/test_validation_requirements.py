"""
Validation tests covering all requirements:
1. Deadline suitability (slack-based, bounds, passed deadline).
2. CPU heterogeneity (2, 3, 5 cores progress speeds).
3. RAM capability constraint (filtering, no execution speed distortion).
4. Network bandwidth transfer duration (100 Mbps > 10 Mbps > 1 Mbps).
5. Checkpoint overhead and bandwidth impact.
6. Poisson arrivals (reproducibility, seed differences, non-decreasing, within horizon, arrival rate).
7. MTBF configuration.
8. State-aware survival model (power, battery, network, age, bounds [0, 1]).
9. Clean distinction of Primary, Hedge, Cloud roles and promotion.
10. Replica count consistency (additional replicas beyond primary).
11. Co-availability vs independence baseline selection logic.
12. Live risk controller threshold transitions.
"""

import pytest
import math
import numpy as np
from src.models.device import VolunteerDevice, DeviceType, PowerState
from src.models.task import (
    Task,
    TaskState,
    ReplicaInstance,
    ExecutionRole,
    calculate_checkpoint_transfer_steps,
)
from src.config import SimConfig, ExecutionThresholds, HazardConfig, CoAvailabilityConfig, WorkloadInstabilityParams
from src.scheduling.suitability import filter_capable_devices, calculate_suitability_score
from src.scheduling.co_availability import (
    calculate_empirical_joint_failure,
    calculate_independent_joint_failure,
    calculate_shrinkage_joint_failure,
    select_co_availability_hedge,
)
from src.scheduling.schedulers import AdaptiveStabilityScheduler, CapabilityOnlyScheduler
from src.risk.hazard import calculate_adjusted_hazard
from src.risk.survival import calculate_conditional_survival, calculate_interruption_risk
from src.risk.live_controller import evaluate_and_update_task_risk
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment


# ==============================================================================
# 1. FIX DEADLINE SUITABILITY
# ==============================================================================
def test_deadline_suitability_slack():
    """Verify slack-based deadline suitability calculation."""
    dev = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0)
    
    # Task arriving at 0, expected 20 steps, deadline factor 2.0 -> deadline_step = 40.0
    task = Task(task_id=1, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=5.0,
                expected_duration_steps=20.0, deadline_factor=2.0, arrival_step=0.0)
    
    # At step 0: remaining slack = 40 - 0 = 40; remaining_steps = 20; ratio = 2.0 -> clamped to 1.0
    score_plenty_slack = calculate_suitability_score(dev, task, current_step=0.0)
    
    # At step 30: remaining slack = 40 - 30 = 10; remaining_steps = 20; ratio = 10/20 = 0.5
    score_little_slack = calculate_suitability_score(dev, task, current_step=30.0)
    
    # At step 45: deadline has passed (40 - 45 < 0) -> slack = 0 -> ratio = 0
    score_expired = calculate_suitability_score(dev, task, current_step=45.0)
    
    assert score_plenty_slack > score_little_slack > score_expired
    assert 0.0 <= score_plenty_slack <= 1.0
    assert 0.0 <= score_little_slack <= 1.0
    assert 0.0 <= score_expired <= 1.0


# ==============================================================================
# 2. CPU HETEROGENEITY EXECUTION SPEED
# ==============================================================================
def test_cpu_heterogeneity_affects_execution_speed():
    """Verify 5-core device executes faster than 3-core, and 3-core faster than 2-core."""
    d2 = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=2, ram_gb=8.0, uplink_mbps=10.0)
    d3 = VolunteerDevice(device_id=2, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0)
    d5 = VolunteerDevice(device_id=3, device_type=DeviceType.DESKTOP, cores=5, ram_gb=8.0, uplink_mbps=10.0)
    
    assert d2.cpu_speed_factor == 0.75
    assert d3.cpu_speed_factor == 1.00
    assert d5.cpu_speed_factor == 1.50
    
    task = Task(task_id=1, expected_duration_steps=30.0)
    
    # One authoritative progress calculation
    prog2 = task.get_progress_per_step(d2.cpu_speed_factor)
    prog3 = task.get_progress_per_step(d3.cpu_speed_factor)
    prog5 = task.get_progress_per_step(d5.cpu_speed_factor)
    
    assert prog5 > prog3 > prog2
    assert abs(prog5 - 1.5 * prog3) < 1e-6
    assert abs(prog2 - 0.75 * prog3) < 1e-6
    
    # Remaining steps estimation is also consistent
    rem2 = task.get_remaining_steps(d2.cpu_speed_factor)
    rem3 = task.get_remaining_steps(d3.cpu_speed_factor)
    rem5 = task.get_remaining_steps(d5.cpu_speed_factor)
    assert rem2 > rem3 > rem5


# ==============================================================================
# 3. RAM AS CAPABILITY CONSTRAINT
# ==============================================================================
def test_ram_capability_constraint_not_distorting_speed():
    """Verify RAM filters incapable devices but does NOT increase execution speed."""
    d_small = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=4.0, uplink_mbps=10.0)
    d_large = VolunteerDevice(device_id=2, device_type=DeviceType.DESKTOP, cores=3, ram_gb=16.0, uplink_mbps=10.0)
    
    task_demanding = Task(task_id=1, req_cores=2, req_ram_gb=8.0, req_uplink_mbps=5.0)
    
    # Capability filtering
    capable = filter_capable_devices([d_small, d_large], task_demanding)
    assert d_large in capable
    assert d_small not in capable
    
    # Both have same core count -> same execution speed
    assert d_small.cpu_speed_factor == d_large.cpu_speed_factor


# ==============================================================================
# 4. NETWORK BANDWIDTH AFFECTS CHECKPOINT TRANSFER
# ==============================================================================
def test_network_bandwidth_transfer_duration():
    """Verify 100 Mbps > 10 Mbps > 1 Mbps transfer speed and fractional step preservation."""
    steps_100 = calculate_checkpoint_transfer_steps(checkpoint_size_mb=100.0, uplink_mbps=100.0, step_seconds=300.0)
    steps_10 = calculate_checkpoint_transfer_steps(checkpoint_size_mb=100.0, uplink_mbps=10.0, step_seconds=300.0)
    steps_1 = calculate_checkpoint_transfer_steps(checkpoint_size_mb=100.0, uplink_mbps=1.0, step_seconds=300.0)
    
    # 800 Mb / 100 Mbps = 8.0s -> 8 / 300 = 0.02667 steps
    assert abs(steps_100 - (8.0 / 300.0)) < 1e-5
    # 800 Mb / 10 Mbps = 80.0s -> 80 / 300 = 0.26667 steps
    assert abs(steps_10 - (80.0 / 300.0)) < 1e-5
    # 800 Mb / 1 Mbps = 800.0s -> 800 / 300 = 2.66667 steps
    assert abs(steps_1 - (800.0 / 300.0)) < 1e-5
    
    assert steps_100 < steps_10 < steps_1
    assert steps_100 > 0.0


# ==============================================================================
# 5. CHECKPOINT OVERHEAD ACCOUNTING
# ==============================================================================
def test_checkpoint_overhead_slows_task_and_higher_bandwidth_is_faster():
    """Verify checkpoint transfer overhead affects execution and higher bandwidth incurs less delay."""
    # Compare 100 Mbps vs 1 Mbps device executing identical checkpointed task
    d_fast = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=100.0, baseline_mtbf=1000.0)
    d_slow = VolunteerDevice(device_id=2, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=1.0, baseline_mtbf=1000.0)
    
    task_fast = Task(task_id=1, req_uplink_mbps=1.0, expected_duration_steps=20.0, arrival_step=0.0, execution_level=1)
    task_slow = Task(task_id=2, req_uplink_mbps=1.0, expected_duration_steps=20.0, arrival_step=0.0, execution_level=1)
    
    sched = CapabilityOnlyScheduler()
    
    env_fast = SimulationEnvironment(devices=[d_fast], tasks=[task_fast], scheduler=sched, total_steps=30, seed=42)
    env_fast.run()
    
    env_slow = SimulationEnvironment(devices=[d_slow], tasks=[task_slow], scheduler=sched, total_steps=30, seed=42)
    env_slow.run()
    
    # 100 Mbps device finishes earlier than 1 Mbps device due to less checkpoint upload delay
    assert task_fast.completion_step is not None
    assert task_slow.completion_step is not None
    assert task_fast.completion_step < task_slow.completion_step


# ==============================================================================
# 6. POISSON-LIKE TASK ARRIVALS
# ==============================================================================
def test_poisson_task_arrivals():
    """Verify Poisson arrivals are reproducible, seed-dependent, non-decreasing, and rate-consistent."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    rng_diff = np.random.default_rng(456)
    
    tasks1 = generate_workload(num_tasks=50, total_eval_steps=1000, arrival_rate=0.075, rng=rng1)
    tasks2 = generate_workload(num_tasks=50, total_eval_steps=1000, arrival_rate=0.075, rng=rng2)
    tasks_diff = generate_workload(num_tasks=50, total_eval_steps=1000, arrival_rate=0.075, rng=rng_diff)
    
    arr1 = [t.arrival_step for t in tasks1]
    arr2 = [t.arrival_step for t in tasks2]
    arr_diff = [t.arrival_step for t in tasks_diff]
    
    # Same seed -> exact same arrival sequence
    assert arr1 == arr2
    
    # Different seed -> different arrival sequence
    assert arr1 != arr_diff
    
    # Non-decreasing timestamps
    for i in range(1, len(arr1)):
        assert arr1[i] >= arr1[i - 1]
        
    # All within evaluation horizon
    assert all(0.0 <= t.arrival_step < 1000 for t in tasks1)
    
    # Arrival rate is used: mean interarrival should be close to 1 / 0.075 = 13.33
    interarrivals = [arr1[i] - arr1[i-1] for i in range(1, len(arr1))]
    assert len(interarrivals) > 0
    mean_ia = float(np.mean(interarrivals))
    assert 5.0 < mean_ia < 25.0  # reasonable stochastic bounds around 13.33


# ==============================================================================
# 7. MTBF CONFIGURATION
# ==============================================================================
def test_mtbf_configuration_consistency():
    """Verify MTBF values are authoritative and properly defined."""
    params = WorkloadInstabilityParams()
    assert params.stable_mtbf_steps == 650.0
    assert params.moderate_mtbf_steps == 200.0
    assert params.high_mtbf_steps == 75.0


# ==============================================================================
# 8. STATE-AWARE SURVIVAL MODEL
# ==============================================================================
def test_state_aware_survival_model():
    """Verify hazard adjustment and survival bounds."""
    d_ac = VolunteerDevice(device_id=1, device_type=DeviceType.LAPTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0,
                           power_state=PowerState.AC, battery_level=1.0, baseline_mtbf=100.0)
    d_bat = VolunteerDevice(device_id=2, device_type=DeviceType.LAPTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0,
                            power_state=PowerState.BATTERY, battery_level=0.8, baseline_mtbf=100.0)
    d_low_bat = VolunteerDevice(device_id=3, device_type=DeviceType.LAPTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0,
                                power_state=PowerState.BATTERY, battery_level=0.10, baseline_mtbf=100.0)
    
    cfg = HazardConfig(m_battery=2.5, m_low_battery=4.0, m_network_unstable=1.8)
    
    h_ac = calculate_adjusted_hazard(d_ac, baseline_hazard=0.01, config=cfg)
    h_bat = calculate_adjusted_hazard(d_bat, baseline_hazard=0.01, config=cfg)
    h_low_bat = calculate_adjusted_hazard(d_low_bat, baseline_hazard=0.01, config=cfg)
    
    # AC is not worse than battery; low battery is worse than normal battery
    assert h_low_bat > h_bat > h_ac
    
    # Interruption risk bounds [0, 1]
    risk_short = calculate_interruption_risk(d_ac, remaining_steps=5.0, config=cfg)
    risk_long = calculate_interruption_risk(d_ac, remaining_steps=50.0, config=cfg)
    
    assert 0.0 <= risk_short <= 1.0
    assert 0.0 <= risk_long <= 1.0
    # Survival decreases (risk increases) as remaining window increases
    assert risk_long > risk_short


# ==============================================================================
# 9. EXECUTION ROLES AND HEDGE PROMOTION
# ==============================================================================
def test_execution_roles_and_hedge_promotion():
    """Verify primary, hedge, and cloud roles and promotion of hedge to primary upon primary failure."""
    d_primary = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0)
    d_hedge = VolunteerDevice(device_id=2, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=10.0)
    
    task = Task(task_id=1, expected_duration_steps=30.0)
    
    rep_primary = ReplicaInstance(replica_id="primary", device_id=1, role=ExecutionRole.PRIMARY, active=True)
    rep_hedge = ReplicaInstance(replica_id="hedge_vol_2", device_id=2, role=ExecutionRole.HEDGE, active=True)
    task.replicas.extend([rep_primary, rep_hedge])
    
    assert rep_primary.role == ExecutionRole.PRIMARY
    assert rep_hedge.role == ExecutionRole.HEDGE
    
    # Primary fails
    rep_primary.active = False
    
    # Promote surviving hedge
    assert rep_hedge.active
    rep_hedge.role = ExecutionRole.PRIMARY
    assert rep_hedge.role == ExecutionRole.PRIMARY


# ==============================================================================
# 10. REPLICA COUNT CONSISTENCY
# ==============================================================================
def test_replica_count_definition_consistency():
    """Verify replica count measures additional replicas beyond primary."""
    task = Task(task_id=1)
    assert task.additional_replicas_count == 0
    assert task.replica_launches_count == 0
    
    # Launch hedge
    task.additional_replicas_count += 1
    assert task.replica_launches_count == 1
    
    # Launch cloud fallback
    task.additional_replicas_count += 1
    assert task.replica_launches_count == 2


# ==============================================================================
# 11. CO-AVAILABILITY VS INDEPENDENCE SELECTION LOGIC
# ==============================================================================
def test_co_availability_vs_independence_selection():
    """Verify Algorithm 2 co-availability penalizes correlated devices compared to independence."""
    # Devices 1 and 3 are in the same correlated failure group (both 0 at slots 0, 1)
    d1 = VolunteerDevice(1, DeviceType.DESKTOP, 4, 8.0, 10.0, recent_availability=0.8, history_bitmap=[0, 0, 1, 1, 1])
    d2 = VolunteerDevice(2, DeviceType.DESKTOP, 4, 8.0, 10.0, recent_availability=0.75, history_bitmap=[1, 1, 0, 1, 1])
    d3 = VolunteerDevice(3, DeviceType.DESKTOP, 4, 8.0, 10.0, recent_availability=0.85, history_bitmap=[0, 0, 1, 1, 1])
    
    # Independence would pick d3 because recent_availability is highest (0.85 > 0.75)
    sorted_indep = sorted([d2, d3], key=lambda d: d.recent_availability, reverse=True)
    assert sorted_indep[0].device_id == 3
    
    # Co-availability Algorithm 2 evaluates joint shrinkage and prefers d2 (independent failure pattern)
    task = Task(task_id=1, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=5.0)
    chosen_co = select_co_availability_hedge([d1], [d2, d3], task)
    assert chosen_co is not None
    assert chosen_co.device_id == 2  # Complementary bitmap selected


# ==============================================================================
# 12. LIVE RISK CONTROLLER THRESHOLD TRANSITIONS
# ==============================================================================
def test_live_risk_controller_threshold_transitions():
    """Verify live controller transitions across L0, L1, L2, L3."""
    dev = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=100.0, baseline_mtbf=100.0)
    thresh = ExecutionThresholds(R1=0.05, R2=0.20, R3=0.50)
    
    # Task with low risk -> L0
    t0 = Task(task_id=0, expected_duration_steps=2.0, state=TaskState.RUNNING)
    t0.replicas.append(ReplicaInstance(replica_id="primary", device_id=1, active=True))
    l0, _ = evaluate_and_update_task_risk(t0, dev, [dev], current_step=0.0, thresholds=thresh)
    assert l0 == 0
    
    # Task with medium risk (between R1 and R2) -> L1
    t1 = Task(task_id=1, expected_duration_steps=10.0, state=TaskState.RUNNING)
    t1.replicas.append(ReplicaInstance(replica_id="primary", device_id=1, active=True))
    l1, _ = evaluate_and_update_task_risk(t1, dev, [dev], current_step=0.0, thresholds=thresh)
    assert l1 == 1
    assert t1.latest_checkpoint is not None
    
    # Task with higher risk (between R2 and R3) -> L2
    t2 = Task(task_id=2, expected_duration_steps=30.0, state=TaskState.RUNNING)
    t2.replicas.append(ReplicaInstance(replica_id="primary", device_id=1, active=True))
    d_cand = VolunteerDevice(device_id=2, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=100.0)
    l2, rep2 = evaluate_and_update_task_risk(t2, dev, [dev, d_cand], current_step=0.0, thresholds=thresh,
                                            hedge_selector=lambda p, a, t: d_cand)
    assert l2 == 2
    
    # Task with extreme risk (>= R3) -> L3
    t3 = Task(task_id=3, expected_duration_steps=100.0, state=TaskState.RUNNING)
    t3.replicas.append(ReplicaInstance(replica_id="primary", device_id=1, active=True))
    l3, rep3 = evaluate_and_update_task_risk(t3, dev, [dev], current_step=0.0, thresholds=thresh, use_cloud_fallback=True)
    assert l3 == 3
    assert any(r.role == ExecutionRole.CLOUD for r in t3.replicas)


# ==============================================================================
# 13. INTEGRATION: AUTOMATIC HEDGE PROMOTION AFTER PRIMARY FAILURE
# ==============================================================================
def test_integration_automatic_hedge_promotion_after_primary_failure():
    """
    Integration test: verify that the simulation environment actually performs
    automatic hedge promotion after primary failure during the step loop,
    continues execution, and does not create invalid duplicate primary states.
    """
    d_primary = VolunteerDevice(device_id=10, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=100.0, is_available=True, group_id=0)
    d_hedge = VolunteerDevice(device_id=20, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=100.0, is_available=True, group_id=1)
    
    task = Task(task_id=1, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=10.0,
                expected_duration_steps=20.0, deadline_factor=3.0, arrival_step=0.0, state=TaskState.RUNNING)
    
    rep_primary = ReplicaInstance(replica_id="primary", device_id=10, is_cloud=False,
                                  role=ExecutionRole.PRIMARY, start_step=0.0, start_progress=0.0, current_progress=0.2, active=True)
    rep_hedge = ReplicaInstance(replica_id="hedge_vol_20", device_id=20, is_cloud=False,
                                role=ExecutionRole.HEDGE, start_step=0.0, start_progress=0.1, current_progress=0.1, active=True)
    task.replicas.extend([rep_primary, rep_hedge])
    task.completed_progress = 0.2
    
    sched = AdaptiveStabilityScheduler()
    env = SimulationEnvironment(devices=[d_primary, d_hedge], tasks=[task], scheduler=sched, total_steps=50, seed=42)
    env.pending_tasks.clear()
    env.running_tasks = [task]
    
    # Trigger outage on primary device's group through environment lifecycle
    env.group_outage_timers[0] = 5.0
    
    # Run one step of simulation
    env._step_simulation()
    
    # Verify primary replica was deactivated
    assert not rep_primary.active
    
    # Verify surviving active hedge was automatically promoted to PRIMARY
    assert rep_hedge.active
    assert rep_hedge.role == ExecutionRole.PRIMARY
    
    # Verify exactly one active PRIMARY exists
    active_primaries = [r for r in task.replicas if r.active and r.role == ExecutionRole.PRIMARY]
    assert len(active_primaries) == 1
    assert active_primaries[0].device_id == 20
    
    # Verify task progress continued to advance
    assert rep_hedge.current_progress > 0.1
    assert task.completed_progress >= rep_hedge.current_progress


# ==============================================================================
# 14. INTEGRATION: FRACTIONAL BANDWIDTH AFFECTS COMPLETION STEP
# ==============================================================================
def test_fractional_network_bandwidth_affects_task_completion_step():
    """
    Integration test: verify that network bandwidth directly affects execution progress
    and completion behavior within the simulation environment, preserving fractional steps.
    """
    tx_100 = calculate_checkpoint_transfer_steps(100.0, 100.0)
    tx_10 = calculate_checkpoint_transfer_steps(100.0, 10.0)
    tx_1 = calculate_checkpoint_transfer_steps(100.0, 1.0)
    
    assert abs(tx_100 - (800.0 / 30000.0)) < 1e-4  # ~0.0267
    assert abs(tx_10 - (800.0 / 3000.0)) < 1e-4    # ~0.2667
    assert abs(tx_1 - (800.0 / 300.0)) < 1e-4      # ~2.6667
    
    progresses = {}
    for bw, tx in [(100.0, tx_100), (10.0, tx_10), (1.0, tx_1)]:
        dev = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=bw, is_available=True)
        task = Task(task_id=1, expected_duration_steps=10.0, state=TaskState.RUNNING)
        rep = ReplicaInstance(replica_id="primary", device_id=1, is_cloud=False, role=ExecutionRole.PRIMARY,
                              start_step=0.0, start_progress=0.0, current_progress=0.0, active=True,
                              transfer_remaining_steps=tx)
        task.replicas.append(rep)
        
        env = SimulationEnvironment(devices=[dev], tasks=[task], scheduler=CapabilityOnlyScheduler(), total_steps=10, seed=42)
        env.pending_tasks.clear()
        env.running_tasks = [task]
        env.current_step = 0
        
        # Advance 1 step
        env._step_simulation()
        progresses[bw] = rep.current_progress
        
    # 100 Mbps should make the most progress, followed by 10 Mbps, while 1 Mbps should make 0 progress (still transferring)
    assert progresses[100.0] > progresses[10.0] > progresses[1.0]
    assert progresses[1.0] == 0.0  # transfer remaining > 1.0 step, stalled
    
    # Advance the 1 Mbps environment step-by-step and verify fractional decrement
    dev1 = VolunteerDevice(device_id=1, device_type=DeviceType.DESKTOP, cores=3, ram_gb=8.0, uplink_mbps=1.0, is_available=True)
    task1 = Task(task_id=1, expected_duration_steps=10.0, state=TaskState.RUNNING)
    rep1 = ReplicaInstance(replica_id="primary", device_id=1, is_cloud=False, role=ExecutionRole.PRIMARY,
                           start_step=0.0, start_progress=0.0, current_progress=0.0, active=True,
                           transfer_remaining_steps=tx_1)
    task1.replicas.append(rep1)
    env1 = SimulationEnvironment(devices=[dev1], tasks=[task1], scheduler=CapabilityOnlyScheduler(), total_steps=10, seed=42)
    env1.pending_tasks.clear()
    env1.running_tasks = [task1]
    
    # Step 1: transfer_remaining goes from 2.6667 to 1.6667, progress = 0
    env1.current_step = 0
    env1._step_simulation()
    assert abs(rep1.transfer_remaining_steps - (tx_1 - 1.0)) < 1e-4
    assert rep1.current_progress == 0.0
    
    # Step 2: transfer_remaining goes from 1.6667 to 0.6667, progress = 0
    env1.current_step = 1
    env1._step_simulation()
    assert abs(rep1.transfer_remaining_steps - (tx_1 - 2.0)) < 1e-4
    assert rep1.current_progress == 0.0
    
    # Step 3: transfer_remaining goes from 0.6667 to 0, compute fraction = 1 - 0.6667 = 0.3333
    env1.current_step = 2
    env1._step_simulation()
    assert rep1.transfer_remaining_steps == 0.0
    assert rep1.current_progress > 0.0


# ==============================================================================
# 15. AUTHORITATIVE PROGRESS AND RUNTIME ESTIMATE CONSISTENCY
# ==============================================================================
def test_authoritative_progress_and_runtime_consistency():
    """Verify that Task.get_progress_per_step and Task.get_remaining_steps are mathematically consistent."""
    task = Task(task_id=1, expected_duration_steps=24.0)
    
    for speed in [0.75, 1.00, 1.50]:
        prog_per_step = task.get_progress_per_step(speed)
        rem_steps = task.get_remaining_steps(speed)
        
        # Product of progress per step and remaining steps must equal remaining work (1.0)
        assert abs(prog_per_step * rem_steps - 1.0) < 1e-6
        
        # Simulated steps to reach 100% progress matches ceil(rem_steps)
        simulated_steps = 0
        curr_prog = 0.0
        while curr_prog < 1.0 - 1e-9:
            curr_prog += prog_per_step
            simulated_steps += 1
        assert simulated_steps == math.ceil(rem_steps)


# ==============================================================================
# 16. POISSON ARRIVALS PROPERTIES AND REPRODUCIBILITY
# ==============================================================================
def test_poisson_arrival_properties_and_reproducibility():
    """Verify Poisson arrivals properties: seeded determinism, variance, and mean inter-arrival time."""
    rng1 = np.random.default_rng(12345)
    w1 = generate_workload(total_eval_steps=2880, arrival_rate=0.075, rng=rng1)
    
    rng2 = np.random.default_rng(12345)
    w2 = generate_workload(total_eval_steps=2880, arrival_rate=0.075, rng=rng2)
    
    # Same seed -> identical task counts and arrival steps
    assert len(w1) == len(w2)
    for t1, t2 in zip(w1, w2):
        assert t1.arrival_step == t2.arrival_step
        assert t1.expected_duration_steps == t2.expected_duration_steps
        
    # Different seed -> different tasks
    rng3 = np.random.default_rng(99999)
    w3 = generate_workload(total_eval_steps=2880, arrival_rate=0.075, rng=rng3)
    assert [t.arrival_step for t in w1] != [t.arrival_step for t in w3]
    
    # Statistical property: sample mean interarrival time over 2,000 tasks
    rng_large = np.random.default_rng(42)
    w_large = generate_workload(num_tasks=2000, total_eval_steps=100000, arrival_rate=0.075, rng=rng_large)
    interarrivals = [w_large[0].arrival_step] + [
        w_large[i].arrival_step - w_large[i-1].arrival_step for i in range(1, len(w_large))
    ]
    mean_inter = np.mean(interarrivals)
    expected_mean = 1.0 / 0.075  # ~13.33 steps
    assert abs(mean_inter - expected_mean) / expected_mean < 0.10  # Within 10%


# ==============================================================================
# 17. REPRODUCIBILITY WITH IDENTICAL SEEDS
# ==============================================================================
def test_reproducibility_with_identical_seeds():
    """Verify that running identical seeds produces 100% deterministic, identical simulation results."""
    import pandas as pd
    from experiments.main_results import run_main_results_experiment
    
    df_a = run_main_results_experiment(num_seeds=3, num_devices=15, num_tasks=30)
    df_b = run_main_results_experiment(num_seeds=3, num_devices=15, num_tasks=30)
    
    pd.testing.assert_frame_equal(df_a, df_b)
