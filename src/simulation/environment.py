"""
Discrete-time event simulation environment (5-minute step size).
Paper Section 12 & 13.
Implements shared group outages, dynamic stability, executable dispatch levels, and strict deadline reliability accounting.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from src.config import SimConfig, StabilityWeights
from src.models.device import VolunteerDevice
from src.models.task import Task, TaskState, ReplicaInstance
from src.models.stability import calculate_stability_score, update_exponential_moving_average
from src.scheduling.schedulers import BaseScheduler, IndependenceReplicationScheduler, AdaptiveStabilityScheduler


class SimulationEnvironment:
    """Discrete-time simulator stepping through evaluation period."""
    def __init__(
        self,
        devices: List[VolunteerDevice],
        tasks: List[Task],
        scheduler: BaseScheduler,
        total_steps: int = 2016,
        seed: int = 42,
        num_groups: int = 4
    ):
        self.devices = {d.device_id: d for d in devices}
        self.device_list = devices
        self.tasks = tasks
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.num_groups = num_groups
        self.rng = np.random.default_rng(seed)
        
        self.current_step = 0
        self.pending_tasks = list(tasks)
        self.running_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []

        # Correlated group outage timers
        self.group_outage_timers: Dict[int, float] = {g: 0.0 for g in range(num_groups)}

    def run(self) -> Tuple[float, float, float, float, float, float]:
        """
        Execute simulation step loop from 0 to total_steps.
        Returns:
            (deadline_completion_pct, wasted_cpu_time, cloud_inst_sec, checkpoint_storage, avg_replica_launches, avg_replica_hours)
        """
        for step in range(self.total_steps):
            self.current_step = step
            self._step_simulation()
            
        # Wrap up uncompleted tasks at end of simulation
        total_task_count = len(self.tasks)
        completed_count = len(self.completed_tasks)
        deadline_completion_pct = (completed_count / total_task_count * 100.0) if total_task_count > 0 else 0.0
        
        total_wasted_cpu = sum(t.wasted_cpu_time_total for t in self.tasks)
        total_cloud_sec = sum(t.cloud_inst_sec_total for t in self.tasks)
        total_ckpt_mb = sum(t.checkpoint_storage_mb for t in self.tasks)
        avg_replicas = sum(t.replica_launches_count for t in self.tasks) / max(1, total_task_count)
        avg_replica_hours = sum(t.secondary_replica_hours_total for t in self.tasks) / max(1, total_task_count)
        
        return (deadline_completion_pct, total_wasted_cpu, total_cloud_sec, total_ckpt_mb, avg_replicas, avg_replica_hours)

    def _step_simulation(self):
        # 1. Update device states, power, group outages, and stochastic interruptions
        self._update_devices()
        
        # 2. Update dynamic stability scores SS_i using exponential moving average (Section 3)
        if self.current_step % 12 == 0:  # Every hour (12 steps)
            self._update_dynamic_stability()
        
        # 3. Process task arrivals
        arriving = [t for t in self.pending_tasks if t.arrival_step <= self.current_step]
        for t in arriving:
            self.pending_tasks.remove(t)
            available = [d for d in self.device_list if d.is_available]
            primary = self.scheduler.select_primary(t, available)
            
            if primary is not None:
                t.state = TaskState.RUNNING
                primary_replica = ReplicaInstance(
                    replica_id="primary",
                    device_id=primary.device_id,
                    is_cloud=False,
                    start_step=self.current_step,
                    start_progress=0.0,
                    current_progress=0.0,
                    active=True
                )
                t.replicas.append(primary_replica)
                t.replica_launches_count += 1
                
                # Executable dispatch actions for multi-level tiering at dispatch time (L1 - L3)
                if t.execution_level >= 1:
                    t.add_checkpoint(self.current_step)
                    
                if t.execution_level == 2 and isinstance(self.scheduler, AdaptiveStabilityScheduler):
                    # L2 Dispatch: Start volunteer hedge
                    hedge = self.scheduler.select_hedge([primary], available, t)
                    if hedge is not None:
                        rep = ReplicaInstance(
                            replica_id=f"hedge_vol_{hedge.device_id}",
                            device_id=hedge.device_id,
                            is_cloud=False,
                            start_step=self.current_step,
                            start_progress=0.0,
                            current_progress=0.0,
                            active=True
                        )
                        t.replicas.append(rep)
                        t.replica_launches_count += 1
                elif t.execution_level == 3 and isinstance(self.scheduler, AdaptiveStabilityScheduler):
                    # L3 Dispatch: Start cloud fallback (or volunteer hedge if cloud disabled)
                    if self.scheduler.use_cloud_fallback:
                        cloud_rep = ReplicaInstance(
                            replica_id="cloud_fallback",
                            device_id=None,
                            is_cloud=True,
                            start_step=self.current_step,
                            start_progress=0.0,
                            current_progress=0.0,
                            active=True
                        )
                        t.replicas.append(cloud_rep)
                        t.replica_launches_count += 1
                    else:
                        hedge = self.scheduler.select_hedge([primary], available, t)
                        if hedge is not None:
                            rep = ReplicaInstance(
                                replica_id=f"hedge_vol_{hedge.device_id}",
                                device_id=hedge.device_id,
                                is_cloud=False,
                                start_step=self.current_step,
                                start_progress=0.0,
                                current_progress=0.0,
                                active=True
                            )
                            t.replicas.append(rep)
                            t.replica_launches_count += 1

                # If Independence Replication scheduler, launch static replicas initially
                if isinstance(self.scheduler, IndependenceReplicationScheduler):
                    sec_devices = self.scheduler.select_replicas(t, available, primary)
                    for sec in sec_devices:
                        rep = ReplicaInstance(
                            replica_id=f"hedge_vol_{sec.device_id}",
                            device_id=sec.device_id,
                            is_cloud=False,
                            start_step=self.current_step,
                            start_progress=0.0,
                            current_progress=0.0,
                            active=True
                        )
                        t.replicas.append(rep)
                        t.replica_launches_count += 1
                        
                self.running_tasks.append(t)
            else:
                # Could not schedule -> check deadline
                if self.current_step > t.deadline_step:
                    t.state = TaskState.FAILED_DEADLINE
                    self.failed_tasks.append(t)

        # 4. Step active running tasks and manage full recovery lifecycle
        active_list = list(self.running_tasks)
        for t in active_list:
            # FIRST: Check deadline expiry for running tasks before stepping
            if self.current_step > t.deadline_step:
                t.state = TaskState.FAILED_DEADLINE
                self.running_tasks.remove(t)
                self.failed_tasks.append(t)
                for r in t.replicas:
                    if r.active:
                        r.active = False
                        if not r.is_cloud:
                            t.wasted_cpu_time_total += r.cpu_time_spent
                continue

            # Find active primary or candidate primary for risk evaluation
            active_vol_replicas = [r for r in t.replicas if r.active and not r.is_cloud]
            primary_rep = next((r for r in active_vol_replicas if r.replica_id == "primary"), None)
            if primary_rep is None and active_vol_replicas:
                primary_rep = active_vol_replicas[0]
                
            primary_device = self.devices[primary_rep.device_id] if (primary_rep and primary_rep.device_id is not None) else None
            
            # Re-evaluate live risk and algorithm actions if primary device available
            if primary_device is not None and primary_device.is_available:
                available = [d for d in self.device_list if d.is_available]
                self.scheduler.on_task_step(t, primary_device, available, self.current_step)
            elif not active_vol_replicas and not any(r.active and r.is_cloud for r in t.replicas):
                # All volunteer replicas failed -> Attempt recovery via cloud fallback or new hedge
                available = [d for d in self.device_list if d.is_available]
                if hasattr(self.scheduler, "select_primary"):
                    rec_device = self.scheduler.select_primary(t, available)
                    if rec_device is not None:
                        ckpt_q = t.latest_checkpoint.completed_fraction if t.latest_checkpoint else t.completed_progress
                        rec_rep = ReplicaInstance(
                            replica_id=f"hedge_vol_{rec_device.device_id}",
                            device_id=rec_device.device_id,
                            is_cloud=False,
                            start_step=self.current_step,
                            start_progress=ckpt_q,
                            current_progress=ckpt_q,
                            active=True
                        )
                        t.replicas.append(rec_rep)
                        t.replica_launches_count += 1
                    elif getattr(self.scheduler, "use_cloud_fallback", True):
                        ckpt_q = t.latest_checkpoint.completed_fraction if t.latest_checkpoint else t.completed_progress
                        cloud_rep = ReplicaInstance(
                            replica_id="cloud_fallback",
                            device_id=None,
                            is_cloud=True,
                            start_step=self.current_step,
                            start_progress=ckpt_q,
                            current_progress=ckpt_q,
                            active=True
                        )
                        t.replicas.append(cloud_rep)
                        t.replica_launches_count += 1

            # Advance progress for all active replicas
            task_completed = False
            for r in t.replicas:
                if not r.active:
                    continue
                    
                dev = self.devices[r.device_id] if r.device_id is not None else None
                
                # Check device failure for volunteer replicas
                if not r.is_cloud and (dev is None or not dev.is_available):
                    r.active = False
                    t.wasted_cpu_time_total += r.cpu_time_spent
                    continue
                    
                # Advance replica progress & track duration
                r.active_duration_steps += 1.0
                step_progress = 1.0 / max(1.0, t.expected_duration_steps)
                r.current_progress = min(1.0, r.current_progress + step_progress)
                
                if not r.is_cloud and r.replica_id != "primary":
                    t.secondary_replica_hours_total += (5.0 / 60.0)  # 5 minutes in hours
                    
                if r.is_cloud:
                    r.cloud_inst_sec += 300.0  # 300 seconds per step
                    t.cloud_inst_sec_total += 300.0
                    t.secondary_replica_hours_total += (5.0 / 60.0)
                else:
                    r.cpu_time_spent += 300.0
                    
                t.completed_progress = max(t.completed_progress, r.current_progress)
                
                # Check completion: ONLY count as COMPLETED if current_step <= deadline_step
                if r.current_progress >= 1.0:
                    if self.current_step <= t.deadline_step:
                        t.state = TaskState.COMPLETED
                        t.completion_step = self.current_step
                        self.running_tasks.remove(t)
                        self.completed_tasks.append(t)
                    else:
                        t.state = TaskState.FAILED_DEADLINE
                        self.running_tasks.remove(t)
                        self.failed_tasks.append(t)
                        
                    task_completed = True
                    
                    # Terminate remaining redundant active replicas
                    for other_r in t.replicas:
                        if other_r.active and other_r != r:
                            other_r.active = False
                            if not other_r.is_cloud:
                                t.wasted_cpu_time_total += other_r.cpu_time_spent
                    break

    def _update_devices(self):
        """Update power, battery, correlated group outages, and stochastic interruptions for all devices."""
        # Update group-level outage events
        for g in range(self.num_groups):
            if self.group_outage_timers[g] > 0:
                self.group_outage_timers[g] -= 1.0
            elif self.rng.random() < 0.005:  # ~0.5% chance of group outage per step
                self.group_outage_timers[g] = float(self.rng.exponential(15.0))  # ~1.25 hours duration

        for d in self.device_list:
            d.update_power_and_battery(step_delta=1.0, rng=self.rng)
            
            # Apply group outage state
            is_group_down = (d.group_id is not None and self.group_outage_timers.get(d.group_id, 0.0) > 0)
            d.is_group_outage = is_group_down
            
            if is_group_down:
                d.is_available = False
                continue
                
            # Individual stochastic interruption
            if d.is_available:
                fail_prob = 1.0 / max(10.0, d.baseline_mtbf)
                if self.rng.random() < fail_prob:
                    d.is_available = False
                    d.current_session_age = 0.0
                    d.time_until_next_event = float(self.rng.exponential(d.mean_repair_time))
            else:
                d.time_until_next_event -= 1.0
                if d.time_until_next_event <= 0.0:
                    d.is_available = True
                    d.current_session_age = 0.0

    def _update_dynamic_stability(self):
        """Update device stability scores SS_i dynamically over time (Section 3)."""
        for d in self.device_list:
            recent_obs = 1.0 if d.is_available else 0.0
            d.recent_availability = update_exponential_moving_average(recent_obs, d.recent_availability, alpha=0.30)
            d.stability_score = calculate_stability_score(
                availability=d.recent_availability,
                utilization=d.utilization_factor,
                network=d.network_suitability,
                history_score=d.historical_stability
            )
