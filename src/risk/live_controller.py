"""
Live Risk Controller implementing Algorithm 1 from Section 8 & 9 of the paper.
Continuously re-evaluates active task risk during execution and manages level transitions (L0-L3).
L3 genuinely represents Cloud Fallback when risk R >= R3.
"""

from typing import List, Optional, Tuple, Dict
from src.config import ExecutionThresholds, HazardConfig
from src.models.device import VolunteerDevice
from src.models.task import Task, TaskState, ReplicaInstance
from src.models.checkpoint import Checkpoint
from src.risk.survival import calculate_interruption_risk


def evaluate_and_update_task_risk(
    task: Task,
    primary_device: VolunteerDevice,
    available_devices: List[VolunteerDevice],
    current_step: float,
    thresholds: ExecutionThresholds = ExecutionThresholds(),
    hazard_config: HazardConfig = HazardConfig(),
    hedge_selector=None,
    use_cloud_fallback: bool = True
) -> Tuple[int, Optional[ReplicaInstance]]:
    """
    Implements Algorithm 1: Live Risk Control (Paper Section 8 & Algorithm 1).
    
    Returns:
        (new_level, new_replica_instance_if_launched)
    """
    if task.state != TaskState.RUNNING:
        return task.execution_level, None
        
    # 1. Observe current device state & estimate remaining execution time tau
    remaining_fraction = max(0.0, 1.0 - task.completed_progress)
    remaining_steps = remaining_fraction * task.expected_duration_steps
    
    # 2. Compute R_i(tau)
    risk = calculate_interruption_risk(primary_device, remaining_steps, hazard_config)
    
    new_replica = None
    prev_level = task.execution_level
    
    # Helper: Check if new checkpoint should be created (avoid duplicate checkpoints on every step)
    def should_checkpoint() -> bool:
        if task.latest_checkpoint is None:
            return True
        progress_delta = task.completed_progress - task.latest_checkpoint.completed_fraction
        return progress_delta >= 0.15  # At least 15% progress advance since last checkpoint
    
    # 3. Algorithm 1 Threshold Comparison
    if risk < thresholds.R1:
        # L0 - Normal execution
        task.execution_level = 0
    elif risk < thresholds.R2:
        # L1 - Checkpointed execution
        task.execution_level = 1
        if prev_level < 1 or should_checkpoint():
            task.add_checkpoint(current_step)
    elif risk < thresholds.R3:
        # L2 - Checkpoint-seeded volunteer hedging
        task.execution_level = 2
        ckpt = task.latest_checkpoint
        if prev_level < 2 or should_checkpoint():
            ckpt = task.add_checkpoint(current_step)
            
        # Check if volunteer hedge is already active
        has_volunteer_hedge = any(
            r.active and not r.is_cloud and r.replica_id.startswith("hedge_vol")
            for r in task.replicas
        )
        if not has_volunteer_hedge and hedge_selector is not None:
            # Select independent hedge device (Algorithm 2)
            candidate = hedge_selector([primary_device], available_devices, task)
            if candidate is not None:
                start_q = ckpt.completed_fraction if ckpt is not None else task.completed_progress
                new_replica = ReplicaInstance(
                    replica_id=f"hedge_vol_{candidate.device_id}",
                    device_id=candidate.device_id,
                    is_cloud=False,
                    start_step=current_step,
                    start_progress=start_q,
                    current_progress=start_q,
                    active=True
                )
                task.replicas.append(new_replica)
                task.replica_launches_count += 1
    else:
        # L3 - Cloud fallback when volunteer protection is insufficient (R >= R3)
        task.execution_level = 3
        ckpt = task.latest_checkpoint
        if prev_level < 3 or should_checkpoint():
            ckpt = task.add_checkpoint(current_step)
            
        has_cloud_replica = any(r.active and r.is_cloud for r in task.replicas)
        
        if not has_cloud_replica:
            start_q = ckpt.completed_fraction if ckpt is not None else task.completed_progress
            if use_cloud_fallback:
                # Directly initiate cloud fallback execution from checkpoint
                new_replica = ReplicaInstance(
                    replica_id="cloud_fallback",
                    device_id=None,
                    is_cloud=True,
                    start_step=current_step,
                    start_progress=start_q,
                    current_progress=start_q,
                    active=True
                )
                task.replicas.append(new_replica)
                task.replica_launches_count += 1
            else:
                # Cloud fallback disabled -> fallback to volunteer hedge if possible
                has_volunteer_hedge = any(r.active and not r.is_cloud and r.replica_id != "primary" for r in task.replicas)
                if not has_volunteer_hedge and hedge_selector is not None:
                    candidate = hedge_selector([primary_device], available_devices, task)
                    if candidate is not None:
                        new_replica = ReplicaInstance(
                            replica_id=f"hedge_vol_{candidate.device_id}",
                            device_id=candidate.device_id,
                            is_cloud=False,
                            start_step=current_step,
                            start_progress=start_q,
                            current_progress=start_q,
                            active=True
                        )
                        task.replicas.append(new_replica)
                        task.replica_launches_count += 1
            
    return task.execution_level, new_replica


class LiveRiskController:
    """Wrapper class for managing live risk evaluations across active tasks."""
    def __init__(
        self,
        thresholds: ExecutionThresholds = ExecutionThresholds(),
        hazard_config: HazardConfig = HazardConfig(),
        hedge_selector=None,
        use_cloud_fallback: bool = True
    ):
        self.thresholds = thresholds
        self.hazard_config = hazard_config
        self.hedge_selector = hedge_selector
        self.use_cloud_fallback = use_cloud_fallback

    def process_task(
        self,
        task: Task,
        primary_device: VolunteerDevice,
        available_devices: List[VolunteerDevice],
        current_step: float
    ) -> Tuple[int, Optional[ReplicaInstance]]:
        return evaluate_and_update_task_risk(
            task=task,
            primary_device=primary_device,
            available_devices=available_devices,
            current_step=current_step,
            thresholds=self.thresholds,
            hazard_config=self.hazard_config,
            hedge_selector=self.hedge_selector,
            use_cloud_fallback=self.use_cloud_fallback
        )
