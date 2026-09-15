"""
Task and ReplicaInstance models representing workload demands, deadlines, and multi-tier executions.
Paper Section 3, 7-9, & 16.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
from src.models.checkpoint import Checkpoint


class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED_DEADLINE = "failed_deadline"


class ExecutionRole(Enum):
    """Execution roles cleanly distinguishing primary, volunteer hedge, and cloud fallback."""
    PRIMARY = "primary"
    HEDGE = "hedge"
    CLOUD = "cloud"


def calculate_checkpoint_transfer_steps(
    checkpoint_size_mb: float = 100.0,
    uplink_mbps: float = 10.0,
    step_seconds: float = 300.0
) -> float:
    """
    Calculate fractional checkpoint transfer duration in simulation steps (Implementation Assumption).
    - 100 MB checkpoint = 800 Megabits.
    - Transfer seconds = (size_mb * 8.0) / uplink_mbps
    - Transfer steps = transfer_seconds / step_seconds (where 1 step = 300s = 5 min)
    
    Examples:
    - 100 Mbps: 800 / 100 = 8.0s -> 8.0 / 300 = 0.02667 steps
    - 10 Mbps: 800 / 10 = 80.0s -> 80.0 / 300 = 0.26667 steps
    - 1 Mbps: 800 / 1 = 800.0s -> 800.0 / 300 = 2.66667 steps
    """
    if checkpoint_size_mb <= 0:
        return 0.0
    uplink = max(0.1, float(uplink_mbps))
    duration_sec = (checkpoint_size_mb * 8.0) / uplink
    return float(duration_sec / step_seconds)


@dataclass
class ReplicaInstance:
    """Represents an active task execution copy (Primary, Volunteer Hedge, or Cloud)."""
    replica_id: str
    device_id: Optional[int]
    is_cloud: bool = False
    role: ExecutionRole = ExecutionRole.PRIMARY
    start_step: float = 0.0
    start_progress: float = 0.0  # Seeded fraction q_0 from checkpoint
    current_progress: float = 0.0
    active: bool = True
    cpu_time_spent: float = 0.0
    cloud_inst_sec: float = 0.0
    active_duration_steps: float = 0.0  # Tracks discrete 5-min steps active
    transfer_remaining_steps: float = 0.0  # Fractional simulation steps remaining for checkpoint transfer


@dataclass
class Task:
    """
    Represents a computation task j with requirements, estimated time, deadline, and replicas.
    Paper Section 3.
    """
    task_id: int
    req_cores: int = 2
    req_ram_gb: float = 4.0
    req_uplink_mbps: float = 10.0
    expected_duration_steps: float = 24.0  # Estimated execution steps (24 steps = 2 hours)
    deadline_factor: float = 2.0           # Deadline factor (1.6x to 3.0x expected duration)
    arrival_step: float = 0.0
    
    state: TaskState = TaskState.PENDING
    execution_level: int = 0  # 0: L0, 1: L1, 2: L2, 3: L3
    completed_progress: float = 0.0  # Global max progress fraction q
    latest_checkpoint: Optional[Checkpoint] = None
    
    replicas: List[ReplicaInstance] = field(default_factory=list)
    completion_step: Optional[float] = None
    
    # Cost, storage, and duration metrics
    checkpoint_storage_mb: float = 0.0
    cloud_inst_sec_total: float = 0.0
    wasted_cpu_time_total: float = 0.0
    
    # Consistent replica count definition: "additional replicas/executions beyond the primary"
    additional_replicas_count: int = 0
    secondary_replica_hours_total: float = 0.0  # Duration-based replica-hours for secondary executions (Section 16, Table 3)

    @property
    def replica_launches_count(self) -> int:
        """Alias for additional_replicas_count to maintain compatibility across existing metrics/tests."""
        return self.additional_replicas_count

    @replica_launches_count.setter
    def replica_launches_count(self, value: int):
        self.additional_replicas_count = value

    @property
    def deadline_step(self) -> float:
        return self.arrival_step + (self.expected_duration_steps * self.deadline_factor)

    def get_progress_per_step(self, cpu_speed_factor: float = 1.0) -> float:
        """
        Authoritative calculation of progress per simulation step (Shared Single Source of Truth).
        progress_per_step = cpu_speed_factor / expected_duration_steps
        """
        return float(cpu_speed_factor) / max(1.0, float(self.expected_duration_steps))

    def get_remaining_steps(self, cpu_speed_factor: float = 1.0) -> float:
        """
        Authoritative calculation of remaining steps (Shared Single Source of Truth).
        remaining_steps = remaining_work / cpu_speed_factor
                        = (remaining_fraction * expected_duration_steps) / cpu_speed_factor
        """
        remaining_fraction = max(0.0, 1.0 - self.completed_progress)
        remaining_work = remaining_fraction * float(self.expected_duration_steps)
        return remaining_work / max(0.01, float(cpu_speed_factor))

    def add_checkpoint(self, current_step: float) -> Checkpoint:
        """Create a new checkpoint at current highest completed progress."""
        seq = (self.latest_checkpoint.seq_number + 1) if self.latest_checkpoint else 1
        ckpt = Checkpoint(
            task_id=self.task_id,
            seq_number=seq,
            completed_fraction=self.completed_progress,
            timestamp_step=current_step
        )
        self.latest_checkpoint = ckpt
        self.checkpoint_storage_mb += 100.0  # 100 MB checkpoint size (Implementation Assumption)
        return ckpt
