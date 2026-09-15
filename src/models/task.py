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


@dataclass
class ReplicaInstance:
    """Represents an active task execution copy (Primary, Volunteer Hedge, or Cloud)."""
    replica_id: str
    device_id: Optional[int]
    is_cloud: bool = False
    start_step: float = 0.0
    start_progress: float = 0.0  # Seeded fraction q_0 from checkpoint
    current_progress: float = 0.0
    active: bool = True
    cpu_time_spent: float = 0.0
    cloud_inst_sec: float = 0.0
    active_duration_steps: float = 0.0  # Tracks discrete 5-min steps active


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
    replica_launches_count: int = 0
    secondary_replica_hours_total: float = 0.0  # Duration-based replica-hours for secondary executions (Section 16, Table 3)

    @property
    def deadline_step(self) -> float:
        return self.arrival_step + (self.expected_duration_steps * self.deadline_factor)

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
        self.checkpoint_storage_mb += 100.0  # 100 MB illustrative checkpoint size
        return ckpt
