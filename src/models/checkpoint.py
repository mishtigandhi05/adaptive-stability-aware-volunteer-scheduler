"""
Checkpoint data structure storing state snapshot for task migration and hedging.
Paper Section 9.
"""

from dataclasses import dataclass
import hashlib
import time


@dataclass
class Checkpoint:
    """
    Simplified checkpoint representation (Section 9):
    - task identifier
    - checkpoint sequence number
    - completed-work indicator (fraction q)
    - application state
    - integrity information
    - timestamp
    """
    task_id: int
    seq_number: int
    completed_fraction: float  # q in [0.0, 1.0]
    app_state: str = "COMPUTATION_PROGRESS_SNAPSHOT"
    integrity_hash: str = ""
    timestamp_step: float = 0.0

    def __post_init__(self):
        if not self.integrity_hash:
            data = f"task-{self.task_id}-seq-{self.seq_number}-q-{self.completed_fraction:.4f}"
            self.integrity_hash = hashlib.sha256(data.encode()).hexdigest()[:12]
