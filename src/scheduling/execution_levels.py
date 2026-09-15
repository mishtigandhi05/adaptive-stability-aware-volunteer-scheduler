"""
Execution Levels definition (Paper Section 7).
"""

from enum import IntEnum


class ExecutionLevel(IntEnum):
    L0_NORMAL = 0
    L1_CHECKPOINTED = 1
    L2_VOLUNTEER_HEDGE = 2
    L3_CLOUD_FALLBACK = 3
