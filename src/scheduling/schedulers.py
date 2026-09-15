"""
Encapsulated scheduling strategies for comparative evaluation.
Paper Section 14, 15, & Tables 1-6.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from src.config import ExecutionThresholds, HazardConfig, CoAvailabilityConfig, SuitabilityWeights
from src.models.device import VolunteerDevice
from src.models.task import Task, ReplicaInstance
from src.scheduling.suitability import filter_capable_devices, calculate_suitability_score
from src.scheduling.co_availability import select_co_availability_hedge
from src.risk.live_controller import evaluate_and_update_task_risk
from src.risk.survival import calculate_interruption_risk


class BaseScheduler(ABC):
    """Abstract base class for all schedulers."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_primary(
        self,
        task: Task,
        available_devices: List[VolunteerDevice],
        current_step: Optional[float] = None
    ) -> Optional[VolunteerDevice]:
        """Select primary execution device for task dispatch."""
        pass

    def on_task_step(
        self,
        task: Task,
        primary_device: VolunteerDevice,
        available_devices: List[VolunteerDevice],
        current_step: float
    ):
        """Hook called every discrete time step for active tasks."""
        pass


class CapabilityOnlyScheduler(BaseScheduler):
    """Method 1: Capability Only (Baseline). Uses basic hardware requirements filter."""
    def __init__(self):
        super().__init__("Capability only")

    def select_primary(
        self,
        task: Task,
        available_devices: List[VolunteerDevice],
        current_step: Optional[float] = None
    ) -> Optional[VolunteerDevice]:
        capable = filter_capable_devices(available_devices, task)
        if not capable:
            return None
        # Rank purely by CPU cores and uplink
        return max(capable, key=lambda d: (d.cores, d.uplink_mbps, d.ram_gb))


class CapabilityStabilityScheduler(BaseScheduler):
    """Method 2: Capability + Stability. Combines capability and static stability score SS_i."""
    def __init__(self):
        super().__init__("Capability + Stability")

    def select_primary(
        self,
        task: Task,
        available_devices: List[VolunteerDevice],
        current_step: Optional[float] = None
    ) -> Optional[VolunteerDevice]:
        capable = filter_capable_devices(available_devices, task)
        if not capable:
            return None
        # Rank by combined capability and static stability score SS_i
        return max(capable, key=lambda d: (d.stability_score, d.cores))


class IndependenceReplicationScheduler(BaseScheduler):
    """
    Method 3: Independence Replication.
    Uses static independence-based hedging (replicates task without live risk re-evaluation or co-availability).
    """
    def __init__(self, replication_factor: int = 2):
        super().__init__("Independence replication")
        self.replication_factor = replication_factor

    def select_primary(
        self,
        task: Task,
        available_devices: List[VolunteerDevice],
        current_step: Optional[float] = None
    ) -> Optional[VolunteerDevice]:
        capable = filter_capable_devices(available_devices, task)
        if not capable:
            return None
        # Sort by individual availability score
        sorted_devices = sorted(capable, key=lambda d: d.recent_availability, reverse=True)
        return sorted_devices[0]

    def select_replicas(
        self,
        task: Task,
        available_devices: List[VolunteerDevice],
        primary: VolunteerDevice
    ) -> List[VolunteerDevice]:
        capable = filter_capable_devices(available_devices, task)
        remaining = [d for d in capable if d.device_id != primary.device_id]
        sorted_remaining = sorted(remaining, key=lambda d: d.recent_availability, reverse=True)
        return sorted_remaining[:self.replication_factor - 1]


class AdaptiveStabilityScheduler(BaseScheduler):
    """
    Proposed Method & Cumulative Ablation Scheduler.
    Supports all 8 ablation stages:
    - use_conditional_survival
    - use_dispatch_levels
    - use_live_risk
    - use_hazard_multipliers
    - use_co_availability
    - use_cloud_fallback
    """
    def __init__(
        self,
        thresholds: ExecutionThresholds = ExecutionThresholds(),
        hazard_config: HazardConfig = HazardConfig(),
        co_avail_config: CoAvailabilityConfig = CoAvailabilityConfig(),
        suitability_weights: SuitabilityWeights = SuitabilityWeights(),
        use_conditional_survival: bool = True,
        use_dispatch_levels: bool = True,
        use_live_risk: bool = True,
        use_hazard_multipliers: bool = True,
        use_co_availability: bool = True,
        use_cloud_fallback: bool = True
    ):
        super().__init__("Proposed")
        self.thresholds = thresholds
        self.hazard_config = hazard_config
        self.co_avail_config = co_avail_config
        self.suitability_weights = suitability_weights
        self.use_conditional_survival = use_conditional_survival
        self.use_dispatch_levels = use_dispatch_levels
        self.use_live_risk = use_live_risk
        self.use_hazard_multipliers = use_hazard_multipliers
        self.use_co_availability = use_co_availability
        self.use_cloud_fallback = use_cloud_fallback

    def select_primary(
        self,
        task: Task,
        available_devices: List[VolunteerDevice],
        current_step: Optional[float] = None
    ) -> Optional[VolunteerDevice]:
        capable = filter_capable_devices(available_devices, task)
        if not capable:
            return None
            
        if self.use_conditional_survival:
            # Rank by suitability score S_ij incorporating conditional survival P_i(tau)
            scores = [(d, calculate_suitability_score(d, task, self.suitability_weights, current_step=current_step)) for d in capable]
            scores.sort(key=lambda x: x[1], reverse=True)
            chosen = scores[0][0]
        else:
            # Rank purely by static stability score SS_i
            chosen = max(capable, key=lambda d: d.stability_score)
            
        # If dispatch levels enabled, evaluate risk R_i at dispatch time
        if self.use_dispatch_levels:
            speed_factor = chosen.cpu_speed_factor if hasattr(chosen, "cpu_speed_factor") else 1.0
            remaining_steps = task.get_remaining_steps(speed_factor)
            risk = calculate_interruption_risk(
                chosen,
                remaining_steps,
                self.hazard_config if self.use_hazard_multipliers else HazardConfig(1.0, 1.0, 1.0)
            )
            if risk < self.thresholds.R1:
                task.execution_level = 0
            elif risk < self.thresholds.R2:
                task.execution_level = 1
            elif risk < self.thresholds.R3:
                task.execution_level = 2
            else:
                task.execution_level = 3 if self.use_cloud_fallback else 2
                
        return chosen

    def select_hedge(
        self,
        primary_set: List[VolunteerDevice],
        available_devices: List[VolunteerDevice],
        task: Task,
        current_step: Optional[float] = None
    ) -> Optional[VolunteerDevice]:
        if self.use_co_availability:
            # Algorithm 2 Co-availability selection
            return select_co_availability_hedge(primary_set, available_devices, task, self.co_avail_config)
        else:
            # Rank candidates purely by individual suitability
            capable = filter_capable_devices(available_devices, task)
            primary_ids = {d.device_id for d in primary_set}
            candidates = [d for d in capable if d.device_id not in primary_ids]
            if not candidates:
                return None
            return max(candidates, key=lambda d: calculate_suitability_score(d, task, self.suitability_weights, current_step=current_step))

    def on_task_step(
        self,
        task: Task,
        primary_device: VolunteerDevice,
        available_devices: List[VolunteerDevice],
        current_step: float
    ):
        if not self.use_live_risk:
            return
            
        evaluate_and_update_task_risk(
            task=task,
            primary_device=primary_device,
            available_devices=available_devices,
            current_step=current_step,
            thresholds=self.thresholds,
            hazard_config=self.hazard_config if self.use_hazard_multipliers else HazardConfig(1.0, 1.0, 1.0),
            hedge_selector=lambda p_set, a_devs, t: self.select_hedge(p_set, a_devs, t, current_step=current_step),
            use_cloud_fallback=self.use_cloud_fallback
        )
