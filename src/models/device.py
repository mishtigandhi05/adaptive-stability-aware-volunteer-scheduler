"""
Volunteer Device model implementing heterogeneous hardware, AC/Battery power dynamics,
individual interruptions, correlated group availability/outages, and network state fluctuations.
Paper Section 3, 5, 10, & 13.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


class DeviceType(Enum):
    DESKTOP = "desktop"
    LAPTOP = "laptop"


class PowerState(Enum):
    AC = "ac"
    BATTERY = "battery"


@dataclass
class VolunteerDevice:
    """
    Represents a heterogeneous volunteer compute device D_i = (C_i, M_i, N_i, P_i, B_i, H_i).
    Paper Section 3 & 13.
    """
    device_id: int
    device_type: DeviceType
    cores: int            # C_i: 2, 3, or 5 cores
    ram_gb: float         # M_i: 4, 8, or 16 GB
    uplink_mbps: float    # N_i: 1, 10, or 100 Mbps (baseline capability)
    power_state: PowerState = PowerState.AC
    battery_level: float = 1.0  # B_i: 0.0 to 1.0
    group_id: Optional[int] = None  # Correlated outage group identifier (Section 13, Table 4)
    
    # Session, network, and interruption state
    is_available: bool = True
    is_group_outage: bool = False  # Tracked group-level correlated outage
    is_network_fluctuating: bool = False  # Temporary network state dropout/fluctuation (Section 5)
    current_session_age: float = 0.0  # Session age in discrete steps (5-min steps)
    time_until_next_event: float = 0.0
    group_repair_timer: float = 0.0
    network_fluctuation_timer: float = 0.0
    
    # Historical availability time bitmap x_i(w) (Section 10)
    history_bitmap: List[int] = field(default_factory=list)
    
    # Tracked metrics for stability score (Section 3)
    recent_availability: float = 1.0  # A_i
    utilization_factor: float = 0.5   # U_i
    network_suitability: float = 0.8  # N_i
    historical_stability: float = 0.9 # H_i
    stability_score: float = 0.85     # SS_i
    
    # Baseline hazard / MTBF characteristics
    baseline_mtbf: float = 100.0  # In discrete 5-min steps
    mean_repair_time: float = 12.0
    
    def update_power_and_battery(self, step_delta: float = 1.0, rng: Optional[np.random.Generator] = None):
        """Simulate power state transitions, battery discharge, and temporary network fluctuations."""
        if self.device_type == DeviceType.LAPTOP:
            if rng is not None and rng.random() < 0.02:
                # Randomly toggle power source
                self.power_state = PowerState.BATTERY if self.power_state == PowerState.AC else PowerState.AC
            
            if self.power_state == PowerState.BATTERY:
                # Discharge battery (~ 2% per step = ~4 hours total)
                self.battery_level = max(0.0, self.battery_level - 0.02)
                if self.battery_level <= 0.0:
                    # Battery exhausted -> force unavailable
                    self.is_available = False
            else:
                # Recharge on AC power
                self.battery_level = min(1.0, self.battery_level + 0.05)
                
        # Stochastic temporary network state fluctuation (distinct from baseline uplink capacity)
        if self.network_fluctuation_timer > 0:
            self.network_fluctuation_timer -= step_delta
            if self.network_fluctuation_timer <= 0:
                self.is_network_fluctuating = False
        elif rng is not None and rng.random() < 0.03:
            # Trigger temporary network fluctuation lasting 2-6 steps
            self.is_network_fluctuating = True
            self.network_fluctuation_timer = float(rng.integers(2, 7))

    def get_hazard_factors(self) -> List[str]:
        """Return active state factors K_i affecting hazard multiplier (Section 5)."""
        factors = []
        if self.device_type == DeviceType.LAPTOP and self.power_state == PowerState.BATTERY:
            factors.append("battery")
            # Low battery <20% is an implementation assumption for state factor
            if self.battery_level < 0.20:
                factors.append("low_battery")
        if self.is_network_fluctuating or self.uplink_mbps < 5.0:
            factors.append("network_unstable")
        return factors
