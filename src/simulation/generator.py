"""
Synthetic workload and heterogeneous population generator.
Paper Section 13 & 16.
Implements paper-specified parameters and documents implementation assumptions.
"""

from typing import List, Tuple, Optional
import numpy as np
from src.config import SimConfig, WorkloadInstabilityParams
from src.models.device import VolunteerDevice, DeviceType, PowerState
from src.models.task import Task


def generate_heterogeneous_population(
    num_devices: int = 30,
    instability: str = "high",
    num_correlated_groups: int = 4,
    history_steps: int = 8064,
    rng: Optional[np.random.Generator] = None
) -> List[VolunteerDevice]:
    """
    Generate heterogeneous desktop and laptop devices (Section 13 & 16).
    - 2, 3, or 5 CPU cores
    - 4, 8, or 16 GB RAM
    - 1, 10, or 100 Mbps uplink
    - desktop or laptop
    - AC or battery operating state
    - Correlated group memberships with shared group outages
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    instability_params = WorkloadInstabilityParams()
    if instability == "stable":
        base_mtbf = instability_params.stable_mtbf_steps
        group_mtbf = base_mtbf * 2.5
    elif instability == "moderate":
        base_mtbf = instability_params.moderate_mtbf_steps
        group_mtbf = base_mtbf * 2.0
    else:
        base_mtbf = instability_params.high_mtbf_steps
        group_mtbf = base_mtbf * 1.5
        
    cores_options = [2, 3, 5]
    ram_options = [4.0, 8.0, 16.0]
    uplink_options = [1.0, 10.0, 100.0]
    
    # 1. Generate shared group availability bitmaps for each correlated group
    group_bitmaps = {}
    for g in range(num_correlated_groups):
        g_bitmap = []
        g_avail = True
        g_counter = int(rng.exponential(group_mtbf))
        for w in range(history_steps):
            if g_counter <= 0:
                g_avail = not g_avail
                scale = group_mtbf if g_avail else instability_params.mean_repair_steps * 1.5
                g_counter = int(rng.exponential(scale))
            else:
                g_counter -= 1
            g_bitmap.append(1 if g_avail else 0)
        group_bitmaps[g] = np.array(g_bitmap, dtype=int)
        
    # 2. Generate individual devices combining group availability and individual failures
    devices = []
    for i in range(num_devices):
        dev_type = DeviceType.LAPTOP if rng.random() < 0.5 else DeviceType.DESKTOP
        cores = rng.choice(cores_options)
        ram = rng.choice(ram_options)
        uplink = rng.choice(uplink_options)
        
        power = PowerState.AC
        bat_level = 1.0
        if dev_type == DeviceType.LAPTOP:
            power = PowerState.BATTERY if rng.random() < 0.4 else PowerState.AC
            bat_level = float(rng.uniform(0.3, 1.0))
            
        group_id = i % num_correlated_groups
        
        # Variations in individual MTBF per device
        dev_mtbf = float(rng.exponential(scale=base_mtbf))
        dev_mtbf = max(15.0, dev_mtbf)
        
        # Individual availability process
        indep_bitmap = []
        is_avail = True
        state_counter = int(rng.exponential(dev_mtbf))
        
        for w in range(history_steps):
            if state_counter <= 0:
                is_avail = not is_avail
                scale = dev_mtbf if is_avail else instability_params.mean_repair_steps
                state_counter = int(rng.exponential(scale))
            else:
                state_counter -= 1
            indep_bitmap.append(1 if is_avail else 0)
            
        # Joint availability bitmap = Group availability AND Individual availability
        indep_arr = np.array(indep_bitmap, dtype=int)
        joint_bitmap = (group_bitmaps[group_id] & indep_arr).tolist()
        
        recent_avail = float(np.mean(joint_bitmap[-2016:])) if len(joint_bitmap) >= 2016 else float(np.mean(joint_bitmap))
        historical_stab = float(np.mean(joint_bitmap))
        
        dev = VolunteerDevice(
            device_id=i,
            device_type=dev_type,
            cores=int(cores),
            ram_gb=float(ram),
            uplink_mbps=float(uplink),
            power_state=power,
            battery_level=bat_level,
            group_id=group_id,
            is_available=True,
            current_session_age=float(rng.uniform(5.0, 100.0)),
            history_bitmap=joint_bitmap,
            recent_availability=recent_avail,
            historical_stability=historical_stab,
            baseline_mtbf=dev_mtbf,
            mean_repair_time=instability_params.mean_repair_steps
        )
        devices.append(dev)
        
    return devices


def generate_workload(
    num_tasks: Optional[int] = None,
    total_eval_steps: int = 2880,
    deadline_factor_range: Tuple[float, float] = (1.6, 3.0),
    arrival_rate: float = 0.075,
    rng: Optional[np.random.Generator] = None
) -> List[Task]:
    """
    Generate independent tasks following a homogeneous Poisson arrival process (Section 13).
    
    Implementation Assumption:
    - Task arrivals follow a Poisson process using exponentially distributed inter-arrival times:
      interarrival ~ Exponential(mean = 1 / arrival_rate)
    - Default arrival_rate = 0.075 tasks/step across total_eval_steps = 2,880 evaluation steps (10 days).
      Expected arrivals: 0.075 * 2880 = 216 tasks.
    - When num_tasks is None (default), the realized number of arrivals varies naturally according to the
      Poisson process over the evaluation horizon [0, total_eval_steps).
    - If num_tasks is explicitly provided, generation stops after reaching num_tasks.
    - Deadlines sampled in [1.6, 3.0] x expected execution duration (Section 13).
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    arrival_times = []
    current_time = 0.0
    mean_interarrival = 1.0 / max(1e-6, float(arrival_rate))
    
    while True:
        interarrival = float(rng.exponential(scale=mean_interarrival))
        current_time += interarrival
        if current_time >= total_eval_steps:
            break
        if num_tasks is not None and len(arrival_times) >= num_tasks:
            break
        arrival_times.append(current_time)
    
    tasks = []
    for j, arr_time in enumerate(arrival_times):
        cores_req = int(rng.choice([2, 3]))
        ram_req = float(rng.choice([4.0, 8.0]))
        uplink_req = float(rng.choice([1.0, 10.0]))
        
        # Execution duration in steps (12 to 48 steps = 1 to 4 hours)
        duration = float(rng.uniform(12.0, 48.0))
        deadline_factor = float(rng.uniform(deadline_factor_range[0], deadline_factor_range[1]))
        
        task = Task(
            task_id=j,
            req_cores=cores_req,
            req_ram_gb=ram_req,
            req_uplink_mbps=uplink_req,
            expected_duration_steps=duration,
            deadline_factor=deadline_factor,
            arrival_step=float(arr_time)
        )
        tasks.append(task)
        
    return tasks
