"""
Experiment 3 & 4: Co-Availability System-Level & Isolated Evaluations (Tables 3 & 4).
Section 16 & Tables 3 & 4.
"""

import copy
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector
from src.scheduling.schedulers import AdaptiveStabilityScheduler
from src.models.task import Task
from src.scheduling.suitability import filter_capable_devices
from src.scheduling.co_availability import (
    calculate_empirical_joint_failure,
    calculate_independent_joint_failure,
    calculate_shrinkage_joint_failure,
    select_co_availability_hedge,
)


def run_co_availability_experiment(num_seeds: int = 300, num_devices: int = 40, num_tasks: Optional[int] = None) -> pd.DataFrame:
    """
    Executes Table 3 experiment: System-level co-availability comparison (40 devices, 150 tasks).
    Methods: Independence hedge choice vs Co-availability.
    Calculates genuine duration-based replica-hours/task.
    """
    methods = [
        ("Independence hedge choice", AdaptiveStabilityScheduler(use_co_availability=False)),
        ("Co-availability", AdaptiveStabilityScheduler(use_co_availability=True)),
    ]
    
    rows = []
    for name, sched in methods:
        collector = MetricCollector()
        
        for seed in range(num_seeds):
            rng_dev = np.random.default_rng(seed + 3000)
            rng_task = np.random.default_rng(seed + 3000)
            
            devices = generate_heterogeneous_population(
                num_devices=num_devices,
                instability="high",
                rng=rng_dev
            )
            tasks = generate_workload(
                num_tasks=num_tasks,
                rng=rng_task
            )
            
            env = SimulationEnvironment(
                devices=copy.deepcopy(devices),
                tasks=copy.deepcopy(tasks),
                scheduler=sched,
                seed=seed
            )
            
            d_pct, wasted_cpu, cloud_sec, ckpt_mb, avg_reps, rep_hours = env.run()
            collector.add_run_result(d_pct, wasted_cpu, cloud_sec, ckpt_mb, avg_reps, rep_hours)
            
        summary = collector.summarize()
        replica_hours_per_task = summary['replica_hours'].mean  # Aggregate duration-based replica-hours/task
        
        rows.append({
            "Method": name,
            "Wasted CPU": f"{int(round(summary['wasted_cpu'].mean))} ± {int(round(summary['wasted_cpu'].ci95))}",
            "Reliability (%)": f"{summary['deadline_pct'].mean:.2f}",
            "Replica-hours/task": f"{replica_hours_per_task:.3f}",
            "WastedCPU_Mean": summary['wasted_cpu'].mean,
            "WastedCPU_CI95": summary['wasted_cpu'].ci95,
            "Reliability_Mean": summary['deadline_pct'].mean,
            "Replica_Hours": replica_hours_per_task,
        })
        
    df = pd.DataFrame(rows)
    return df


def run_isolated_correlated_experiment(num_seeds: int = 500, num_devices: int = 28, num_groups: int = 4, max_replicas: int = 3) -> pd.DataFrame:
    """
    Executes Table 4 experiment: Isolated correlated-placement experiment (Section 16 & Table 4).
    500 independently sampled correlated device configurations (28 devices, 4 correlated groups).
    Methods: Independence ranking vs Co-availability.
    
    Experimental Protocol & Implementation Assumptions (Documented in CONFIG.md):
    - Both methods start with the same randomly assigned primary device.
    - Both methods evaluate from the same capable candidate pool matching task requirements.
    - Both methods place up to max_replicas hedge replicas (default 3 replicas, yielding up to 4 instances total).
    - Method 1 (Independence ranking): sorts capable candidates by individual availability score (recent_availability)
      and selects top candidates up to max_replicas.
    - Method 2 (Co-availability aware): iteratively evaluates remaining capable candidates using Algorithm 2:
      objective(k) = F_shrink(R u {k}) + lambda * Cost(k), selecting the candidate minimizing the combined objective.
    - The true empirical joint interruption probability F_emp(R) is evaluated using the full historical trace.
    - Runs exceeding target (%) evaluates the fraction of runs where true F_emp(R) > target_threshold (0.02 / 2%).
    """
    indep_replicas = []
    indep_joint_fails = []
    indep_exceed_target = []
    
    co_replicas = []
    co_joint_fails = []
    co_exceed_target = []
    
    target_threshold = 0.02  # Target joint interruption risk 2% (Section 16)
    sample_task = Task(task_id=0, req_cores=2, req_ram_gb=4.0, req_uplink_mbps=1.0)
    
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed + 4000)
        devices = generate_heterogeneous_population(
            num_devices=num_devices,
            instability="high",
            num_correlated_groups=num_groups,
            rng=rng
        )
        
        # Primary set (pick 1 primary device)
        primary = devices[0]
        primary_set = [primary]
        candidates = devices[1:]
        capable_candidates = filter_capable_devices(candidates, sample_task)
        
        # Method 1: Independence ranking
        # Sort capable candidates by individual availability score
        sorted_indep = sorted(capable_candidates, key=lambda d: d.recent_availability, reverse=True)
        hedge_set_indep = list(primary_set) + sorted_indep[:max_replicas]
            
        f_indep = calculate_empirical_joint_failure(hedge_set_indep)
        indep_replicas.append(len(hedge_set_indep) - 1)
        indep_joint_fails.append(f_indep * 100.0)
        indep_exceed_target.append(1.0 if f_indep > target_threshold else 0.0)
        
        # Method 2: Co-availability aware placement (Algorithm 2)
        hedge_set_co = list(primary_set)
        remaining_candidates = list(capable_candidates)
        
        for _ in range(max_replicas):
            best = select_co_availability_hedge(hedge_set_co, remaining_candidates, sample_task)
            if best is not None and best not in hedge_set_co:
                hedge_set_co.append(best)
                remaining_candidates.remove(best)
            else:
                break
                
        f_co = calculate_empirical_joint_failure(hedge_set_co)
        co_replicas.append(len(hedge_set_co) - 1)
        co_joint_fails.append(f_co * 100.0)
        co_exceed_target.append(1.0 if f_co > target_threshold else 0.0)
        
    rows = [
        {
            "Method": "Independence ranking",
            "Mean replicas": f"{np.mean(indep_replicas):.2f}",
            "Joint interruption (%)": f"{np.mean(indep_joint_fails):.2f}",
            "Runs exceeding target (%)": f"{np.mean(indep_exceed_target) * 100.0:.1f}",
        },
        {
            "Method": "Co-availability",
            "Mean replicas": f"{np.mean(co_replicas):.2f}",
            "Joint interruption (%)": f"{np.mean(co_joint_fails):.2f}",
            "Runs exceeding target (%)": f"{np.mean(co_exceed_target) * 100.0:.1f}",
        },
    ]
    
    df = pd.DataFrame(rows)
    return df
