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
from src.scheduling.co_availability import (
    calculate_empirical_joint_failure,
    select_co_availability_hedge,
)


def run_co_availability_experiment(num_seeds: int = 300, num_devices: int = 40, num_tasks: int = 150) -> pd.DataFrame:
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


def run_isolated_correlated_experiment(num_seeds: int = 500, num_devices: int = 28, num_groups: int = 4) -> pd.DataFrame:
    """
    Executes Table 4 experiment: Isolated correlated-placement experiment (Section 16 & Table 4).
    500 independently sampled correlated device configurations. Each with 28 devices and 4 correlated groups.
    Methods: Independence ranking vs Co-availability.
    """
    indep_replicas = []
    indep_joint_fails = []
    indep_exceed_target = []
    
    co_replicas = []
    co_joint_fails = []
    co_exceed_target = []
    
    target_threshold = 0.02  # Target joint interruption risk 2% (Section 16)
    
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
        
        # Method 1: Independence ranking (picks candidates based purely on individual availability score)
        sorted_indep = sorted(candidates, key=lambda d: d.recent_availability, reverse=True)
        # Select up to 3 replicas or until joint risk target is met under independence assumption
        hedge_set_indep = list(primary_set)
        for cand in sorted_indep:
            if len(hedge_set_indep) >= 4:
                break
            hedge_set_indep.append(cand)
            
        f_indep = calculate_empirical_joint_failure(hedge_set_indep)
        
        indep_replicas.append(len(hedge_set_indep) - 1)
        indep_joint_fails.append(f_indep * 100.0)
        indep_exceed_target.append(1.0 if f_indep > target_threshold else 0.0)
        
        # Method 2: Co-availability (Algorithm 2 shrinkage estimator)
        sample_task = generate_workload(1, rng=rng)[0]
        hedge_set_co = list(primary_set)
        for _ in range(3):
            best = select_co_availability_hedge(hedge_set_co, candidates, sample_task)
            if best is not None and best not in hedge_set_co:
                hedge_set_co.append(best)
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
