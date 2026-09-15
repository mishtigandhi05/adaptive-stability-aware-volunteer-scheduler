"""
Experiment 1: Main Simulation Results (Table 1).
Compares 4 scheduling methods across Stable, Moderate, and High instability conditions.
Paper Section 14 & Table 1.
"""

import copy
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector
from src.scheduling.schedulers import (
    CapabilityOnlyScheduler,
    CapabilityStabilityScheduler,
    IndependenceReplicationScheduler,
    AdaptiveStabilityScheduler,
)


def run_main_results_experiment(num_seeds: int = 300, num_devices: int = 30, num_tasks: Optional[int] = None) -> pd.DataFrame:
    """
    Executes Table 1 experiment:
    Conditions: Stable, Moderate, High
    Methods: Capability only, Capability + Stability, Independence replication, Proposed
    """
    conditions = ["Stable", "Moderate", "High"]
    rows = []
    
    for cond in conditions:
        instability_key = cond.lower()
        
        schedulers = [
            CapabilityOnlyScheduler(),
            CapabilityStabilityScheduler(),
            IndependenceReplicationScheduler(),
            AdaptiveStabilityScheduler(),
        ]
        
        for sched in schedulers:
            collector = MetricCollector()
            
            for seed in range(num_seeds):
                rng_dev = np.random.default_rng(seed + 1000)
                rng_task = np.random.default_rng(seed + 1000)
                
                devices = generate_heterogeneous_population(
                    num_devices=num_devices,
                    instability=instability_key,
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
            
            rows.append({
                "Condition": cond,
                "Method": sched.name,
                "Deadline (%)": f"{summary['deadline_pct'].mean:.2f} ± {summary['deadline_pct'].ci95:.2f}",
                "Wasted CPU": f"{int(round(summary['wasted_cpu'].mean))} ± {int(round(summary['wasted_cpu'].ci95))}",
                "Cloud (inst-sec)": f"{int(round(summary['cloud_inst_sec'].mean))}" if summary['cloud_inst_sec'].mean > 0 else "–",
                "Deadline_Mean": summary['deadline_pct'].mean,
                "Deadline_CI95": summary['deadline_pct'].ci95,
                "WastedCPU_Mean": summary['wasted_cpu'].mean,
                "WastedCPU_CI95": summary['wasted_cpu'].ci95,
                "Cloud_Mean": summary['cloud_inst_sec'].mean,
                "Cloud_CI95": summary['cloud_inst_sec'].ci95,
            })
            
    df = pd.DataFrame(rows)
    return df
