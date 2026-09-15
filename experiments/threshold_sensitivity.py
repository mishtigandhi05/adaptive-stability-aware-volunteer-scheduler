"""
Experiment 5: Threshold Sensitivity (Table 5).
Varies thresholds R2 and R3 while keeping R1 = 0.05.
Paper Section 17 & Table 5.
"""

import copy
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.config import ExecutionThresholds
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector
from src.scheduling.schedulers import AdaptiveStabilityScheduler


def run_threshold_sensitivity_experiment(num_seeds: int = 300, num_devices: int = 30, num_tasks: Optional[int] = None) -> pd.DataFrame:
    """
    Executes Table 5 experiment:
    Threshold variations (R2, R3) keeping R1 = 0.05:
    - (0.10, 0.40)
    - (0.20, 0.50)
    - (0.30, 0.60)
    - (0.40, 0.70)
    """
    threshold_pairs = [
        (0.10, 0.40),
        (0.20, 0.50),
        (0.30, 0.60),
        (0.40, 0.70),
    ]
    
    rows = []
    
    for r2, r3 in threshold_pairs:
        thresh = ExecutionThresholds(R1=0.05, R2=r2, R3=r3)
        sched = AdaptiveStabilityScheduler(thresholds=thresh)
        
        collector = MetricCollector()
        
        for seed in range(num_seeds):
            rng_dev = np.random.default_rng(seed + 5000)
            rng_task = np.random.default_rng(seed + 5000)
            
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
        
        rows.append({
            "R2": f"{r2:.2f}",
            "R3": f"{r3:.2f}",
            "Wasted CPU": f"{int(round(summary['wasted_cpu'].mean))} ± {int(round(summary['wasted_cpu'].ci95))}",
            "Deadline (%)": f"{summary['deadline_pct'].mean:.2f} ± {summary['deadline_pct'].ci95:.2f}",
            "WastedCPU_Mean": summary['wasted_cpu'].mean,
            "WastedCPU_CI95": summary['wasted_cpu'].ci95,
            "Deadline_Mean": summary['deadline_pct'].mean,
            "Deadline_CI95": summary['deadline_pct'].ci95,
        })
        
    df = pd.DataFrame(rows)
    return df
