"""
Experiment 2: Ablation Study under High Instability (Table 2).
Evaluates cumulative contribution of 8 pipeline components.
Paper Section 15.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.simulation.generator import generate_heterogeneous_population, generate_workload
from src.simulation.environment import SimulationEnvironment
from src.simulation.metrics import MetricCollector
from src.scheduling.schedulers import (
    CapabilityOnlyScheduler,
    CapabilityStabilityScheduler,
    AdaptiveStabilityScheduler,
)


def run_ablation_experiment(num_seeds: int = 300, num_devices: int = 30, num_tasks: Optional[int] = None) -> pd.DataFrame:
    """
    Executes Table 2 experiment:
    Configurations under High instability (8 cumulative stages):
    1. Capability only
    2. + Stability
    3. + Conditional survival
    4. + Dispatch levels
    5. + Live risk controller
    6. + Hazard multipliers
    7. + Co-availability
    8. Full lifecycle
    """
    configs = [
        ("Capability only", CapabilityOnlyScheduler()),
        ("+ Stability", CapabilityStabilityScheduler()),
        ("+ Conditional survival", AdaptiveStabilityScheduler(
            use_conditional_survival=True, use_dispatch_levels=False, use_live_risk=False,
            use_hazard_multipliers=False, use_co_availability=False, use_cloud_fallback=False
        )),
        ("+ Dispatch levels", AdaptiveStabilityScheduler(
            use_conditional_survival=True, use_dispatch_levels=True, use_live_risk=False,
            use_hazard_multipliers=False, use_co_availability=False, use_cloud_fallback=False
        )),
        ("+ Live risk controller", AdaptiveStabilityScheduler(
            use_conditional_survival=True, use_dispatch_levels=True, use_live_risk=True,
            use_hazard_multipliers=False, use_co_availability=False, use_cloud_fallback=False
        )),
        ("+ Hazard multipliers", AdaptiveStabilityScheduler(
            use_conditional_survival=True, use_dispatch_levels=True, use_live_risk=True,
            use_hazard_multipliers=True, use_co_availability=False, use_cloud_fallback=False
        )),
        ("+ Co-availability", AdaptiveStabilityScheduler(
            use_conditional_survival=True, use_dispatch_levels=True, use_live_risk=True,
            use_hazard_multipliers=True, use_co_availability=True, use_cloud_fallback=False
        )),
        ("Full lifecycle", AdaptiveStabilityScheduler(
            use_conditional_survival=True, use_dispatch_levels=True, use_live_risk=True,
            use_hazard_multipliers=True, use_co_availability=True, use_cloud_fallback=True
        )),
    ]
    
    rows = []
    
    for cfg_name, sched in configs:
        collector = MetricCollector()
        
        for seed in range(num_seeds):
            rng = np.random.default_rng(seed + 2000)
            devices = generate_heterogeneous_population(
                num_devices=num_devices,
                instability="high",
                rng=rng
            )
            tasks = generate_workload(
                num_tasks=num_tasks,
                rng=rng
            )
            
            env = SimulationEnvironment(
                devices=devices,
                tasks=tasks,
                scheduler=sched,
                seed=seed
            )
            
            d_pct, wasted_cpu, cloud_sec, ckpt_mb, avg_reps, replica_hours = env.run()
            collector.add_run_result(d_pct, wasted_cpu, cloud_sec, ckpt_mb, avg_reps, replica_hours)
            
        summary = collector.summarize()
        
        rows.append({
            "Configuration": cfg_name,
            "Deadline (%)": f"{summary['deadline_pct'].mean:.2f} ± {summary['deadline_pct'].ci95:.2f}",
            "Wasted CPU": f"{int(round(summary['wasted_cpu'].mean))} ± {int(round(summary['wasted_cpu'].ci95))}",
            "Cloud": f"{int(round(summary['cloud_inst_sec'].mean))}" if summary['cloud_inst_sec'].mean > 0 else "–",
            "Deadline_Mean": summary['deadline_pct'].mean,
            "Deadline_CI95": summary['deadline_pct'].ci95,
            "WastedCPU_Mean": summary['wasted_cpu'].mean,
            "WastedCPU_CI95": summary['wasted_cpu'].ci95,
            "Cloud_Mean": summary['cloud_inst_sec'].mean,
            "Cloud_CI95": summary['cloud_inst_sec'].ci95,
        })
        
    df = pd.DataFrame(rows)
    return df
