"""
Single-command CLI script executing all 5 experiment modules and generating all 6 experimental tables.

Usage:
    python run_experiments.py --fast     (Fast sanity check using 5-10 seeds)
    python run_experiments.py --full     (Full Monte Carlo benchmark matching paper seed counts)
"""

import os
import sys
import argparse
import time
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

# Import experiment modules
from experiments.main_results import run_main_results_experiment
from experiments.ablation_study import run_ablation_experiment
from experiments.co_availability_eval import run_co_availability_experiment, run_isolated_correlated_experiment
from experiments.threshold_sensitivity import run_threshold_sensitivity_experiment
from experiments.scaling_eval import run_scaling_experiment


# Published paper target values for comparative reporting
PAPER_PUBLISHED_METRICS = {
    "Table 1": {
        "High Proposed Wasted CPU": "30,546 ± 10,67",
        "High Proposed Deadline": "97.38 ± 0.34%",
        "High Proposed Cloud": "60,561",
        "High Independence Wasted CPU": "41,174 ± 1,739",
        "High Independence Deadline": "97.18 ± 0.42%",
        "High Wasted CPU Reduction": "25.8%"
    },
    "Table 2": {
        "Dispatch Only Wasted CPU": "79,679 ± 4,234",
        "Live Controller Wasted CPU": "33,184 ± 1,152",
        "Live Controller Reduction": "58.4%",
        "Full Lifecycle Wasted CPU": "30,546 ± 1,067"
    },
    "Table 3": {
        "Independence Wasted CPU": "29,945 ± 1,069",
        "Co-availability Wasted CPU": "29,839 ± 1,065",
        "Independence Reliability": "98.71%",
        "Co-availability Reliability": "98.72%"
    },
    "Table 4": {
        "Independence Mean Replicas": "3.99",
        "Independence Joint Interruption": "15.85%",
        "Independence Exceed Target": "73.1%",
        "Co-availability Mean Replicas": "3.52",
        "Co-availability Joint Interruption": "6.93%",
        "Co-availability Exceed Target": "51.1%"
    },
    "Table 5": {
        "(0.10, 0.40) Wasted CPU": "28,251 ± 945",
        "(0.10, 0.40) Deadline": "97.77 ± 0.31%",
        "(0.40, 0.70) Wasted CPU": "30,471 ± 1,059",
        "(0.40, 0.70) Deadline": "97.33 ± 0.34%"
    },
    "Table 6": {
        "10 Devices Deadline": "76.80 ± 1.43%",
        "10 Devices Cloud": "102,701 ± 6,546",
        "50 Devices Deadline": "98.29 ± 0.27%",
        "50 Devices Cloud": "28,249 ± 3,212"
    }
}


def ensure_output_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)


def save_and_print_table(df: pd.DataFrame, table_num: int, title: str, filename: str):
    csv_path = os.path.join("results", f"table{table_num}_{filename}.csv")
    tbl_path = os.path.join("results/tables", f"table{table_num}_{filename}.txt")
    
    df.to_csv(csv_path, index=False)
    
    table_str = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    
    header = f"\n{'=' * 80}\nTable {table_num}: {title}\n{'=' * 80}\n"
    print(header)
    print(table_str)
    print("\n")
    
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(header + table_str + "\n")


def generate_summary_plots(df1: pd.DataFrame, df6: pd.DataFrame):
    """Generate visual figures for paper metrics."""
    try:
        # Figure 1: Wasted CPU across instability conditions (Table 1)
        plt.figure(figsize=(8, 5))
        for method in df1["Method"].unique():
            m_df = df1[df1["Method"] == method]
            plt.plot(m_df["Condition"], m_df["WastedCPU_Mean"], marker='o', label=method)
        plt.title("Wasted CPU Time across Instability Regimes (Table 1)")
        plt.xlabel("Volunteer Instability Condition")
        plt.ylabel("Wasted CPU Units")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("results/figures/figure1_wasted_cpu.png", dpi=300)
        plt.close()

        # Figure 2: Population Scaling Impact on Deadline & Cloud Usage (Table 6)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        
        ax1.plot(df6["Devices"], df6["Deadline_Mean"], 'g-o', label="Deadline Completion (%)")
        ax2.plot(df6["Devices"], df6["Cloud_Mean"], 'b-s', label="Cloud Execution (inst-sec)")
        
        ax1.set_xlabel("Volunteer Population Devices")
        ax1.set_ylabel("Deadline Completion (%)", color='g')
        ax2.set_ylabel("Cloud Execution (inst-sec)", color='b')
        plt.title("Scaling Volunteer Population vs Deadline & Cloud Fallback (Table 6)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("results/figures/figure2_population_scaling.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Plot generation notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run paper simulation experiments.")
    parser.add_argument("--fast", action="store_true", help="Run fast verification with reduced seed count (10 seeds).")
    parser.add_argument("--full", action="store_true", help="Run full benchmark matching paper seeds (300/500 seeds).")
    args = parser.parse_args()

    # Default to fast mode if no flag specified
    fast_mode = not args.full
    
    seeds_main = 10 if fast_mode else 300
    seeds_correlated = 20 if fast_mode else 500
    
    mode_str = "FAST VERIFICATION MODE (10 seeds)" if fast_mode else "FULL PAPER BENCHMARK MODE (300/500 seeds)"
    print(f"\n{'=' * 80}\nAdaptive Stability-Aware Scheduler Experiment Runner\nMode: {mode_str}\nAuthoritative Paper Source: paper/cloud_paper.pdf\n{'=' * 80}\n")
    
    ensure_output_directories()
    start_time = time.time()

    # ----------------------------------------------------
    # Table 1: Main Simulation Results
    # ----------------------------------------------------
    print("Running Experiment 1: Main Simulation Results (Table 1)...")
    df1 = run_main_results_experiment(num_seeds=seeds_main)
    save_and_print_table(df1, 1, "Main simulation results. Values are mean ± 95% CI.", "main_results")

    # ----------------------------------------------------
    # Table 2: Ablation Study
    # ----------------------------------------------------
    print("Running Experiment 2: Ablation Study under High Instability (Table 2)...")
    df2 = run_ablation_experiment(num_seeds=seeds_main)
    save_and_print_table(df2, 2, "Ablation study under high instability.", "ablation")

    # ----------------------------------------------------
    # Table 3: System-Level Co-Availability Evaluation
    # ----------------------------------------------------
    print("Running Experiment 3: System-Level Co-Availability Comparison (Table 3)...")
    df3 = run_co_availability_experiment(num_seeds=seeds_main)
    save_and_print_table(df3, 3, "System-level co-availability comparison.", "coavailability")

    # ----------------------------------------------------
    # Table 4: Isolated Correlated-Placement Experiment
    # ----------------------------------------------------
    print("Running Experiment 4: Correlated-Placement Experiment (Table 4)...")
    df4 = run_isolated_correlated_experiment(num_seeds=seeds_correlated)
    save_and_print_table(df4, 4, "Correlated-placement experiment.", "correlated_placement")

    # ----------------------------------------------------
    # Table 5: Threshold Sensitivity
    # ----------------------------------------------------
    print("Running Experiment 5: Sensitivity to Risk Thresholds (Table 5)...")
    df5 = run_threshold_sensitivity_experiment(num_seeds=seeds_main)
    save_and_print_table(df5, 5, "Sensitivity to risk thresholds.", "threshold_sensitivity")

    # ----------------------------------------------------
    # Table 6: Scaling with Volunteer Population
    # ----------------------------------------------------
    print("Running Experiment 6: Scaling Results with 150 Tasks (Table 6)...")
    df6 = run_scaling_experiment(num_seeds=seeds_main)
    save_and_print_table(df6, 6, "Scaling results with 150 tasks.", "scaling")

    # Generate summary plots
    generate_summary_plots(df1, df6)
    
    elapsed = time.time() - start_time
    print(f"{'=' * 80}\nAll 6 Experimental Tables successfully generated and saved to results/\nTotal execution time: {elapsed:.2f} seconds\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
