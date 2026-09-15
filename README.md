# Adaptive Stability-Aware Scheduling for Cloud-Orchestrated Volunteer Computing

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

This repository contains a complete, modular Python discrete-time simulation implementing the research paper:

> **"Adaptive Stability-Aware Scheduling for Cloud-Orchestrated Volunteer Computing Using Heterogeneous Desktop and Laptop Resources"**  
> *Mishti Parikshit Gandhi, Alan James, Abhyuday Thapliyal* (Vellore Institute of Technology, Vellore, India).  
> Authoritative Paper PDF: [`paper/cloud_paper.pdf`](paper/cloud_paper.pdf)

---

## 🌟 Key Architecture & Contributions

Traditional volunteer computing schedulers evaluate device reliability statically at task dispatch time. This work introduces an adaptive, live stability-aware scheduling system built around four mechanisms:

1. **Continuous Interruption-Risk Estimation**: Evaluates device failure risk $R_i(\tau)$ for remaining task execution steps $\tau$ rather than historical lifetime alone.
2. **Checkpoint-Seeded Hedging (Algorithm 1)**: When risk crosses thresholds ($R_2 \le R < R_3$), captures execution progress fraction $q$ and starts a secondary execution on an independent volunteer resource from $q$ (avoiding restarting work from scratch).
3. **Co-Availability-Aware Selection (Algorithm 2)**: Evaluates historical joint availability time bitmaps $x_i(w)$ and shrinkage estimator $F(R)$ to prevent selecting hedge resources with correlated failure patterns.
4. **Cloud Execution Fallback**: Escalates to cloud infrastructure ($R \ge R_3$) when volunteer resources cannot provide adequate reliability.

The interruption-risk model additionally incorporates state-aware hazard adjustment for current device conditions (e.g., laptop operating on battery power, battery state, and network fluctuations).

---

## 📐 Mathematical Formulation

### 1. Device Stability Score ($SS_i$)
$$SS_i = w_a A_i + w_u U_i + w_n N_i + w_h H_i \quad \text{subject to } w_a + w_u + w_n + w_h = 1$$
Observations update via exponential smoothing:
$$X_i^{\text{new}} = \alpha X_i^{\text{recent}} + (1 - \alpha) X_i^{\text{history}}$$

### 2. State-Adjusted Hazard & Survival Probability
$$\lambda_i^{\text{adj}}(t) = \lambda_i(t) \prod_{k \in K_i} m_k$$
$$P_i^{\text{adj}}(\tau) = \exp\left( - \int_{a_i}^{a_i + \tau} \lambda_i^{\text{adj}}(u) du \right)$$
$$\text{Interruption Risk } R_i(\tau) = 1 - P_i^{\text{adj}}(\tau)$$

### 3. Task-Device Suitability Score ($S_{ij}$)
$$S_{ij} = w_c C_{ij} + w_m M_{ij} + w_n N_{ij} + w_s SS_i + w_p P_i(\tau_j) + w_d D_{ij} - w_k K_{ij}$$

### 4. Co-Availability Shrinkage Estimator ($F(R)$)
$$F_{\text{emp}}(R) = \frac{1}{N} \sum_{w=1}^N \prod_{i \in R} (1 - x_i(w)), \quad F_{\text{ind}}(R) = \prod_{i \in R} (1 - p_i)$$
$$F(R) = \frac{N F_{\text{emp}}(R) + \alpha_s F_{\text{ind}}(R)}{N + \alpha_s}$$
$$\text{Objective}(k) = F(R \cup \{k\}) + \lambda \text{Cost}_k$$

---

## 📁 Repository Structure

```
.
├── paper/
│   └── cloud_paper.pdf             # Authoritative research paper document
├── CONFIG.md                       # Explicit mapping of paper parameters vs implementation assumptions
├── requirements.txt                # Dependencies
├── README.md                       # Main documentation
├── run_experiments.py              # Single-command CLI runner for all 6 tables
├── src/
│   ├── config.py                   # Centralized parameter definitions
│   ├── models/                     # Device, Task, Checkpoint, and Stability models
│   ├── risk/                       # Hazard adjustments, Survival model, & Live Risk Controller (Alg 1)
│   ├── scheduling/                 # Suitability, Co-Availability (Alg 2), & 4 Schedulers
│   └── simulation/                 # Discrete-time simulator, Generator, & 95% CI Metrics
├── experiments/
│   ├── main_results.py             # Table 1: Main Simulation Results
│   ├── ablation_study.py           # Table 2: 8-Stage Ablation Study
│   ├── co_availability_eval.py     # Table 3 & 4: Co-Availability Evaluations
│   ├── threshold_sensitivity.py    # Table 5: Threshold Sensitivity (R2, R3)
│   └── scaling_eval.py             # Table 6: Population Scaling (10-50 devices)
├── results/                        # Generated output tables and figures
│   ├── table1_main_results.csv
│   ├── table2_ablation.csv
│   ├── table3_coavailability.csv
│   ├── table4_correlated_placement.csv
│   ├── table5_threshold_sensitivity.csv
│   ├── table6_scaling.csv
│   ├── tables/                     # Formatted Markdown/Text output tables
│   └── figures/                    # Generated metric visualization plots
└── tests/                          # Complete pytest suite
    ├── test_models.py
    ├── test_risk_controller.py
    ├── test_co_availability.py
    └── test_simulation.py
```

---

## ⚙️ Reproducibility & Parameter Attribution

As detailed in [`CONFIG.md`](CONFIG.md), this repository implements the experimental configurations described in the paper and independently evaluates them. Every formula, algorithm, execution level, and parameter explicitly specified in the paper is implemented as specified; unspecified parameters are documented as implementation assumptions in [`CONFIG.md`](CONFIG.md).

> **Reproducibility note**: The paper does not specify every stochastic simulation parameter. Therefore, this repository does not hardcode the reported results. Parameters explicitly stated in the paper are implemented as specified; unspecified parameters are documented as implementation assumptions in `CONFIG.md`. Results generated by this repository should therefore be interpreted as an independent implementation of the described methodology, not as a claim of exact numerical reproduction.

---

## 🚀 Quick Start & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run All Experiments (Single Command)

**Fast Verification Mode** (10 seeds; execution time depends on hardware/environment):
```bash
python run_experiments.py --fast
```

**Full Paper Benchmark Mode** (300/500 Monte Carlo seeds):
```bash
python run_experiments.py --full
```

Generated outputs will be saved in `results/` (`table1_main_results.csv` through `table6_scaling.csv`).

---

## 📊 Summary of Experimental Tables

- **Table 1: Main Simulation Results**: Compares `Capability only`, `Capability + Stability`, `Independence replication`, and `Proposed` across Stable, Moderate, and High instability regimes.
- **Table 2: Ablation Study**: Tracks cumulative impact across 8 pipeline stages under High instability.
- **Table 3: System-Level Co-Availability**: Evaluates independence hedge choice vs co-availability on 40 devices.
- **Table 4: Correlated Placement Experiment**: Evaluates joint failure rates across 500 seeds with 4 correlated groups.
- **Table 5: Threshold Sensitivity**: Evaluates $(R_2, R_3)$ threshold variations with $R_1=0.05$.
- **Table 6: Population Scaling**: Evaluates scaling from 10 to 50 volunteer devices.

---

## 📜 License
MIT License.
