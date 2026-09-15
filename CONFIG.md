# Configuration & Reproducibility Specification (`CONFIG.md`)

This document provides the formal mapping between parameters **explicitly specified in the research paper** and **implementation assumptions** required for the simulation.

Authoritative Paper Source: [`paper/cloud_paper.pdf`](paper/cloud_paper.pdf)

---

## 1. Paper-Specified Parameters & Formulations

The following specifications are taken directly from the research paper:

| Specification / Formula | Paper Reference | Value / Formulation in Implementation |
| :--- | :--- | :--- |
| **Simulation Step Size** | Section 13 | 1 discrete step = 5 minutes |
| **History Period** | Section 13 | 4 weeks = 28 days $\times$ 288 slots/day = 8,064 time steps |
| **Evaluation Period** | Section 13 | 1 week = 7 days $\times$ 288 slots/day = 2,016 time steps |
| **Hardware Profiles** | Section 13 | Cores: $\{2, 3, 5\}$, RAM: $\{4, 8, 16\}$ GB, Uplink: $\{1, 10, 100\}$ Mbps |
| **Device Types & States** | Section 13 | Desktop vs Laptop; Operating state: AC vs Battery |
| **Instability Regimes** | Section 13 | `Stable`, `Moderate`, `High` volunteer availability profiles |
| **Workload Size** | Section 13, 16 | 150 independent tasks per simulation run |
| **Task Deadlines** | Section 13 | Deadline factor randomly sampled in range $[1.6, 3.0] \times \text{expected\_time}$ |
| **Monte Carlo Seeds** | Section 13 | 300 random seeds per configuration (Main, Ablation, Thresholds, Scaling) |
| **Isolated Correlated Seeds** | Section 16 (Table 4) | 500 independent random seeds (28 devices, 4 correlated groups) |
| **Confidence Interval** | Section 13 | 95% CI computed as $\bar{x} \pm 1.96 \frac{s}{\sqrt{n}}$ |
| **Device Stability Score ($SS_i$)** | Section 3 | $SS_i = w_a A_i + w_u U_i + w_n N_i + w_h H_i$, where $w_a + w_u + w_n + w_h = 1$ |
| **Exponential Update** | Section 3 | $X_i^{\text{new}} = \alpha X_i^{\text{recent}} + (1 - \alpha) X_i^{\text{history}}$ |
| **State-Adjusted Hazard** | Section 5 | $\lambda_i^{\text{adj}}(t) = \lambda_i(t) \prod_{k \in K_i} m_k$ |
| **Conditional Survival Probability** | Section 4, 5 | $P_i^{\text{adj}}(\tau) = \exp\left(-\int_{a_i}^{a_i+\tau} \lambda_i^{\text{adj}}(u) du\right)$ |
| **Interruption Risk ($R_i$)** | Section 4, 5 | $R_i(\tau) = 1 - P_i^{\text{adj}}(\tau)$ |
| **Task-Device Suitability ($S_{ij}$)** | Section 6 | $S_{ij} = w_c C_{ij} + w_m M_{ij} + w_n N_{ij} + w_s SS_i + w_p P_i(\tau_j) + w_d D_{ij} - w_k K_{ij}$ |
| **Execution Tier Thresholds** | Section 7 | $L_0$ ($R < R_1$), $L_1$ ($R_1 \le R < R_2$), $L_2$ ($R_2 \le R < R_3$), $L_3$ ($R \ge R_3$). Default $R_1=0.05, R_2=0.20, R_3=0.50$ |
| **Co-Availability Shrinkage ($F(R)$)** | Section 10 | $F(R) = \frac{N F_{\text{emp}}(R) + \alpha_s F_{\text{ind}}(R)}{N + \alpha_s}$ |
| **Hedge Selection Metric** | Section 10 | Select candidate $k$ minimizing $F(R \cup \{k\}) + \lambda \text{Cost}_k$ |

---

## 2. Implementation Assumptions (Unspecified Paper Constants)

Where the paper does not specify exact scalar weights, hazard multiplier values, or baseline failure rates, the following implementation assumptions are used:

| Parameter Name | Symbol | Assumed Value | Justification / Description |
| :--- | :--- | :--- | :--- |
| **Stability Weights** | $(w_a, w_u, w_n, w_h)$ | $(0.35, 0.25, 0.15, 0.25)$ | Normalizes stability dimensions to sum to $1.0$ |
| **Smoothing Factor** | $\alpha$ | $0.30$ | Standard exponential moving average weight for recent state |
| **Suitability Weights** | $(w_c, w_m, w_n, w_s, w_p, w_d, w_k)$ | $(0.25, 0.20, 0.15, 0.15, 0.15, 0.10, 0.05)$ | Balanced weighting across compute, memory, network, reliability, deadline, and cost |
| **Battery Hazard Multiplier** | $m_{\text{battery}}$ | $2.50$ | Multiplier when laptop operates on battery vs AC power |
| **Low Battery Threshold & Multiplier** | $m_{\text{low\_battery}}$ | $4.00$ (Threshold: $<20\%$) | **Implementation Assumption**: The paper notes battery state as a hazard factor without specifying a numerical threshold; $20\%$ is used as the low battery threshold |
| **Network Fluctuation Multiplier** | $m_{\text{network\_unstable}}$ | $1.80$ | Accounts for temporary uplink connectivity drops |
| **Shrinkage Weight** | $\alpha_s$ | $50.0$ | Balances sample window size $N$ with empirical joint failure probability |
| **Hedge Cost Penalty** | $\lambda$ | $0.01$ | Trade-off weight for candidate network/transfer cost |
| **Stable Interruption MTBF** | MTBF (Stable) | $600$ steps ($\sim 50$ hrs) | Synthetic baseline mean time between failure steps |
| **Moderate Interruption MTBF** | MTBF (Moderate) | $200$ steps ($\sim 16.6$ hrs) | Synthetic baseline mean time between failure steps |
| **High Interruption MTBF** | MTBF (High) | $75$ steps ($\sim 6.25$ hrs) | Synthetic baseline mean time between failure steps |
| **Poisson Arrival Rate** | $\lambda_{\text{arrival}}$ | $0.075$ tasks/step | Average 150 tasks arriving across 2,016 evaluation steps |
| **Task Workload Range** | $W_{\text{task}}$ | $[12, 48]$ steps | Task compute demand equivalent to 1–4 hours of continuous CPU execution |
| **Checkpoint Size Overhead** | $T_{\text{checkpoint}}$ | $1$ step ($5$ mins) | Transfer overhead required to resume execution from checkpoint |

---

## 3. Independent Result Verification Strategy

The simulation runs independently of any hardcoded target metrics. Result outputs are logged to CSV files in `results/` and formatted in stdout. Any variations between published figures and independent runs are documented in the generated summary reports.
