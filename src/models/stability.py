"""
Stability score calculation and exponential moving average updates.
Paper Section 3.
"""

from src.config import StabilityWeights


def update_exponential_moving_average(
    x_recent: float,
    x_history: float,
    alpha: float = 0.30
) -> float:
    """
    Update historical observations using exponential smoothing (Section 3):
    X_i_new = alpha * X_i_recent + (1 - alpha) * X_i_history
    """
    return alpha * x_recent + (1.0 - alpha) * x_history


def calculate_stability_score(
    availability: float,    # A_i
    utilization: float,     # U_i
    network: float,         # N_i
    history_score: float,   # H_i
    weights: StabilityWeights = StabilityWeights()
) -> float:
    """
    Calculate device stability score (Section 3):
    SS_i = wa * A_i + wu * U_i + wn * N_i + wh * H_i
    subject to wa + wu + wn + wh = 1.
    """
    ss = (
        weights.wa * availability +
        weights.wu * utilization +
        weights.wn * network +
        weights.wh * history_score
    )
    return max(0.0, min(1.0, ss))
