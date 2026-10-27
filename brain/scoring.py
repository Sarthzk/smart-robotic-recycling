"""Recovery score: a weighted sum of what we know about each item."""
import numpy as np
import config


def features(item, t: float) -> np.ndarray:
    """[value, contamination, hazard, cost, urgency] each roughly 0..1."""
    time_left = item.arrival + config.DWELL - t
    urgency = float(np.clip(1.0 - time_left / config.DWELL, 0.0, 1.0))
    return np.array([item.value / 10.0, item.contamination, item.hazard,
                     item.cost / 3.0, urgency])


def feature_matrix(items, t: float) -> np.ndarray:
    return np.stack([features(i, t) for i in items])


def recovery_score(item, weights: np.ndarray, t: float) -> float:
    return float(features(item, t) @ weights)
