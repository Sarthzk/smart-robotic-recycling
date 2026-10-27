"""Decides which visible item to pick next.

Modes:
  score  - pick by recovery score (weights can be hand-made or learned)
  colour - ordinary colour-based sorting: colour order, then oldest first (ignores value/hazard)
  fifo   - oldest item first
Replanning (bin full, gripper failure) is handled together with RecyclingEnv:
the env only offers items that still have a free bin, and the planner re-chooses after every pick/failure."""
import numpy as np
import config
from brain.scoring import feature_matrix


class Planner:
    def __init__(self, weights=None, mode="score", greedy=True, temperature=0.3):
        self.w = np.array(config.DEFAULT_WEIGHTS if weights is None else weights, dtype=float)
        self.mode = mode
        self.greedy = greedy              # greedy = always take the best; False = sample (used while learning)
        self.temperature = temperature

    def probs(self, F):
        z = (F @ self.w) / self.temperature
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def choose(self, cands, t, rng):
        F = feature_matrix(cands, t)
        if self.mode == "colour":
            idx = min(range(len(cands)), key=lambda i: (cands[i].color_id, cands[i].arrival))
        elif self.mode == "fifo":
            idx = min(range(len(cands)), key=lambda i: cands[i].arrival)
        elif self.greedy:
            idx = int(np.argmax(F @ self.w))
        else:
            idx = int(rng.choice(len(cands), p=self.probs(F)))
        return idx, F

    def rank(self, items, t):
        """Items sorted best-first with their scores (used by the dashboard / demo labels)."""
        s = feature_matrix(items, t) @ self.w
        order = np.argsort(-s)
        return [(items[i], float(s[i])) for i in order]
