"""REINFORCE: the robot tries many episodes and nudges its score weights toward
whatever earned more reward (value recovered, hazards removed)."""
import os
import numpy as np
import config
from brain.env import run_episode
from brain.planner import Planner

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights.npy")


def train(episodes=3000, lr=0.15, gamma=0.97, batch=10, seed=0, log_every=300, verbose=True, init=None):
    rng = np.random.default_rng(seed)
    # by default the robot starts knowing nothing (all weights zero = picks at random)
    w = np.zeros(len(config.FEATURES)) if init is None else np.array(init, dtype=float)
    history = []
    for start in range(0, episodes, batch):
        grad = np.zeros_like(w)
        returns_all, advs, steps = [], [], []
        planner = Planner(w, greedy=False, temperature=0.3)
        for b in range(batch):
            ep_seed = int(rng.integers(0, 10**9))
            m, traj = run_episode(planner, ep_seed, record=True)
            G, rets = 0.0, []
            for F, idx, r in reversed(traj):
                G = r + gamma * G
                rets.append(G)
            rets.reverse()
            steps.append((traj, rets))
            returns_all += rets
            history.append(m["reward"])
        baseline = np.mean(returns_all)
        scale = np.std(returns_all) + 1e-6
        n = 0
        for traj, rets in steps:
            for (F, idx, r), G in zip(traj, rets):
                p = planner.probs(F)
                grad += (F[idx] - p @ F) / planner.temperature * ((G - baseline) / scale)
                n += 1
        w = w + lr * grad / max(n, 1) * 10
        if verbose and (start + batch) % log_every == 0:
            print(f"episode {start + batch:5d}  avg reward (last {log_every}) = "
                  f"{np.mean(history[-log_every:]):6.2f}   weights = {np.round(w, 2)}")
    return w, history


def save_weights(w, path=WEIGHTS_PATH):
    np.save(path, w)


def load_weights(path=WEIGHTS_PATH):
    return np.load(path) if os.path.exists(path) else np.array(config.DEFAULT_WEIGHTS)
