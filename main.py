"""Smart Robotic Recycling System - command line entry point.

  python main.py train     learn item priorities (about 10 seconds)
  python main.py compare   colour sorting vs fixed score vs learned (graphs + csv)
  python main.py demo      3D PyBullet robot demo (add --headless to run without a window)
"""
import argparse
import os
import numpy as np
import config

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def cmd_train(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from brain.learner import train, save_weights
    os.makedirs(RESULTS, exist_ok=True)
    w, hist = train(episodes=args.episodes, seed=args.seed)
    save_weights(w)
    print("\nLearned weights:")
    for name, val in zip(config.FEATURES, w):
        print(f"  {name:14s} {val:6.2f}")
    k = max(1, args.episodes // 60)
    smooth = np.convolve(hist, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(len(smooth)) + k, smooth, color="#1b7f5c", lw=2)
    plt.xlabel("Training episode")
    plt.ylabel("Reward per episode (smoothed)")
    plt.title("The robot learns which waste to recover first")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "learning_curve.png"), dpi=150)
    print("Saved weights.npy and results/learning_curve.png")


def cmd_compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from brain.env import run_episode
    from brain.planner import Planner
    from brain.learner import load_weights
    os.makedirs(RESULTS, exist_ok=True)
    seeds = range(50_000, 50_000 + args.episodes)       # unseen episodes, same for every method
    methods = {
        "Colour sorting": Planner(mode="colour"),
        "Fixed score": Planner(),
        "Learned score": Planner(load_weights()),
    }
    rows = []
    for name, planner in methods.items():
        ms = [run_episode(planner, s) for s in seeds]
        rows.append(dict(method=name,
                         value_recovered=np.mean([m["value"] for m in ms]),
                         recovery_rate_pct=np.mean([m["recovery_rate"] for m in ms]),
                         hazards_caught=np.mean([m["hazards_caught"] for m in ms]),
                         hazards_escaped=np.mean([m["hazards_escaped"] for m in ms]),
                         items_sorted=np.mean([m["sorted"] for m in ms])))
    df = pd.DataFrame(rows).round(2)
    df.to_csv(os.path.join(RESULTS, "comparison.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#9aa5b1", "#e0a03a", "#1b7f5c"]
    for a, col, title in zip(ax, ["recovery_rate_pct", "value_recovered", "hazards_escaped"],
                             ["Recovery rate (%)", "Value recovered", "Hazardous items missed"]):
        a.bar(df["method"], df[col], color=colors)
        a.set_title(title)
        a.tick_params(axis="x", labelrotation=15)
        for i, v in enumerate(df[col]):
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "comparison.png"), dpi=150)
    print("Saved results/comparison.csv and results/comparison.png")


def cmd_demo(args):
    from sim.demo import run_demo
    run_demo(headless=args.headless, seed=args.seed, max_picks=args.max_picks,
             screenshot_dir=args.screenshots, speed=args.speed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--episodes", type=int, default=3000); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("compare"); p.add_argument("--episodes", type=int, default=200)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("demo"); p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=7); p.add_argument("--max-picks", type=int, default=40)
    p.add_argument("--screenshots", default=None, help="folder to save camera screenshots")
    p.add_argument("--speed", type=float, default=3.0, help="animation speed, higher = faster (default 3; try 6 if slow)")
    p.set_defaults(fn=cmd_demo)
    args = ap.parse_args()
    args.fn(args)
