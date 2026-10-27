"""3D demo: the learned planner drives the KUKA arm to sort waste. Writes live metrics for the dashboard."""
import json
import os
import time
import numpy as np
import pybullet as p
import config
from brain.env import RecyclingEnv
from brain.planner import Planner
from brain.learner import load_weights
from sim.world import World, BELT_Y, BELT_TOP, BIN_X, BIN_Y
from sim.robot import Robot, HOME_TIP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(ROOT, "logs", "live_metrics.json")
FRAMES = 150                    # animation frames per pick at --speed 1
SUBSTEPS = 4                    # physics steps per frame (240 Hz physics -> 60 fps)

# to make the demo show replanning, the metal and plastic bins are tiny and the gripper breaks on two picks
DEMO_CAPACITY = {**config.BIN_CAPACITY, "metal": 2, "plastic": 2}
DEMO_FAIL_ATTEMPTS = [6, 7]


def smooth(a):
    return a * a * (3 - 2 * a)


class Demo:
    def __init__(self, headless, seed, screenshot_dir=None, speed=3.0):
        self.headless = headless
        self.speed = max(0.25, float(speed))
        self.frames = max(30, int(FRAMES / self.speed))          # animation frames per pick
        self.n_frames, self.t_start = 0, time.perf_counter()
        self._last = time.perf_counter()
        self.env = RecyclingEnv(seed=seed, capacity=DEMO_CAPACITY, forced_fail_attempts=DEMO_FAIL_ATTEMPTS)
        self.planner = Planner(load_weights())
        self.world = World(self.env.items, headless=headless)
        self.robot = Robot()
        self.vt = self.env.t
        self.picked = set()
        self.shots = screenshot_dir
        self.banner = "Waiting for waste..."
        self.shot_no = 0
        if self.shots:
            os.makedirs(self.shots, exist_ok=True)
        self.robot.settle(HOME_TIP, 300)

    @property
    def banner(self):
        return self._banner

    @banner.setter
    def banner(self, text):
        self._banner = text
        if hasattr(self, "world"):
            self.world.set_status(text)          # shown in the 3D window (no-op when headless)

    # --------------------------------------------------------------- one animation frame
    def frame(self, vt, tip, held=None):
        self.vt = vt
        self.robot.command(tip)
        for _ in range(SUBSTEPS):
            p.stepSimulation()
        pos = self.robot.update_tool_visual()
        if held is not None:
            self.world.place_item(held, (pos - [0, 0, self.world.shape_half[held.uid]]).tolist())
        self.world.update_belt(vt, self.picked)
        self.n_frames += 1
        if not self.headless:                    # pace to 60 fps but never wait longer than needed
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - self._last)))
            self._last = time.perf_counter()

    def screenshot(self, name):
        if not self.shots:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = self.world.screenshot()
        m = self.env.metrics()
        fig = plt.figure(figsize=(9.6, 6.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img); ax.axis("off")
        txt = (f"{self.banner}\nsorted {m['sorted']}   recovered value {m['value']:.1f}   "
               f"recovery {m['recovery_rate']:.0f}%   hazards caught {m['hazards_caught']}   "
               f"escaped {m['hazards_escaped']}   tool: {m['tool']}")
        fig.text(0.01, 0.01, txt, color="white", fontsize=9, va="bottom",
                 bbox=dict(facecolor="black", alpha=0.75, pad=4))
        fig.savefig(os.path.join(self.shots, f"{self.shot_no:02d}_{name}.png"))
        plt.close(fig)
        self.shot_no += 1

    # --------------------------------------------------------------- one pick
    def do_pick(self, item, info, ranking):
        t0, t1 = info["t0"], info["t1"]
        failed = info["outcome"] == "failed"
        tool = info["tool"] if not failed else "fail"
        n_g = int(self.frames * 0.45)                      # grasp happens here
        start = self.robot.tip_position()
        gx = float(np.clip(self.world.item_pos_on_belt(item, t0 + (t1 - t0) * 0.45)[0], -0.42, 0.42))
        above = np.array([gx, BELT_Y, 0.22])
        grasp = np.array([gx, BELT_Y, BELT_TOP + self.world.shape_half[item.uid] + 0.02])
        for k in range(n_g):                          # approach: hover above, then descend onto the item
            a = smooth((k + 1) / n_g)
            tip = start + (above - start) * min(1, a * 2) if a < 0.5 else above + (grasp - above) * (a - 0.5) * 2
            self.frame(t0 + (t1 - t0) * k / self.frames, tip)
        self.robot.set_grip(True)                    # fingers close on the item (or on nothing, if the grasp fails)
        if failed:
            self.robot.set_tool_color("fail")
            self.banner = f"GRIPPER FAILED on {item.kind}  ->  replanning"
            if info["tool"] == "suction":
                self.banner += "  (switching to backup suction tool)"
            self.screenshot("gripper_failure")
        else:
            self.picked.add(item.uid)
            self.robot.set_tool_color("suction" if info["tool"] == "suction" else "gripper")
            self.banner = (f"Picking #1 of {len(ranking)}: {item.kind}  value {item.recovered_value:.1f}  hazard {item.hazard:.1f}")
            self.screenshot("pick")
        dest = info["dest"]
        lift = np.array([gx, BELT_Y, 0.25])
        if failed:
            target_end = HOME_TIP
        else:
            bx = BIN_X[dest]
            drop = np.array([bx, BIN_Y, 0.20])
            mid = np.array([0.0, 0.0, 0.42])
        if failed:
            self.robot.set_grip(False)               # give up and open the gripper again
        n_b = self.frames - n_g
        cur = self.robot.tip_position()
        for k in range(n_b):
            a = smooth((k + 1) / n_b)
            if failed:
                tip = cur + (HOME_TIP - cur) * a
            elif a < 0.35:
                tip = cur + (lift - cur) * (a / 0.35)
            elif a < 0.65:
                tip = lift + (mid - lift) * ((a - 0.35) / 0.3)
            else:
                tip = mid + (drop - mid) * ((a - 0.65) / 0.35)
            self.frame(t0 + (t1 - t0) * (n_g + k) / self.frames, tip, held=None if failed else item)
        self.robot.set_grip(False)                   # release over the bin
        if not failed:
            self.world.drop_in_bin(item, dest)
            if dest == "overflow":
                self.banner = f"{item.bin.upper()} bin FULL -> {item.kind} sent to OVERFLOW"
                self.screenshot("bin_full_overflow")
            self.robot.set_tool_color("gripper")
        self.robot.set_tool_color("gripper" if info["tool"] == "gripper" else "suction")
        self.vt = t1

    def idle_until(self, t_next):
        t0 = self.vt
        n = int(np.clip((t_next - t0) * 30 / self.speed, 6, 120))
        cur = self.robot.tip_position()
        for k in range(n):
            a = smooth((k + 1) / n)
            self.frame(t0 + (t_next - t0) * (k + 1) / n, cur + (HOME_TIP - cur) * a)
        self.vt = t_next

    # --------------------------------------------------------------- metrics
    def write_metrics(self, ranking, finished=False):
        self.world.update_bin_counters(self.env.bins, self.env.capacity)
        m = self.env.metrics()
        m.update(time=round(self.env.t, 1), finished=finished, banner=self.banner,
                 events=self.env.log[-14:],
                 capacity=self.env.capacity,
                 ranking=[dict(item=f"{i.kind} #{i.uid}", score=round(s, 2)) for i, s in ranking[:5]],
                 stamp=time.time())
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        tmp = METRICS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f)
        os.replace(tmp, METRICS_PATH)

    # --------------------------------------------------------------- main loop
    def run(self, max_picks):
        env, picks = self.env, 0
        self.write_metrics([])
        self.screenshot("start")
        while not env.done and picks < max_picks:
            cands = env.candidates()
            if not cands:
                self.world.label_rank = {}
                nxt = [i.arrival for i in env.pending if i.arrival > env.t]
                self.idle_until(min(nxt) if nxt else env.t + 1.0)
                env.advance_idle()
                continue
            ranking = self.planner.rank(cands, env.t)
            self.world.label_rank = {i.uid: r + 1 for r, (i, _) in enumerate(ranking[:3])}   # label the top 3 only
            idx, _ = self.planner.choose(cands, env.t, np.random.default_rng(0))
            item = cands[idx]
            self.write_metrics(ranking)
            info = env.step(item)
            self.do_pick(item, info, ranking)
            picks += 1
            self.write_metrics(self.planner.rank(env.candidates(), env.t) if env.candidates() else [])
        self.banner = "Run finished"
        self.screenshot("finish")
        self.write_metrics([], finished=True)
        return env.metrics()


def run_demo(headless=False, seed=7, max_picks=40, screenshot_dir=None, speed=3.0):
    d = Demo(headless, seed, screenshot_dir, speed)
    try:
        m = d.run(max_picks)
        print(f"\nDemo finished. Sorted {m['sorted']} items, recovered value {m['value']:.1f} "
              f"({m['recovery_rate']:.0f}% of possible), hazards caught {m['hazards_caught']}, "
              f"escaped {m['hazards_escaped']}, replans {m['replans']}, tool: {m['tool']}")
        print("Bins:", m["bins"])
        print("\nEvent log (last 15):")
        print("\n".join(d.env.log[-15:]))
        if not headless:
            fps = d.n_frames / max(1e-6, time.perf_counter() - d.t_start)
            print(f"\nAverage speed: {fps:.0f} frames/second (60 is smooth; tell me this number if it still feels slow)")
            input("\nPress Enter to close the window...")
    finally:
        d.world.close()
