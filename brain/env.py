"""Headless conveyor simulation: items arrive, the robot picks one at a time,
unpicked items leave the belt. Fast (pure Python/NumPy) so we can train on thousands of episodes."""
import numpy as np
import config
from brain.items import spawn_items


class RecyclingEnv:
    def __init__(self, seed=0, capacity=None, forced_fail_attempts=None, n_items=config.N_ITEMS):
        self.seed = seed
        self.capacity = dict(config.BIN_CAPACITY if capacity is None else capacity)
        self.forced_fail_attempts = set(forced_fail_attempts or [])   # pick attempts that must fail (for demos)
        self.n_items = n_items
        self.reset()

    # ---------------------------------------------------------------- setup
    def reset(self):
        self.item_rng = np.random.default_rng(self.seed)
        self.fail_rng = np.random.default_rng(self.seed + 10_000)
        self.items = spawn_items(self.item_rng, self.n_items)
        self.pending = list(self.items)
        self.t = self.items[0].arrival
        self.bins = {b: 0 for b in config.BINS}
        self.tool = "gripper"
        self.consecutive_fails = 0
        self.attempts = 0
        self.log = []                                  # human-readable event log
        self.m = dict(sorted=0, value=0.0, hazards_caught=0, hazards_escaped=0,
                      replans=0, gripper_fails=0, reward=0.0,
                      possible_value=sum(i.recovered_value for i in self.items))
        self._expire()

    # ---------------------------------------------------------------- queries
    def visible(self):
        return [i for i in self.pending if i.arrival <= self.t < i.arrival + config.DWELL]

    def destination(self, item):
        """Where would this item go right now? None = no room anywhere."""
        if self.bins[item.bin] < self.capacity[item.bin]:
            return item.bin
        if self.bins["overflow"] < self.capacity["overflow"]:
            return "overflow"
        return None

    def candidates(self):
        return [i for i in self.visible() if self.destination(i) is not None]

    @property
    def done(self):
        return not self.pending

    # ---------------------------------------------------------------- dynamics
    def _say(self, msg):
        self.log.append(f"t={self.t:5.1f}  {msg}")

    def _expire(self):
        """Items that left the robot's reach are lost; escaped hazards are punished."""
        penalty = 0.0
        for it in [i for i in self.pending if i.arrival + config.DWELL <= self.t]:
            self.pending.remove(it)
            if it.hazard > 0.5:
                self.m["hazards_escaped"] += 1
                penalty += config.HAZARD_ESCAPE_PENALTY * it.hazard
                self._say(f"HAZARD ESCAPED: {it.kind} #{it.uid}")
        self.m["reward"] -= penalty
        return -penalty

    def advance_idle(self):
        """Nothing to pick: wait for the next arrival."""
        nxt = [i.arrival for i in self.pending if i.arrival > self.t]
        self.t = min(nxt) if nxt else self.t + 1.0
        return self._expire()

    def step(self, item):
        """Robot tries to pick `item`. Returns dict(reward, outcome, dest, duration, tool, events)."""
        events = []
        dur = config.PICK_TIME + config.COST_TIME * item.cost
        if self.tool == "suction":
            dur *= config.BACKUP_TOOL_SLOWDOWN
        self.attempts += 1
        t0 = self.t

        fail = False
        if self.tool == "gripper":
            p = config.GRIPPER_FAIL_PROB * (0.5 + item.cost / 2.0)
            fail = self.attempts in self.forced_fail_attempts or self.fail_rng.random() < p

        self.t += dur
        reward = -config.TIME_PENALTY * dur
        if fail:
            self.consecutive_fails += 1
            self.m["gripper_fails"] += 1
            events.append(f"gripper failed on {item.kind} #{item.uid}")
            if self.consecutive_fails >= config.MAX_GRIPPER_FAILS:
                self.tool = "suction"
                self.m["replans"] += 1
                events.append("REPLAN: gripper unreliable -> switching to backup suction tool")
            else:
                self.m["replans"] += 1
                events.append("REPLAN: retrying / choosing next best item")
            outcome, dest = "failed", None
        else:
            self.consecutive_fails = 0
            dest = self.destination(item)
            factor = 1.0
            if dest != item.bin:
                factor = config.OVERFLOW_VALUE_FACTOR
                self.m["replans"] += 1
                events.append(f"REPLAN: {item.bin} bin full -> {item.kind} sent to overflow")
            self.bins[dest] += 1
            self.pending.remove(item)
            gained = item.recovered_value * factor
            reward += gained
            self.m["sorted"] += 1
            self.m["value"] += gained
            if item.hazard > 0.5:
                self.m["hazards_caught"] += 1
                reward += config.HAZARD_PICK_BONUS * item.hazard
            outcome = "picked"
            events.append(f"picked {item.kind} #{item.uid} -> {dest}")

        for e in events:
            self._say(e)
        self.m["reward"] += reward
        reward += self._expire()
        return dict(reward=reward, outcome=outcome, dest=dest, duration=dur, t0=t0,
                    t1=self.t, tool=self.tool, events=events, item=item)

    # ---------------------------------------------------------------- results
    def metrics(self):
        m = dict(self.m)
        m["recovery_rate"] = 100.0 * m["value"] / m["possible_value"]
        m["bins"] = dict(self.bins)
        m["tool"] = self.tool
        return m


def run_episode(planner, seed, record=False, **env_kwargs):
    """Run one full episode with `planner`. If record, also return the
    (feature matrix, chosen index, reward) trajectory used for learning."""
    env = RecyclingEnv(seed=seed, **env_kwargs)
    rng = np.random.default_rng(seed + 99)
    traj = []
    while not env.done:
        cands = env.candidates()
        if not cands:
            r = env.advance_idle()
            if traj:
                traj[-1][2] += r            # idle-time escapes count against the last decision
            continue
        idx, F = planner.choose(cands, env.t, rng)
        info = env.step(cands[idx])
        if record:
            traj.append([F, idx, info["reward"]])
    return (env.metrics(), traj) if record else env.metrics()
