import numpy as np
import config
from brain.env import RecyclingEnv, run_episode
from brain.items import WasteItem
from brain.planner import Planner
from brain.scoring import recovery_score

W = np.array([2.7, -0.9, 2.4, 0.3, 0.7])     # roughly the learned weights


def item(kind, uid=0, arrival=0.0, contamination=0.1):
    b, (vlo, vhi), hazard, cost, _ = config.ITEM_TYPES[kind]
    return WasteItem(uid, kind, b, (vlo + vhi) / 2, contamination, hazard, cost, arrival)


def test_score_order_hazard_and_value_beat_plastic():
    leak, pcb, pet = item("leaking_battery"), item("circuit_board"), item("pet_bottle")
    s = {i.kind: recovery_score(i, W, 0.0) for i in (leak, pcb, pet)}
    assert s["leaking_battery"] > s["pet_bottle"] and s["circuit_board"] > s["pet_bottle"]


def test_contamination_lowers_score():
    clean, dirty = item("phone", contamination=0.0), item("phone", contamination=0.8)
    assert recovery_score(clean, W, 0) > recovery_score(dirty, W, 0)


def test_bin_full_reroutes_to_overflow():
    env = RecyclingEnv(seed=1, capacity={**config.BIN_CAPACITY, "plastic": 0}, forced_fail_attempts=[])
    pet = item("pet_bottle")
    assert env.destination(pet) == "overflow"
    env.pending, env.items, env.t = [pet], [pet], 0.0
    info = env.step(pet)
    assert info["dest"] == "overflow" and any("overflow" in e for e in info["events"])


def test_no_room_anywhere_means_not_a_candidate():
    cap = {b: 0 for b in config.BINS}
    env = RecyclingEnv(seed=1, capacity=cap)
    env.t = env.items[0].arrival
    assert env.candidates() == []


def test_gripper_failures_switch_to_backup_tool():
    env = RecyclingEnv(seed=1, forced_fail_attempts=[1, 2])
    env.pending = [item("phone", 1), item("phone", 2)]
    env.t = 0.0
    r1 = env.step(env.pending[0]); r2 = env.step(env.pending[0])
    assert r1["outcome"] == r2["outcome"] == "failed" and env.tool == "suction"
    slow = env.step(env.pending[0])
    assert slow["outcome"] == "picked" and slow["duration"] > config.PICK_TIME + config.COST_TIME * 1.0


def test_hazard_escape_is_punished():
    env = RecyclingEnv(seed=2)
    env.pending = [item("leaking_battery", 5)]
    env.t = 100.0
    env._expire()
    assert env.m["hazards_escaped"] == 1 and env.m["reward"] < 0


def test_smart_planner_beats_colour_sorting():
    seeds = range(7000, 7040)
    colour = np.mean([run_episode(Planner(mode="colour"), s)["recovery_rate"] for s in seeds])
    smart = np.mean([run_episode(Planner(W), s)["recovery_rate"] for s in seeds])
    assert smart > colour + 10


def test_learning_improves_over_untrained():
    from brain.learner import train
    w, _ = train(episodes=600, seed=3, verbose=False)
    seeds = range(8000, 8040)
    untrained = np.mean([run_episode(Planner(np.zeros(5)), s)["reward"] for s in seeds])
    learned = np.mean([run_episode(Planner(w), s)["reward"] for s in seeds])
    assert learned > untrained + 10


def test_3d_demo_smoke():
    """The PyBullet scene runs headless: the arm sorts a few items and writes live metrics."""
    import json
    from sim.demo import run_demo, METRICS_PATH
    run_demo(headless=True, seed=7, max_picks=3)
    m = json.load(open(METRICS_PATH))
    assert m["finished"] and m["sorted"] >= 1
