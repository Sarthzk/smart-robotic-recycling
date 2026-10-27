"""Live dashboard. Start the 3D demo in one terminal (python main.py demo) and run
    streamlit run dashboard/app.py
in another. It refreshes every second from logs/live_metrics.json."""
import json
import os
import time
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "logs", "live_metrics.json")
BIN_COLORS = {"ewaste": "#3366e6", "metal": "#99a6b3", "plastic": "#f2cc33", "hazardous": "#e62626", "overflow": "#b366e6"}

st.set_page_config(page_title="Smart Robotic Recycling", page_icon="♻️", layout="wide")
st.title("♻️ Smart Robotic Recycling System")
st.caption("A robot arm that learns which waste to recover first")


def load():
    try:
        with open(METRICS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@st.fragment(run_every=1)
def live():
    m = load()
    if m is None:
        st.info("Waiting for the simulation... run `python main.py demo` in another terminal.")
        return
    stale = time.time() - m.get("stamp", 0) > 10 and not m.get("finished")
    st.markdown(f"**Status:** {'⚠️ simulation stopped' if stale else ('✅ run finished' if m['finished'] else '🟢 running')}"
                f" &nbsp;|&nbsp; belt time {m['time']} &nbsp;|&nbsp; tool: **{m['tool']}**")
    st.info(m["banner"])
    c = st.columns(5)
    c[0].metric("Items sorted", m["sorted"])
    c[1].metric("Value recovered", f"{m['value']:.1f}")
    c[2].metric("Recovery rate", f"{m['recovery_rate']:.0f}%")
    c[3].metric("Hazards caught", m["hazards_caught"])
    c[4].metric("Hazards escaped", m["hazards_escaped"])
    c = st.columns(3)
    c[0].metric("Replans", m["replans"])
    c[1].metric("Gripper failures", m["gripper_fails"])
    c[2].metric("Possible value", f"{m['possible_value']:.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Bin fill levels")
        for name, n in m["bins"].items():
            cap = m["capacity"][name]
            st.markdown(f"<span style='color:{BIN_COLORS[name]}'>■</span> **{name}** — {n} / {cap}"
                        f"{'  🔴 FULL' if n >= cap else ''}", unsafe_allow_html=True)
            st.progress(min(1.0, n / cap))
    with right:
        st.subheader("Robot's current priorities (best first)")
        if m["ranking"]:
            st.table([{"item": r["item"], "recovery score": r["score"]} for r in m["ranking"]])
        else:
            st.write("No items in reach.")
        st.subheader("Recent events")
        st.code("\n".join(m["events"][-8:]) or "-", language=None)


live()

st.divider()
a, b = st.columns(2)
for col, name, title in ((a, "learning_curve.png", "Learning curve"), (b, "comparison.png", "Colour vs fixed vs learned sorting")):
    path = os.path.join(ROOT, "results", name)
    if os.path.exists(path):
        col.subheader(title)
        col.image(path)
