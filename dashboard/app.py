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
st.markdown("""
<style>
.block-container {padding-top: 4.6rem; max-width: 1250px;}
pre, code {white-space: pre-wrap !important; word-break: break-word;}
.hero {background: linear-gradient(135deg, #123a2e 0%, #18202b 70%); border: 1px solid #25414a;
       border-radius: 14px; padding: 18px 24px; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 1.9rem;}
.hero p {margin: 4px 0 0 0; color: #9fb3c8;}
div[data-testid="stMetric"] {background: #18202b; border: 1px solid #25303d; border-radius: 12px; padding: 12px 16px;}
div[data-testid="stMetricLabel"] {color: #9fb3c8;}
.pill {display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin-right: 8px;}
.bar {background:#222c39; border-radius: 8px; height: 14px; overflow: hidden; margin: 4px 0 12px 0;}
.bar > div {height: 100%; border-radius: 8px;}
.banner {background:#18202b; border-left: 5px solid #2ecc8f; border-radius: 8px; padding: 12px 16px; margin: 10px 0 14px 0;}
.banner.warn {border-left-color: #ff5a5a;}
</style>
<div class="hero"><h1>♻️ Smart Robotic Recycling System</h1>
<p>A robot arm that learns which waste to recover first &nbsp;·&nbsp; live simulation dashboard</p></div>
""", unsafe_allow_html=True)


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
    state, bg = (("STOPPED", "#6b3030") if stale else (("FINISHED", "#2b4a7a") if m["finished"] else ("RUNNING", "#1f6b4a")))
    tool_bg = "#7a5a10" if m["tool"] == "suction" else "#1f4a3a"
    st.markdown(f"<span class='pill' style='background:{bg}'>● {state}</span>"
                f"<span class='pill' style='background:{tool_bg}'>tool: {m['tool']}</span>"
                f"<span style='color:#9fb3c8'>belt time {m['time']}</span>", unsafe_allow_html=True)
    warn = any(k in m["banner"] for k in ("FAILED", "FULL", "ESCAPED"))
    st.markdown(f"<div class='banner{' warn' if warn else ''}'>{m['banner']}</div>", unsafe_allow_html=True)
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
            full = n >= cap
            st.markdown(f"<span style='color:{BIN_COLORS[name]}'>■</span> **{name}** &nbsp; {n} / {cap}"
                        f"{' &nbsp; <b style=\"color:#ff5a5a\">FULL</b>' if full else ''}"
                        f"<div class='bar'><div style='width:{min(1.0, n / cap) * 100:.0f}%;"
                        f"background:{'#ff5a5a' if full else BIN_COLORS[name]}'></div></div>", unsafe_allow_html=True)
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
