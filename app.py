"""Streamlit demo: diagnose the root cause of a simulated incident.

Run with:  streamlit run app.py

Streamlit reruns this whole file from top to bottom every time a widget
changes. Values that must survive a rerun (which incident we are on, and its
root cause) are kept in st.session_state.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import diagnose` works

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.lines import Line2D

from diagnose.baselines import most_alarmed_descendants, most_upstream, random_ranking
from diagnose.exact import exact_posterior
from diagnose.graph import Graph, generate_graph
from diagnose.model import Params
from diagnose.sampling import likelihood_weighting
from diagnose.simulate import Incident, simulate_incident

# Colors shared with the experiment figures.
EXACT_COLOR = "#2a78d6"
LW_COLOR = "#eb6834"
FAILED_COLOR = "#e34948"
HEALTHY_COLOR = "#c9c8c2"
INK = "#0b0b0b"

plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})

# Page setup, with larger fonts so the demo is readable on a projector.
st.set_page_config(page_title="Root-cause diagnosis", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 21px; }
    [data-testid="stWidgetLabel"] p { font-size: 18px !important; }
    [data-testid="stTable"] td, [data-testid="stTable"] th { font-size: 20px !important; }
    p.caption { font-size: 20px; opacity: 0.8; font-style: italic; margin-top: -0.4rem; margin-bottom: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def caption(text: str) -> None:
    """One-sentence plain-language caption under an element."""
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


def plural(k: int, word: str) -> str:
    """Format a count with a singular or plural noun."""
    return f"{k} {word}" if k == 1 else f"{k} {word}s"


def layout(graph: Graph) -> np.ndarray:
    """Layered layout: y from topological depth, x spread within each layer.

    Nodes in a layer are ordered by the mean x position of their parents,
    which keeps most edges short and reduces crossings.
    """
    depth = graph.layers()
    pos = np.zeros((graph.n, 2))
    for d in range(max(depth) + 1):
        nodes = [i for i in range(graph.n) if depth[i] == d]
        if d > 0:
            nodes.sort(key=lambda i: np.mean([pos[j, 0] for j in graph.parents[i]]))
        xs = np.arange(len(nodes)) - (len(nodes) - 1) / 2
        for x, i in zip(xs, nodes):
            pos[i] = (x, -d)
    return pos


def draw_graph(graph: Graph, inc: Incident) -> plt.Figure:
    """Draw services in layers: red if failed, bold outline if alarmed."""
    pos = layout(graph)
    width = max(8.0, 1.6 * (np.ptp(pos[:, 0]) + 2))
    height = max(4.0, 1.5 * (np.ptp(pos[:, 1]) + 1.5))
    fig, ax = plt.subplots(figsize=(width, height))
    # One arrow per dependency, from parent j to child i.
    for i, ps in enumerate(graph.parents):
        for j in ps:
            ax.annotate("", xy=pos[i], xytext=pos[j],
                        arrowprops=dict(arrowstyle="-|>", color="#8a8984", lw=1.5,
                                        shrinkA=22, shrinkB=22, mutation_scale=18))
    # One circle per service: fill shows failed or healthy, outline shows alarm.
    for i, name in enumerate(graph.names()):
        color = FAILED_COLOR if inc.failures[i] else HEALTHY_COLOR
        edge_w = 4.5 if inc.alarms[i] else 1.0
        ax.scatter(*pos[i], s=1500, color=color, edgecolors=INK, linewidths=edge_w, zorder=3)
        ax.text(*pos[i], name, ha="center", va="center", fontsize=15, zorder=4,
                color="white" if inc.failures[i] else INK, fontweight="bold")
    legend = [
        Line2D([], [], marker="o", ls="", ms=16, mfc=HEALTHY_COLOR, mec=INK, label="healthy"),
        Line2D([], [], marker="o", ls="", ms=16, mfc=FAILED_COLOR, mec=INK, label="failed"),
        Line2D([], [], marker="o", ls="", ms=16, mfc="white", mec=INK, mew=4, label="alarm raised"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=14)
    ax.set_xlim(pos[:, 0].min() - 0.8, pos[:, 0].max() + 0.8)
    ax.set_ylim(pos[:, 1].min() - 0.6, 0.6)
    ax.axis("off")
    return fig


def draw_posterior(names: list[str], exact: np.ndarray, lw: np.ndarray, root: int) -> plt.Figure:
    """Side by side bars of the two posteriors with the true root marked."""
    n = len(names)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.85 * n), 4.6))
    ax.bar(x - 0.21, exact, width=0.4, color=EXACT_COLOR, label="exact inference")
    ax.bar(x + 0.21, lw, width=0.4, color=LW_COLOR, label="likelihood weighting")
    top = max(exact.max(), lw.max())
    ax.annotate("true root", xy=(root, max(exact[root], lw[root]) + 0.02), xytext=(root, top + 0.15),
                ha="center", fontsize=14, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.set_xticks(x, names)
    ax.get_xticklabels()[root].set_fontweight("bold")
    ax.set_ylim(0, min(1.25, top + 0.3))
    ax.set_ylabel("P(root cause | alarms)")
    ax.grid(axis="y", color="#e4e3df")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    return fig


# Sidebar controls.
st.sidebar.header("Simulation settings")
n = st.sidebar.slider("Number of services", 6, 16, 12)
p_prop = st.sidebar.slider("Failure propagation probability", 0.1, 1.0, 0.7, 0.05)
fp = st.sidebar.slider("False alarm rate (false positive)", 0.0, 0.4, 0.05, 0.01)
fn = st.sidebar.slider("Missed alarm rate (false negative)", 0.0, 0.5, 0.1, 0.01)
n_samples = st.sidebar.select_slider("Samples for likelihood weighting", [500, 1000, 5000, 20000, 50000], 5000)
seed = int(st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=0, step=1))

# Every new incident bumps incident_id, which changes the random seed used to
# simulate the cascade and the alarms.
if "incident_id" not in st.session_state:
    st.session_state.incident_id = 0
    st.session_state.root = int(np.random.default_rng([seed, 0]).integers(0, n))

if st.sidebar.button("Inject random failure", width="stretch"):
    st.session_state.incident_id += 1
    st.session_state.root = int(np.random.default_rng([seed, st.session_state.incident_id]).integers(0, n))

graph = generate_graph(n, seed=seed)
chosen = st.sidebar.selectbox("Root cause to inject", graph.names())
if st.sidebar.button("Choose root cause", width="stretch"):
    st.session_state.incident_id += 1
    st.session_state.root = graph.names().index(chosen)
if st.session_state.root >= n:
    # The graph was shrunk with the slider; keep the root inside the graph.
    st.session_state.root = n - 1

# Simulate the incident, then run both inference methods on its alarms only.
params = Params(p_prop=p_prop, fp_rate=fp, fn_rate=fn)
rng = np.random.default_rng([seed, st.session_state.incident_id, 1])
inc = simulate_incident(graph, params, rng, root=st.session_state.root)
exact = exact_posterior(graph, params, inc.alarms)
lw = likelihood_weighting(graph, params, inc.alarms, n_samples, np.random.default_rng([seed, 2]))
names = graph.names()

# Page body: the graph, the two posteriors, and a summary table.
st.title("Which service caused the outage?")
st.markdown("All data on this page is **simulated**. Nothing here comes from a real system.")

st.subheader("Service dependency graph (simulated)")
st.pyplot(draw_graph(graph, inc), width="content")
caption(
    f"Arrows point from a service to the services that depend on it; {plural(int(inc.failures.sum()), 'service')} "
    f"failed after we broke {names[inc.root]}, and {plural(int(inc.alarms.sum()), 'alarm')} went off, which can include false alarms."
)

st.subheader("Who is to blame? Posterior over suspects")
st.pyplot(draw_posterior(names, exact, lw.posterior, inc.root), width="content")
caption(
    f"Taller bars mean the alarms point more strongly at that service; likelihood weighting used "
    f"{n_samples:,} samples with an effective sample size of {lw.ess:,.0f}, and more noise spreads the bars out."
)

st.subheader("What each method blamed")
# Each method's single top suspect.
blame = {
    "Exact inference": int(np.argmax(exact)),
    "Likelihood weighting": int(np.argmax(lw.posterior)),
    "Most upstream alarm": most_upstream(graph, inc.alarms)[0],
    "Most alarmed descendants": most_alarmed_descendants(graph, inc.alarms)[0],
    "Random guess": random_ranking(graph, np.random.default_rng([seed, 3]))[0],
}
st.table({
    "Method": list(blame),
    "Blamed service": [names[b] for b in blame.values()],
    "Correct?": ["yes" if b == inc.root else "no" for b in blame.values()],
})
caption(f"The true root cause in this simulated incident is {names[inc.root]}; each row shows the top suspect of one method.")
