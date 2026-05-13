from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RESULTS_DIR))

import style  # noqa: E402 — applies rcParams as a side effect

from src.formation import SWARM_STDDEVS_30KM, generate_swarm_random_orbital_parameters
from src.network import get_nodes_by_type, simulate_network
from src.params import SimulationParams
from src.radio import radio_for_target
from src.scheduling import (
    SlotBudgetPolicy,
    _compute_visibility,
    _visibility_windows_fast,
    build_schedule,
    equitable_slot_budget,
)
from config import (
    SEED, N_SATELLITES,
    BASE_ORBITAL_PARAMS, STATIONS,
    DURATION_S, DT_S, R_MAX_M, ELEV_MIN_DEG,
    DL_RATE_BPS, DL_DISTANCE_M, DL_FREQ_HZ, DL_POWER_TX_W, DL_GAIN_TX_DBI, DL_GAIN_RX_DBI,
    XL_DISTANCE_M, XL_FREQ_HZ, XL_POWER_TX_W, XL_GAIN_TX_DBI, XL_GAIN_RX_DBI,
)

# ---------------------------------------------------------------------------
# Snapshot-specific parameters
# ---------------------------------------------------------------------------

INITIAL_BUFFER_LENGTH_BITS = int(5000 * 8e6)
XL_RATE_BPS = 25e6

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _visibility_spans(
    graphs: list[nx.Graph],
    station: str,
    time_step_s: float,
) -> list[tuple[float, float]]:
    """Return (t_start, t_end) intervals where *station* has degree > 0."""
    visible = np.array([graphs[t].degree(station) > 0 for t in range(len(graphs))])
    spans: list[tuple[float, float]] = []
    in_pass = False
    for t, v in enumerate(visible):
        if v and not in_pass:
            t_start = t * time_step_s
            in_pass = True
        elif not v and in_pass:
            spans.append((t_start, t * time_step_s))
            in_pass = False
    if in_pass:
        spans.append((t_start, len(graphs) * time_step_s))
    return spans


def _compress_timeline_fixed_gap(
    active_windows_s: list[tuple[float, float]],
    total_s: float,
    gap_w: float,
) -> tuple[Callable[[float], float], float, list[tuple[float, float, float, float, bool]]]:
    """Compress timeline so all inter-pass gaps share the same display width *gap_w*."""
    if not active_windows_s:
        return lambda t: t, total_s, [(0.0, total_s, 0.0, total_s, False)]

    segments: list[tuple[float, float, float, float, bool]] = []
    comp, prev_end = 0.0, 0.0

    for ws, we in active_windows_s:
        if ws > prev_end + 1e-9:
            segments.append((prev_end, ws, comp, comp + gap_w, False))
            comp += gap_w
        segments.append((ws, we, comp, comp + (we - ws), True))
        comp += we - ws
        prev_end = we

    if prev_end < total_s - 1e-9:
        segments.append((prev_end, total_s, comp, comp + gap_w, False))
        comp += gap_w

    def real_to_comp(t: float) -> float:
        for rs, re, cs, ce, _ in segments:
            if t <= re + 1e-9:
                frac = (t - rs) / (re - rs) if (re - rs) > 1e-9 else 0.0
                return cs + frac * (ce - cs)
        return comp

    return real_to_comp, comp, segments


def plot_average_degree(
    graphs: list[nx.Graph],
    time_step_s: float,
    file_name: Path,
    include_ground: bool = False,
    show_std: bool = False,
    draw_visibility_station: str | None = None,
) -> None:
    """Plot mean satellite network degree over time, optionally with ±std and visibility spans."""
    nodes = (
        list(graphs[0].nodes())
        if include_ground
        else [n for n in graphs[0].nodes() if n.startswith("SAT_")]
    )
    degree_arrays = [
        np.array([G.subgraph(nodes).degree(n) for n in nodes]) if nodes else np.zeros(1)
        for G in graphs
    ]
    means = np.array([a.mean() for a in degree_arrays])
    times = np.arange(len(graphs)) * time_step_s

    fig, ax = plt.subplots()
    if draw_visibility_station is not None:
        spans = _visibility_spans(graphs, draw_visibility_station, time_step_s)
        for i, (t0, t1) in enumerate(spans):
            ax.axvspan(
                t0, t1,
                hatch="//////",
                facecolor="none",
                edgecolor="steelblue",
                linewidth=0.0,
                label="visibility window" if i == 0 else "_nolegend_",
            )
    ax.plot(times, means, label="average", color="red")
    if show_std:
        stds = np.array([a.std() for a in degree_arrays])
        ax.fill_between(times, means - stds, means + stds, alpha=0.25, color="red",
                        label="standard deviation")
    if draw_visibility_station is not None or show_std:
        ax.legend(loc="upper right", fontsize=5.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degree")
    ax.set_ylim(0, means.max() + 7)
    ax.tick_params(axis="both")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(file_name)


def plot_schedule(
    graphs: list[nx.Graph],
    params: SimulationParams,
    instances: list[list[str] | None],
    file_name: Path,
    instance_labels: list[str] | None = None,
    slot_budget: SlotBudgetPolicy = equitable_slot_budget,
    gap_display_s: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Multi-instance schedule visualisation — one row per instance, shared compressed time axis.

    Args:
        graphs: connectivity graphs, one per simulation step.
        params: simulation parameters.
        instances: one relay_nodes list per row (None → direct mode, all satellites).
        instance_labels: y-axis label for each row; defaults to "Instance i".
        slot_budget: slot allocation policy.
        gap_display_s: display width of inter-pass gaps; defaults to 10% of the shortest pass.
        figsize: figure size override.
        file_name: output path for the saved figure.
    """
    fs = plt.rcParams["font.size"] + 14

    sat_nodes, station_nodes = get_nodes_by_type(graphs[0])
    dt_s = params.dt_s
    deadline_step = math.ceil(params.deadline_s / dt_s)
    reconfig_steps = math.ceil(params.gs_reconfig_s / dt_s)
    reconfig_s = reconfig_steps * dt_s
    sat_index = {sat: i for i, sat in enumerate(sat_nodes)}
    gs_list = sorted(station_nodes)

    visibility = _compute_visibility(graphs, sat_nodes, station_nodes)
    labels = instance_labels or [f"Instance {i}" for i in range(len(instances))]

    instance_data = []
    for relay_nodes in instances:
        schedulable_sats = relay_nodes if relay_nodes is not None else sat_nodes
        sorted_sats = sorted(schedulable_sats, key=lambda x: int(x.split("_")[1]))
        schedule = build_schedule(graphs, params, sorted_sats, slot_budget=slot_budget)
        instance_data.append((schedulable_sats, schedule))

    global_visible = np.zeros(deadline_step, dtype=bool)
    for schedulable_sats, _ in instance_data:
        for gs_name in gs_list:
            for sat in schedulable_sats:
                global_visible |= visibility[gs_name][sat_index[sat], :deadline_step]
    global_windows_s = [
        (ws * dt_s, we * dt_s)
        for ws, we in _visibility_windows_fast(global_visible, deadline_step)
    ]

    n_rows = len(instances)
    if not global_windows_s:
        plt.subplots(n_rows, 1, figsize=figsize or (12, 2 * n_rows))
        return

    if gap_display_s is None:
        min_pass_s = min(we - ws for ws, we in global_windows_s)
        gap_display_s = max(min_pass_s * 0.1, reconfig_s, 1.0)

    real_to_comp, total_comp, segments = _compress_timeline_fixed_gap(
        global_windows_s, params.deadline_s, gap_display_s
    )
    deadline_comp = real_to_comp(params.deadline_s)

    comm_color = "#4878CF"
    reconfig_color = "#aaaaaa"

    fig_w = max(10.0, len(global_windows_s) * 2.0 + 4)
    fig_h = max(2.8, n_rows * 1.2 + 1.2)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize or (fig_w, fig_h),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for row_idx, (schedulable_sats, schedule) in enumerate(instance_data):
        ax = axes[row_idx]

        for _, _, cs, ce, is_active in segments:
            if not is_active:
                ax.add_patch(mpatches.Rectangle(
                    (cs, 0.0), ce - cs, 1.0,
                    facecolor="#f0f0f0", edgecolor="#bbbbbb",
                    hatch="///", linewidth=0.4, zorder=1,
                ))

        vis_union = np.zeros(deadline_step, dtype=bool)
        for gs_name in gs_list:
            for sat in schedulable_sats:
                vis_union |= visibility[gs_name][sat_index[sat], :deadline_step]
        for ws, we in _visibility_windows_fast(vis_union, deadline_step):
            cs = real_to_comp(ws * dt_s)
            ce = real_to_comp(we * dt_s)
            ax.add_patch(mpatches.Rectangle(
                (cs, 0.0), ce - cs, 1.0,
                facecolor="#e8e8e8", linewidth=0, zorder=2,
            ))

        all_slots = []
        for gs_name in gs_list:
            all_slots.extend(schedule.get(gs_name, []))
        all_slots.sort(key=lambda s: s.step_start)

        prev_slot = None
        for slot in all_slots:
            s_start = slot.step_start * dt_s
            s_end = slot.step_end * dt_s
            cs_start = real_to_comp(s_start)
            cs_end = real_to_comp(s_end)

            if (
                prev_slot is not None
                and reconfig_s > 0
                and slot.satellite != prev_slot.satellite
            ):
                rc_start_step = slot.step_start - reconfig_steps
                between = vis_union[prev_slot.step_end : slot.step_start]
                same_window = len(between) == 0 or bool(between.all())
                if same_window and rc_start_step >= prev_slot.step_end:
                    ax.add_patch(mpatches.Rectangle(
                        (real_to_comp(rc_start_step * dt_s), 0.1),
                        real_to_comp(s_start) - real_to_comp(rc_start_step * dt_s),
                        0.8,
                        facecolor=reconfig_color, edgecolor="white",
                        linewidth=0.4, hatch="xx", alpha=0.9, zorder=3,
                    ))

            ax.add_patch(mpatches.Rectangle(
                (cs_start, 0.1), cs_end - cs_start, 0.8,
                facecolor=comm_color, linewidth=0, zorder=4, alpha=0.85,
            ))

            if (cs_end - cs_start) > total_comp * 0.01:
                ax.text(
                    (cs_start + cs_end) / 2, 0.5,
                    slot.satellite.split("_")[1],
                    ha="center", va="center",
                    fontsize=fs - 1.5, color="white", fontweight="bold",
                    zorder=5, clip_on=True,
                )

            prev_slot = slot

        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, deadline_comp * 1.01)
        ax.set_yticks([0.5])
        ax.set_yticklabels([""])
        ax.tick_params(left=False)
        ax.spines[["top", "right", "left"]].set_visible(False)

    tick_pos = [cs for _, _, cs, _, is_active in segments if is_active]
    tick_lab = [f"{rs:.0f}" for rs, _, _, _, is_active in segments if is_active]
    tick_pos.append(deadline_comp)
    tick_lab.append(f"{params.deadline_s:.0f}")
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_lab, fontsize=fs - 2, rotation=40, ha="right")
    axes[-1].set_xlabel("Time (s)", fontsize=fs - 1)

    legend_handles = [
        mpatches.Patch(facecolor=comm_color, alpha=0.85, label="Communication"),
        mpatches.Patch(facecolor=reconfig_color, edgecolor="white", hatch="xx",
                       linewidth=0.4, alpha=0.9, label="Reconfiguration"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#bbbbbb", hatch="///",
                       linewidth=0.4, label="Inter-pass gap (compressed)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=fs,
        bbox_to_anchor=(0.5, -0.1),
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(file_name, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

downlink = radio_for_target(
    DL_RATE_BPS, DL_DISTANCE_M,
    frequency_hz=DL_FREQ_HZ, power_tx_w=DL_POWER_TX_W,
    gain_tx_dbi=DL_GAIN_TX_DBI, gain_rx_dbi=DL_GAIN_RX_DBI,
)
crosslink = radio_for_target(
    XL_RATE_BPS, XL_DISTANCE_M,
    frequency_hz=XL_FREQ_HZ, power_tx_w=XL_POWER_TX_W,
    gain_rx_dbi=XL_GAIN_RX_DBI, gain_tx_dbi=XL_GAIN_TX_DBI,
)

parameters = generate_swarm_random_orbital_parameters(
    N_SATELLITES, BASE_ORBITAL_PARAMS, seed=SEED, stddevs=SWARM_STDDEVS_30KM
)

graphs, positions = simulate_network(
    parameters, STATIONS,
    duration=DURATION_S, R_max=R_MAX_M,
    time_step_s=DT_S, elev_min_deg=ELEV_MIN_DEG,
)

plot_average_degree(
    graphs,
    time_step_s=DT_S,
    include_ground=True,
    show_std=True,
    draw_visibility_station="GS_0",
    file_name=PROJECT_ROOT / "figures" / "figure_2.pdf",
)

params = SimulationParams(
    duration_s=DURATION_S,
    dt_s=DT_S,
    initial_buffer_length_bits=INITIAL_BUFFER_LENGTH_BITS,
    isl_radio=crosslink,
    downlink_radio=downlink,
    deadline_s=DURATION_S,
    gs_reconfig_s=30.0,
)

plot_schedule(
    graphs,
    params,
    instances=[[f"SAT_{i}" for i in range(3)], [f"SAT_{i}" for i in range(15)]],
    instance_labels=["3 relays", "Direct"],
    file_name=PROJECT_ROOT / "figures" / "figure_3.pdf",
)
