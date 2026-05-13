"""
scheduling.py
-------------
Pre-computed downlink schedule: visibility computation, slot budget
policies, schedule construction, and schedule visualisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np

from .network import get_nodes_by_type
from .params import SimulationParams

# ---------------------------------------------------------------------------
# Core scheduling types
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    """Transmission slot assigned to a satellite."""

    satellite: str
    step_start: int  # inclusive
    step_end: int  # exclusive


Schedule = dict[str, list[Slot]]


# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------


def _compute_visibility(
    graphs: list[nx.Graph],
    sat_nodes: list[str],
    station_nodes: frozenset[str],
) -> dict[str, np.ndarray]:
    """
    Precomputes satellite↔station visibility in a single pass over the graphs.

    Returns {gs: array(N_sat, N_time, dtype=bool)}.
    Replaces N_sat × N_time calls to G.has_edge by iterating only over
    actually visible satellites via G.neighbors(gs).
    """
    N_sat = len(sat_nodes)
    N_time = len(graphs)
    sat_index = {sat: idx for idx, sat in enumerate(sat_nodes)}

    visibility: dict[str, np.ndarray] = {
        gs: np.zeros((N_sat, N_time), dtype=bool) for gs in station_nodes
    }

    for t, G in enumerate(graphs):
        for gs in station_nodes:
            if gs not in G:
                continue
            for neighbor in G.neighbors(gs):
                if neighbor in sat_index:
                    visibility[gs][sat_index[neighbor], t] = True

    return visibility


def _visibility_windows_fast(
    visible_row: np.ndarray,
    deadline_step: int,
) -> list[tuple[int, int]]:
    """
    Extracts (step_start, step_end) windows from a precomputed boolean vector.

    O(deadline_step) — no graph access.
    """
    v = visible_row[:deadline_step].view(np.uint8)
    padded = np.empty(len(v) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1 : 1 + len(v)] = v
    padded[-1] = 0
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Slot budget policies
# ---------------------------------------------------------------------------
#
# A SlotBudgetPolicy determines how many GS steps to allocate per satellite.
# The policy is passed to build_schedule and called once before assignment.
# Constraint: no knowledge of future traffic — only visibility windows and
# static parameters.
#
# SlotBudgetPolicy : (schedulable_sats, sat_nodes, params, visibility,
#                    sat_index, station_nodes, deadline_step, reconfig_steps)
#                   → dict[sat → tx_steps]

SlotBudgetPolicy = Callable[
    [
        list[str],  # schedulable_sats
        list[str],  # sat_nodes
        SimulationParams,
        dict[str, np.ndarray],  # visibility[gs] → array(N_sat, N_time, bool)
        dict[str, int],  # sat_index
        frozenset[str],  # station_nodes
        int,  # deadline_step
        int,  # reconfig_steps
    ],
    dict[str, int],  # sat → tx_steps
]


def equitable_slot_budget(
    schedulable_sats: list[str],
    sat_nodes: list[str],
    params: SimulationParams,
    visibility: dict[str, np.ndarray],
    sat_index: dict[str, int],
    station_nodes: frozenset[str],
    deadline_step: int,
    reconfig_steps: int,
) -> dict[str, int]:
    """
    Relay mode — equitable: divides total available GS time equally.

    Computes the union of all visibility windows across all relays and GS
    (each step counted once), then subtracts reconfiguration waste before
    dividing by the number of relays.
    """
    n_relays = len(schedulable_sats)
    visible_at = np.zeros(deadline_step, dtype=bool)
    for gs in station_nodes:
        for sat in schedulable_sats:
            visible_at |= visibility[gs][sat_index[sat], :deadline_step]
    union_steps = int(visible_at.sum())
    reconfig_waste = (n_relays - 1) * reconfig_steps
    equity_target = max(1, (union_steps - reconfig_waste) // n_relays)
    return {sat: equity_target for sat in schedulable_sats}


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------


def _assign_slots(
    schedulable_sats: list[str],
    station_nodes: frozenset[str],
    visibility: dict[str, np.ndarray],
    sat_index: dict[str, int],
    slot_budgets: dict[str, int],
    deadline_step: int,
    reconfig_steps: int,
) -> Schedule:
    """
    Window-first round-robin assignment (best effort, latest-first).

    Processes GS windows from latest to earliest. Within each window,
    all eligible satellites share the available time in rotation — the
    satellite that gets the latest (most favourable) slot rotates each
    window. Satellites that cannot fill their full budget receive as many
    steps as available without error.
    """
    schedule: Schedule = {gs: [] for gs in sorted(station_nodes)}
    remaining = {sat: slot_budgets[sat] for sat in schedulable_sats}

    for gs in sorted(station_nodes):
        per_sat_windows: dict[str, list[tuple[int, int]]] = {
            sat: _visibility_windows_fast(visibility[gs][sat_index[sat]], deadline_step)
            for sat in schedulable_sats
        }

        # Union of all visibility windows for this GS
        union_vis = np.zeros(deadline_step, dtype=bool)
        for sat in schedulable_sats:
            union_vis |= visibility[gs][sat_index[sat], :deadline_step]
        union_windows = _visibility_windows_fast(union_vis, deadline_step)

        cursor = deadline_step

        for win_idx, (win_start, win_end) in enumerate(reversed(union_windows)):
            pos = min(win_end, cursor)
            if pos <= win_start:
                continue

            # Satellites visible somewhere in [win_start, pos] with remaining budget
            eligible = [
                s
                for s in schedulable_sats
                if remaining[s] > 0
                and any(ws < pos and we > win_start for ws, we in per_sat_windows[s])
            ]
            if not eligible:
                continue

            # Rotate which satellite gets the latest (most favourable) slot
            offset = win_idx % len(eligible)
            ordered = eligible[offset:] + eligible[:offset]

            for sat in ordered:
                if pos <= win_start or remaining[sat] <= 0:
                    break
                # Latest visibility sub-window for sat within [win_start, pos]
                sat_wins = [
                    (ws, we)
                    for ws, we in per_sat_windows[sat]
                    if ws < pos and we > win_start
                ]
                if not sat_wins:
                    continue
                sw_start, sw_end = max(sat_wins, key=lambda w: w[1])
                end = min(sw_end, pos)
                start = max(end - remaining[sat], sw_start)
                chunk = end - start
                if chunk <= 0:
                    continue
                schedule[gs].append(Slot(sat, start, end))
                remaining[sat] -= chunk
                pos = start - reconfig_steps

            cursor = pos

    return schedule


def build_schedule(
    graphs: list[nx.Graph],
    params: SimulationParams,
    schedulable_sats: list[str],
    slot_budget: SlotBudgetPolicy = equitable_slot_budget,
) -> Schedule:
    """
    Builds a pre-computed downlink schedule using the given slot budget policy.

    Satellites are scheduled latest-first (maximising ISL collection time
    before downlink). Assignment is best-effort: satellites whose full budget
    cannot be filled before the deadline receive as many steps as available.

    Args:
        graphs: connectivity graphs, one per simulation step.
        params: simulation parameters (deadline, rates, reconfig time…).
        schedulable_sats: satellites to schedule (relays or all satellites).
        slot_budget: policy determining the target allocation per satellite.
    """
    if not schedulable_sats:
        return {}

    dt_s = params.dt_s

    sat_nodes, station_nodes = get_nodes_by_type(graphs[0])
    reconfig_steps = math.ceil(params.gs_reconfig_s / dt_s)
    deadline_step = math.ceil(params.deadline_s / dt_s)
    visibility = _compute_visibility(graphs, sat_nodes, station_nodes)
    sat_index = {sat: i for i, sat in enumerate(sat_nodes)}

    slot_budgets = slot_budget(
        schedulable_sats,
        sat_nodes,
        params,
        visibility,
        sat_index,
        station_nodes,
        deadline_step,
        reconfig_steps,
    )

    schedule = _assign_slots(
        schedulable_sats,
        station_nodes,
        visibility,
        sat_index,
        slot_budgets,
        deadline_step,
        reconfig_steps,
    )

    return schedule
