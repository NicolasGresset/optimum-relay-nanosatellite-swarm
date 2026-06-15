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


def _gs_windows(
    schedulable_sats: list[str],
    gs: str,
    visibility: dict[str, np.ndarray],
    sat_index: dict[str, int],
    deadline_step: int,
) -> tuple[dict[str, list[tuple[int, int]]], list[tuple[int, int]]]:
    """Per-satellite visibility windows and their union, for one GS."""
    per_sat = {
        sat: _visibility_windows_fast(visibility[gs][sat_index[sat]], deadline_step)
        for sat in schedulable_sats
    }
    union_vis = np.zeros(deadline_step, dtype=bool)
    for sat in schedulable_sats:
        union_vis |= visibility[gs][sat_index[sat], :deadline_step]
    return per_sat, _visibility_windows_fast(union_vis, deadline_step)


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
# Slot assignment policies
# ---------------------------------------------------------------------------
#
# A SlotAssignmentPolicy decides *when* (within the visibility windows) the
# steps allocated by a SlotBudgetPolicy are placed.
#
# SlotAssignmentPolicy : (schedulable_sats, station_nodes, visibility,
#                         sat_index, slot_budgets, deadline_step,
#                         reconfig_steps)
#                        → Schedule

SlotAssignmentPolicy = Callable[
    [
        list[str],  # schedulable_sats
        frozenset[str],  # station_nodes
        dict[str, np.ndarray],  # visibility[gs] → array(N_sat, N_time, bool)
        dict[str, int],  # sat_index
        dict[str, int],  # slot_budgets
        int,  # deadline_step
        int,  # reconfig_steps
    ],
    Schedule,
]


def round_robin_latest_first_assignment(
    schedulable_sats: list[str],
    station_nodes: frozenset[str],
    visibility: dict[str, np.ndarray],
    sat_index: dict[str, int],
    slot_budgets: dict[str, int],
    deadline_step: int,
    reconfig_steps: int,
) -> Schedule:
    """
    Budget-driven round-robin, latest-first.

    Traverses GS windows from latest to earliest; within each window assigns
    satellites their latest possible slot first. This maximises the crosslink
    collection time available before each downlink.

    Budget: external (slot_budgets). Best-effort: satellites that cannot fill
    their full budget receive as many steps as available.
    Rotation: the satellite that gets the most-favourable (latest) slot
    rotates by one position each window.
    """
    schedule: Schedule = {gs: [] for gs in sorted(station_nodes)}
    remaining = {sat: slot_budgets[sat] for sat in schedulable_sats}

    for gs in sorted(station_nodes):
        per_sat_windows, union_windows = _gs_windows(
            schedulable_sats, gs, visibility, sat_index, deadline_step
        )
        cursor = deadline_step

        for win_idx, (win_start, win_end) in enumerate(reversed(union_windows)):
            pos = min(win_end, cursor)
            if pos <= win_start:
                continue

            eligible = [
                s for s in schedulable_sats
                if remaining[s] > 0
                and any(ws < pos and we > win_start for ws, we in per_sat_windows[s])
            ]
            if not eligible:
                continue

            offset = win_idx % len(eligible)
            ordered = eligible[offset:] + eligible[:offset]

            for sat in ordered:
                if pos <= win_start or remaining[sat] <= 0:
                    break
                sat_wins = [
                    (ws, we) for ws, we in per_sat_windows[sat]
                    if ws < pos and we > win_start
                ]
                if not sat_wins:
                    continue
                sw_start, sw_end = max(sat_wins, key=lambda w: w[1])  # latest sub-window
                end = min(sw_end, pos)
                start = max(end - remaining[sat], sw_start)
                chunk = end - start
                if chunk <= 0:
                    continue
                schedule[gs].append(Slot(sat, start, end))
                remaining[sat] -= chunk
                pos = start - reconfig_steps  # advance backward

            cursor = pos

    return schedule


def round_robin_earliest_first_assignment(
    schedulable_sats: list[str],
    station_nodes: frozenset[str],
    visibility: dict[str, np.ndarray],
    sat_index: dict[str, int],
    slot_budgets: dict[str, int],
    deadline_step: int,
    reconfig_steps: int,
) -> Schedule:
    """
    Budget-driven round-robin, earliest-first.

    Temporal mirror of round_robin_latest_first_assignment: traverses GS
    windows from earliest to latest; within each window assigns satellites
    their earliest possible slot first. This minimises latency at the cost
    of less crosslink collection time.

    Budget: external (slot_budgets). Best-effort: satellites that cannot fill
    their full budget receive as many steps as available.
    Rotation: the satellite that gets the most-favourable (earliest) slot
    rotates by one position each window.
    """
    schedule: Schedule = {gs: [] for gs in sorted(station_nodes)}
    remaining = {sat: slot_budgets[sat] for sat in schedulable_sats}

    for gs in sorted(station_nodes):
        per_sat_windows, union_windows = _gs_windows(
            schedulable_sats, gs, visibility, sat_index, deadline_step
        )
        cursor = 0

        for win_idx, (win_start, win_end) in enumerate(union_windows):
            pos = max(win_start, cursor)
            if pos >= win_end:
                continue

            eligible = [
                s for s in schedulable_sats
                if remaining[s] > 0
                and any(ws < win_end and we > pos for ws, we in per_sat_windows[s])
            ]
            if not eligible:
                continue

            offset = win_idx % len(eligible)
            ordered = eligible[offset:] + eligible[:offset]

            for sat in ordered:
                if pos >= win_end or remaining[sat] <= 0:
                    break
                sat_wins = [
                    (ws, we) for ws, we in per_sat_windows[sat]
                    if ws < win_end and we > pos
                ]
                if not sat_wins:
                    continue
                sw_start, sw_end = min(sat_wins, key=lambda w: w[0])  # earliest sub-window
                start = max(sw_start, pos)
                end = min(start + remaining[sat], sw_end)
                chunk = end - start
                if chunk <= 0:
                    continue
                schedule[gs].append(Slot(sat, start, end))
                remaining[sat] -= chunk
                pos = end + reconfig_steps  # advance forward

            cursor = pos

    return schedule


def pure_round_robin(
    schedulable_sats: list[str],
    station_nodes: frozenset[str],
    visibility: dict[str, np.ndarray],
    sat_index: dict[str, int],
    slot_budgets: dict[str, int],  # unused — kept for SlotAssignmentPolicy compatibility
    deadline_step: int,
    reconfig_steps: int,
) -> Schedule:
    """
    Budget-free round-robin, pass-local equal split.

    Each pass is scheduled independently: the available GS time (pass
    duration minus reconfiguration overhead) is divided equally among all
    eligible satellites. Every eligible satellite is served exactly once per
    pass, earliest slot first, in rotating order.

    Budget: none — allocation is determined entirely by pass duration, number
    of eligible satellites, and reconfiguration time. The number of
    reconfigurations per pass is exactly n_eligible - 1 (minimum for a
    non-fragmenting scheduler).
    Rotation: the satellite that gets the most-favourable (earliest) slot
    rotates by one position each pass.
    """
    schedule: Schedule = {gs: [] for gs in sorted(station_nodes)}

    for gs in sorted(station_nodes):
        per_sat_windows, union_windows = _gs_windows(
            schedulable_sats, gs, visibility, sat_index, deadline_step
        )

        for win_idx, (win_start, win_end) in enumerate(union_windows):
            eligible = [
                s for s in schedulable_sats
                if any(ws < win_end and we > win_start for ws, we in per_sat_windows[s])
            ]
            if not eligible:
                continue

            reconfig_overhead = (len(eligible) - 1) * reconfig_steps
            per_pass = max(1, (win_end - win_start - reconfig_overhead) // len(eligible))

            offset = win_idx % len(eligible)
            ordered = eligible[offset:] + eligible[:offset]

            pos = win_start
            for sat in ordered:
                if pos >= win_end:
                    break
                sat_wins = [
                    (ws, we) for ws, we in per_sat_windows[sat]
                    if ws < win_end and we > win_start
                ]
                if not sat_wins:
                    continue
                sw_start, sw_end = min(sat_wins, key=lambda w: w[0])  # earliest sub-window
                start = max(sw_start, pos)
                end = min(start + per_pass, sw_end, win_end)
                chunk = end - start
                if chunk <= 0:
                    continue
                schedule[gs].append(Slot(sat, start, end))
                pos = end + reconfig_steps  # advance forward

    return schedule


def build_schedule(
    graphs: list[nx.Graph],
    params: SimulationParams,
    schedulable_sats: list[str],
    slot_budget: SlotBudgetPolicy = equitable_slot_budget,
    assignment: SlotAssignmentPolicy = round_robin_latest_first_assignment,
) -> Schedule:
    """
    Builds a pre-computed downlink schedule using the given slot budget and
    assignment policies.

    Assignment is best-effort: satellites whose full budget cannot be filled
    before the deadline receive as many steps as available.

    Args:
        graphs: connectivity graphs, one per simulation step.
        params: simulation parameters (deadline, rates, reconfig time…).
        schedulable_sats: satellites to schedule (relays or all satellites).
        slot_budget: policy determining the target allocation per satellite.
        assignment: policy determining when allocated steps are placed.
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

    schedule = assignment(
        schedulable_sats,
        station_nodes,
        visibility,
        sat_index,
        slot_budgets,
        deadline_step,
        reconfig_steps,
    )

    return schedule
