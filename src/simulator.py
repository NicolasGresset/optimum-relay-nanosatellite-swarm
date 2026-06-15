"""
simulator.py
------------
Core simulation engine: data transfer model and Simulator class.
"""

from __future__ import annotations
from itertools import chain

import networkx as nx

from .network import get_nodes_by_type
from .params import SimulationParams, SimulationResult
from .radio import link_capacity_for_edge
from .routing import (
    RelaySelectPolicy,
    NextHopPolicy,
    nearest_relay,
    shortest_path_next_hop,
)
from .scheduling import (
    Slot,
    SlotAssignmentPolicy,
    SlotBudgetPolicy,
    build_schedule,
    equitable_slot_budget,
    round_robin_latest_first_assignment,
)

# ---------------------------------------------------------------------------
# Data transfer primitives
# ---------------------------------------------------------------------------


def _effective_capacity(
    capacity_bps: float,
    n_concurrent_senders: int,
    dt_s: float,
) -> float:
    """Transferable bits over dt_s with fair sharing among n senders."""
    if n_concurrent_senders <= 0:
        return 0.0
    return capacity_bps / n_concurrent_senders * dt_s


def _transfer_data(
    sender: str,
    receiver: str,
    available_bits: float,
    capacity_bps: float,
    n_concurrent_senders: int,
    dt_s: float,
    buffers: dict[str, float],
) -> None:
    """
    Transfers data from sender to receiver.

    Updates buffers in-place.
    """
    transferable = _effective_capacity(capacity_bps, n_concurrent_senders, dt_s)
    transferred = min(available_bits, transferable)
    if transferred <= 0.0:
        return

    buffers[sender] -= transferred
    buffers[receiver] += transferred


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class Simulator:
    """
    Simulates nanosatellite swarm downlink over a pre-computed sequence of
    connectivity graphs (one graph per time step).

    Each time step has two phases:
      1. Crosslink — non-relay satellites route buffered data toward their
         nearest relay over ISLs (no-op in direct mode).
      2. Downlink — relay satellites transmit to the ground station following
         a pre-computed schedule built before the simulation starts.

    Results are collected as per-node buffer histories and summarised by
    ``compute_metrics`` in ``framework.py`` (completion ratio, delivered bits).

    Parameters
    ----------
    params       : simulation parameters (radio, duration, deadline…).
    relay_nodes  : relay satellites. ``None`` → direct mode (all satellites
                   relay; crosslink phase is a no-op).
    relay_select : policy selecting the target relay for a given satellite.
    next_hop     : policy selecting the next-hop toward the relay.
    slot_budget  : policy allocating GS slots per relay (relay mode only).
    assignment   : policy deciding when allocated GS slots are placed
                   (relay mode only).
    """

    def __init__(
        self,
        params: SimulationParams,
        relay_nodes: list[str] | None = None,
        relay_select: RelaySelectPolicy = nearest_relay,
        next_hop: NextHopPolicy = shortest_path_next_hop,
        slot_budget: SlotBudgetPolicy = equitable_slot_budget,
        assignment: SlotAssignmentPolicy = round_robin_latest_first_assignment,
    ) -> None:
        self.params = params
        self._relay_nodes = relay_nodes
        self._relay_select = relay_select
        self._next_hop = next_hop
        self._slot_budget = slot_budget
        self._assignment = assignment

    def run(self, graphs: list[nx.Graph]) -> SimulationResult:
        """Run the simulation and return aggregated results."""
        self._setup(graphs)
        for step_index, G in enumerate(graphs):
            self._step(G, step_index)
        return self._collect_result()

    # ---- Setup ----

    def _setup(self, graphs: list[nx.Graph]) -> None:
        self._sat_nodes, self._station_nodes = get_nodes_by_type(graphs[0])
        self._strategy = "relay" if self._relay_nodes is not None else "direct"
        self._relay_set = (
            frozenset(self._relay_nodes)
            if self._relay_nodes is not None
            else frozenset(self._sat_nodes)
        )
        self._schedule = build_schedule(
            graphs,
            self.params,
            sorted(self._relay_set, key=lambda x: int(x.split("_")[1])),
            slot_budget=self._slot_budget,
            assignment=self._assignment,
        )
        self._step_to_slot: dict[str, dict[int, Slot]] = {
            gs: {}
            for gs in sorted(self._station_nodes, key=lambda x: int(x.split("_")[1]))
        }
        for gs, slots in self._schedule.items():
            for slot in slots:
                for step in range(slot.step_start, slot.step_end):
                    self._step_to_slot[gs][step] = slot
        self._buffers: dict[str, float] = {
            **{node: float(self.params.initial_buffer_length_bits) for node in self._sat_nodes},
            **{node: 0.0 for node in self._station_nodes},
        }
        self._buffer_history: dict[str, list[float]] = {
            node: [] for node in chain(self._sat_nodes, self._station_nodes)
        }

    def _collect_result(self) -> SimulationResult:
        return SimulationResult(
            buffer_history=self._buffer_history,
        )

    # ---- Simulation loop ----

    def _step(self, G: nx.Graph, step_index: int) -> None:
        self._crosslink_step(G)
        self._downlink_step(G, step_index)
        for node in chain(self._sat_nodes, self._station_nodes):
            self._buffer_history[node].append(self._buffers.get(node, 0.0))

    def _crosslink_step(self, G: nx.Graph) -> None:
        """
        Multi-hop crosslink phase: routes data from non-relays toward their
        nearest relay, one hop at a time, with per-receiver TDMA sharing.

        No-op in direct mode (relay_set == sat_nodes → forwarders is empty).
        """
        senders_per_hop: dict[str, list[str]] = {}
        G_isl = G.subgraph([n for n in G.nodes if n not in self._station_nodes])
        forwarders = [
            n
            for n in G.nodes
            if n not in self._relay_set
            and n not in self._station_nodes
            and self._buffers.get(n, 0.0) > 0.0
        ]
        for sat in forwarders:
            target = self._relay_select(sat, G, self._relay_set, self._station_nodes)
            if target is None:
                continue
            hop = self._next_hop(sat, target, G_isl, self._relay_set)
            if hop is None:
                continue
            senders_per_hop.setdefault(hop, []).append(sat)

        for receiver, senders in senders_per_hop.items():
            n_senders = len(senders)
            for sender in senders:
                capacity = link_capacity_for_edge(
                    G,
                    sender,
                    receiver,
                    self._station_nodes,
                    self.params.isl_radio,
                    self.params.downlink_radio,
                )
                _transfer_data(
                    sender=sender,
                    receiver=receiver,
                    available_bits=self._buffers.get(sender, 0.0),
                    capacity_bps=capacity,
                    n_concurrent_senders=n_senders,
                    dt_s=self.params.dt_s,
                    buffers=self._buffers,
                )

    def _downlink_step(self, G: nx.Graph, step_index: int) -> None:
        """Downlink phase: follows the pre-computed schedule for all GS."""
        for gs in self._step_to_slot:
            self._gs_downlink(gs, step_index, G)

    def _gs_downlink(self, gs: str, step_index: int, G: nx.Graph) -> None:
        """Downlink for one GS over one timestep."""
        slot = self._step_to_slot[gs].get(step_index)
        if slot is None:
            return
        sat = slot.satellite
        if not (G.has_edge(sat, gs) or G.has_edge(gs, sat)):
            return  # satellite left visibility (rare edge case)
        capacity = link_capacity_for_edge(
            G,
            sat,
            gs,
            self._station_nodes,
            self.params.isl_radio,
            self.params.downlink_radio,
        )
        _transfer_data(
            sender=sat,
            receiver=gs,
            available_bits=self._buffers.get(sat, 0.0),
            capacity_bps=capacity,
            n_concurrent_senders=1,
            dt_s=self.params.dt_s,
            buffers=self._buffers,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def simulate(
    graphs: list[nx.Graph],
    params: SimulationParams,
    relay_nodes: list[str] | None = None,
    relay_select: RelaySelectPolicy = nearest_relay,
    next_hop: NextHopPolicy = shortest_path_next_hop,
    slot_budget: SlotBudgetPolicy = equitable_slot_budget,
    assignment: SlotAssignmentPolicy = round_robin_latest_first_assignment,
) -> SimulationResult:
    """Functional shortcut: creates a Simulator and calls run()."""
    return Simulator(params, relay_nodes, relay_select, next_hop, slot_budget, assignment).run(graphs)
