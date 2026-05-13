"""
routing.py
----------
Routing policy protocols and default implementations for relay-based
data forwarding in a satellite constellation.

RelaySelectPolicy : (sat, G, relay_set, station_nodes) → target relay | None
    Selects which relay a non-relay satellite should route its data toward.

NextHopPolicy : (sat, relay_target, G, relay_set) → next-hop node | None
    Selects the immediate neighbour toward the target relay.
"""

from __future__ import annotations

from typing import Callable

import networkx as nx


RelaySelectPolicy = Callable[
    [str, nx.Graph, frozenset[str], frozenset[str]], str | None
]
NextHopPolicy = Callable[[str, str, nx.Graph, frozenset[str]], str | None]


def nearest_relay(
    sat: str,
    G: nx.Graph,
    relay_set: frozenset[str],
    station_nodes: frozenset[str],
) -> str | None:
    """
    Default relay-select policy: returns the closest relay by BFS hop count
    over the ISL sub-graph (ground stations excluded).
    """
    if sat not in G:
        return None
    for node in nx.bfs_tree(
        G.subgraph([n for n in G.nodes if n not in station_nodes]), sat
    ).nodes():
        if node in relay_set and node != sat:
            return node
    return None


def shortest_path_next_hop(
    sat: str,
    relay_target: str,
    G: nx.Graph,
    relay_set: frozenset[str],
) -> str | None:
    """
    Default next-hop policy: returns the first hop on the shortest path
    (by hop count) between sat and relay_target in G.
    """
    try:
        path = nx.shortest_path(G, sat, relay_target)
        return path[1] if len(path) > 1 else None
    except nx.NetworkXNoPath:
        return None
