"""
network.py
----------
Network-level simulation pipeline: orbital position computation,
connectivity graph construction, and simulation orchestration.
"""

from __future__ import annotations

import math

import numpy as np
import networkx as nx
from tqdm import tqdm

from . import constants
from .orbit import (
    OrbitalParameters,
    ground_station_ecef,
    enu_matrix,
    compute_all_positions,
)

# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------


def get_nodes_by_type(G: nx.Graph) -> tuple[list[str], frozenset[str]]:
    sat_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "satellite"]
    station_nodes = frozenset(
        n for n, d in G.nodes(data=True) if d.get("type") == "ground"
    )
    return sat_nodes, station_nodes


# ---------------------------------------------------------------------------
# Station preprocessing
# ---------------------------------------------------------------------------


def _precompute_station_data(
    stations,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Precomputes per-station data that does not change over time:
      - ECEF position of the station
      - ENU matrix (depends only on lat/lon)

    Returns a list of (gs_ecef, enu_matrix) in the same order as stations.
    """
    return [
        (ground_station_ecef(lat, lon, alt), enu_matrix(lat, lon))
        for lat, lon, alt in stations
    ]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph(positions, station_data, R_max, elev_min_rad, t_idx):
    G = nx.Graph()

    N_sat = positions.shape[0]

    for i in range(N_sat):
        G.add_node(f"SAT_{i}", type="satellite")
    for id_station in range(len(station_data)):
        G.add_node(f"GS_{id_station}", type="ground")

    pos_t = positions[:, t_idx]  # (N_sat, 3)

    # ISL — vectorised distances: one numpy op instead of N²/2 norm() calls
    diff = pos_t[:, None, :] - pos_t[None, :, :]  # (N, N, 3)
    sq_dists = np.einsum("ijk,ijk->ij", diff, diff)  # (N, N)
    i_idx, j_idx = np.where(np.triu(sq_dists <= R_max**2, k=1))
    for i, j in zip(i_idx, j_idx):
        G.add_edge(f"SAT_{i}", f"SAT_{j}", weight=float(sq_dists[i, j]))

    # GS links — vectorised elevation over all satellites at once
    # https://ntrs.nasa.gov/api/citations/19660001054/downloads/19660001054.pdf
    for id_station, (gs_ecef, enu_mat) in enumerate(station_data):
        diffs = pos_t - gs_ecef  # (N_sat, 3)
        enu = diffs @ enu_mat.T  # (N_sat, 3)
        elevations = np.arctan2(enu[:, 2], np.sqrt(enu[:, 0] ** 2 + enu[:, 1] ** 2))
        sq_dists_gs = np.einsum("ij,ij->i", diffs, diffs)  # (N_sat,)
        for id_sat in np.where(elevations >= elev_min_rad)[0]:
            G.add_edge(
                f"SAT_{id_sat}", f"GS_{id_station}", weight=float(sq_dists_gs[id_sat])
            )

    return G


def simulate_network(
    orbital_parameters_list: list[OrbitalParameters],
    stations,
    duration,
    time_step_s,
    R_max,
    elev_min_deg,
):
    """
    Compute satellite positions and build connectivity graphs for each timestep.

    Returns
    -------
    graphs : list[nx.Graph]
    positions : np.ndarray, shape (N_sat, N_time, 3)
    """
    N_time = int(duration / time_step_s) + 1

    elev_min_rad = math.radians(elev_min_deg)
    station_data = _precompute_station_data(stations)
    positions = compute_all_positions(orbital_parameters_list, N_time, time_step_s)

    graphs = [
        _build_graph(positions, station_data, R_max, elev_min_rad, t_idx)
        for t_idx in tqdm(range(N_time), desc="Building graphs")
    ]

    return graphs, positions
