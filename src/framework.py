"""
framework.py
------------
Shared utilities for simulation experiments: topology caching,
parallel simulation, and standard metric extraction.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orbit import OrbitalParameters
from src.formation import generate_swarm_random_orbital_parameters, SWARM_STDDEVS_30KM
from src.network import simulate_network
from src.params import SimulationParams, SimulationResult
from src.simulator import simulate as _simulate

# ---------------------------------------------------------------------------
# Topology cache
# ---------------------------------------------------------------------------


class TopologyCache:
    """Builds and caches connectivity graphs keyed by (n_satellites, seed)."""

    def __init__(
        self,
        base_params: OrbitalParameters,
        stations: list,
        duration_s: float,
        dt_s: float,
        r_max_m: float,
        elev_min_deg: float,
        stddevs: dict | None = None,
    ) -> None:
        self._base_params = base_params
        self._stations = stations
        self._duration_s = duration_s
        self._dt_s = dt_s
        self._r_max_m = r_max_m
        self._elev_min_deg = elev_min_deg
        self._stddevs = stddevs if stddevs is not None else SWARM_STDDEVS_30KM
        self._store: dict[tuple[int, int], tuple[list, list]] = {}

    def get(self, n_satellites: int, seed: int) -> tuple[list, list]:
        """Return (orbital_params, graphs), building on first call."""
        key = (n_satellites, seed)
        if key not in self._store:
            orbital_params = generate_swarm_random_orbital_parameters(
                n_satellites, self._base_params, stddevs=self._stddevs, seed=seed
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                graphs, _ = simulate_network(
                    orbital_params,
                    self._stations,
                    duration=self._duration_s,
                    time_step_s=self._dt_s,
                    R_max=self._r_max_m,
                    elev_min_deg=self._elev_min_deg,
                )
            self._store[key] = (orbital_params, graphs)
        return self._store[key]


# ---------------------------------------------------------------------------
# Parallel simulation
# ---------------------------------------------------------------------------

# Module-level store used by pool workers.  Graphs are sent once per worker
# via Pool initializer — not once per task — to avoid repeated serialization.
_worker_graphs: list | None = None


def _worker_init(graphs: list) -> None:
    global _worker_graphs
    _worker_graphs = graphs


Task = tuple[SimulationParams, "list[str] | None"] | tuple[SimulationParams, "list[str] | None", dict]


def _split_task(task: Task) -> tuple[SimulationParams, list[str] | None, dict]:
    """Normalizes a task tuple into (params, relay_nodes, extra_kwargs)."""
    if len(task) == 3:
        params, relay_nodes, extra_kwargs = task
        return params, relay_nodes, extra_kwargs
    params, relay_nodes = task
    return params, relay_nodes, {}


def _worker_run(
    task: tuple[int, SimulationParams, list[str] | None, dict],
) -> tuple[int, SimulationResult]:
    idx, params, relay_nodes, extra_kwargs = task
    assert _worker_graphs is not None
    buf = io.StringIO()
    with redirect_stdout(buf):
        return idx, _simulate(_worker_graphs, params, relay_nodes, **extra_kwargs)


def run_batch(
    graphs: list,
    tasks: list[Task],
    max_workers: int = 4,
) -> list[SimulationResult]:
    """
    Run multiple simulations on the same graph set in parallel.

    Args:
        graphs: Pre-built connectivity graphs (shared across all tasks).
        tasks: List of (SimulationParams, relay_nodes) or
            (SimulationParams, relay_nodes, extra_kwargs) tuples, where
            extra_kwargs is passed through to ``simulate`` (e.g. assignment).
        max_workers: Number of worker processes. 1 disables multiprocessing.

    Returns:
        List of SimulationResult in the same order as tasks.
    """
    n = len(tasks)

    if max_workers <= 1 or n == 1:
        results = []
        for i, task in enumerate(tasks):
            params, relay_nodes, extra_kwargs = _split_task(task)
            buf = io.StringIO()
            with redirect_stdout(buf):
                results.append(_simulate(graphs, params, relay_nodes, **extra_kwargs))
            print(f"  [{i + 1}/{n}]", end="\r", flush=True)
        print()
        return results

    indexed = [
        (i, *_split_task(task)) for i, task in enumerate(tasks)
    ]
    collected: list[tuple[int, SimulationResult]] = []

    with Pool(
        processes=max_workers,
        initializer=_worker_init,
        initargs=(graphs,),
    ) as pool:
        for idx, result in pool.imap_unordered(_worker_run, indexed):
            collected.append((idx, result))
            print(f"  [{len(collected)}/{n}]", end="\r", flush=True)

    print()
    return [r for _, r in sorted(collected)]


def iter_batch(
    graphs: list,
    tasks: list[Task],
    max_workers: int = 4,
) -> Iterator[tuple[int, SimulationResult]]:
    """Yield (original_index, result) as simulations complete (fastest first).

    Args:
        graphs: Pre-built connectivity graphs shared across all tasks.
        tasks: List of (SimulationParams, relay_nodes) or
            (SimulationParams, relay_nodes, extra_kwargs) tuples, where
            extra_kwargs is passed through to ``simulate`` (e.g. assignment).
        max_workers: Number of worker processes. 1 disables multiprocessing.

    Yields:
        (original_index, SimulationResult) pairs in completion order.
    """
    n = len(tasks)
    if max_workers <= 1 or n == 1:
        for i, task in enumerate(tasks):
            params, relay_nodes, extra_kwargs = _split_task(task)
            buf = io.StringIO()
            with redirect_stdout(buf):
                yield i, _simulate(graphs, params, relay_nodes, **extra_kwargs)
        return

    indexed = [
        (i, *_split_task(task)) for i, task in enumerate(tasks)
    ]
    with Pool(
        processes=max_workers, initializer=_worker_init, initargs=(graphs,)
    ) as pool:
        yield from pool.imap_unordered(_worker_run, indexed)


def compute_metrics(
    result: SimulationResult,
    n_satellites: int,
    v_bits: int,
) -> dict[str, Any]:
    """
    Extract standard scalar metrics from a SimulationResult.

    Returns a dict with keys:
      completion_ratio    : fraction of total bits delivered (0–1)
      total_delivered_bits
    """
    total_bits = n_satellites * v_bits
    total_delivered_bits = result.buffer_history["GS_0"][-1]
    comp_rate = total_delivered_bits / total_bits if total_bits > 0 else 0.0

    return {
        "completion_ratio": round(comp_rate, 6),
        "total_delivered_bits": int(total_delivered_bits),
    }
