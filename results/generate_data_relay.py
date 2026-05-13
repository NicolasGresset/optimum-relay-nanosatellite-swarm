from __future__ import annotations

import csv
import itertools
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RESULTS_DIR))

from src.radio import radio_for_target
from src.params import SimulationParams
from src.formation import SWARM_STDDEVS_30KM
from src.framework import TopologyCache, iter_batch, compute_metrics
from config import (
    SEED, N_SATELLITES,
    BASE_ORBITAL_PARAMS, STATIONS,
    DURATION_S, DT_S, R_MAX_M, ELEV_MIN_DEG,
    DL_RATE_BPS, DL_DISTANCE_M, DL_FREQ_HZ, DL_POWER_TX_W, DL_GAIN_TX_DBI, DL_GAIN_RX_DBI,
    XL_DISTANCE_M, XL_FREQ_HZ, XL_POWER_TX_W, XL_GAIN_TX_DBI, XL_GAIN_RX_DBI,
)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

MAX_COMBINATIONS = 30
MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)

# ---------------------------------------------------------------------------
# Parameter grids
# Each entry: (VOLUMES_MB, GS_RECONFIGS_S, XL_DL_RATIOS, output_subfolder)
# ---------------------------------------------------------------------------

GRIDS: list[tuple[list[int], list[float], list[tuple[int, int]], str]] = [
    (
        [500, 2_000, 5_000, 10_000],  # VOLUMES_MB
        [30.0],  # GS_RECONFIGS_S
        [  # XL_DL_RATIOS
            (1, 4),
        ],
        "D_sweep",
    ),
    (
        [1_000],  # VOLUMES_MB
        [70.0],  # GS_RECONFIGS_S
        [  # XL_DL_RATIOS
            (1, 200),
            (1, 100),
            (1, 50),
            (1, 20),
        ],
        "XL_sweep",
    ),
    (
        [1_000],  # VOLUMES_MB
        [1.0, 10.0, 30.0, 70.0, 100.0],  # GS_RECONFIGS_S
        [  # XL_DL_RATIOS
            (1, 4),
        ],
        "Tr_sweep",
    ),
]

CSV_FIELDS = [
    "k",
    "combo_id",
    "relay_subset",
    "n_satellites",
    "v_bits",
    "seed",
    "gs_reconfig_s",
    "xl_rate_bps",
    "completion_ratio",
    "total_delivered_bits",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(
    v_bits: int, gs_reconfig_s: float, xl_rate_bps: float
) -> SimulationParams:
    downlink = radio_for_target(
        DL_RATE_BPS,
        DL_DISTANCE_M,
        frequency_hz=DL_FREQ_HZ,
        power_tx_w=DL_POWER_TX_W,
        gain_tx_dbi=DL_GAIN_TX_DBI,
        gain_rx_dbi=DL_GAIN_RX_DBI,
    )
    crosslink = radio_for_target(
        xl_rate_bps,
        XL_DISTANCE_M,
        frequency_hz=XL_FREQ_HZ,
        power_tx_w=XL_POWER_TX_W,
        power_rx_w=XL_POWER_TX_W,
        gain_rx_dbi=XL_GAIN_RX_DBI,
        gain_tx_dbi=XL_GAIN_TX_DBI,
    )
    return SimulationParams(
        duration_s=DURATION_S,
        dt_s=DT_S,
        initial_buffer_length_bits=v_bits,
        isl_radio=crosslink,
        downlink_radio=downlink,
        deadline_s=DURATION_S,
        gs_reconfig_s=gs_reconfig_s,
    )


def _csv_name(v_mb: int, gs_s: float, xl_mbps: float) -> str:
    return f"V{v_mb}_GS{gs_s:.0f}s_XL{xl_mbps:.3f}Mbps.csv"


def _sample_combinations(
    sat_names: list[str], k: int, seed: int
) -> list[tuple[str, ...]]:
    all_combos = list(itertools.combinations(sat_names, k))
    random.seed(seed)
    if len(all_combos) > MAX_COMBINATIONS:
        return random.sample(all_combos, MAX_COMBINATIONS)
    return all_combos


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    sat_names = [f"SAT_{i}" for i in range(N_SATELLITES)]

    print(f"Building topology for N={N_SATELLITES}…", end=" ", flush=True)
    t0 = time.perf_counter()
    cache = TopologyCache(
        BASE_ORBITAL_PARAMS,
        STATIONS,
        DURATION_S,
        DT_S,
        R_MAX_M,
        ELEV_MIN_DEG,
        stddevs=SWARM_STDDEVS_30KM,
    )
    _, graphs = cache.get(N_SATELLITES, SEED)
    print(f"done ({time.perf_counter() - t0:.1f} s)")

    # Pre-generate relay subsets once — reused across all grids and configurations
    task_meta: list[tuple[int, int, str, list[str] | None]] = []
    for k in range(1, N_SATELLITES + 1):
        combos = _sample_combinations(sat_names, k, seed=SEED + k)
        for combo_id, combo in enumerate(combos):
            task_meta.append((k, combo_id, ",".join(combo), list(combo)))
    n_tasks = len(task_meta)
    print(
        f"Relay subsets: {n_tasks} tasks per configuration  |  workers: {MAX_WORKERS}"
    )

    for grid_idx, (
        volumes_mb,
        gs_reconfigs_s,
        xl_dl_ratios,
        output_subdir,
    ) in enumerate(GRIDS, start=1):
        output_dir = PROJECT_ROOT / "data" / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        existing_csvs = list(output_dir.glob("*.csv"))
        if existing_csvs:
            for p in existing_csvs:
                p.unlink()
            print(f"  Removed {len(existing_csvs)} existing CSV(s) from '{output_subdir}'")

        param_grid = list(itertools.product(volumes_mb, gs_reconfigs_s, xl_dl_ratios))
        n_configs = len(param_grid)
        xl_mbps_values = [
            f"{DL_RATE_BPS * num / den / 1e6:.2g}" for num, den in xl_dl_ratios
        ]
        print(
            f"\n{'='*60}\n"
            f"Grid {grid_idx}/{len(GRIDS)}: '{output_subdir}'\n"
            f"  Volumes   : {volumes_mb} MB\n"
            f"  GS reconf : {gs_reconfigs_s} s\n"
            f"  XL rates  : {xl_mbps_values} Mbps\n"
            f"  Total     : {n_configs} configs × {n_tasks} tasks"
            f" = {n_configs * n_tasks} simulations\n"
            f"{'='*60}"
        )

        for cfg_idx, (v_mb, gs_s, (xl_num, xl_den)) in enumerate(param_grid, start=1):
            v_bits = int(v_mb * 8e6)
            xl_mbps = DL_RATE_BPS * xl_num / xl_den / 1e6
            xl_bps = xl_mbps * 1e6
            out_path = output_dir / _csv_name(v_mb, gs_s, xl_mbps)

            print(
                f"\n[{cfg_idx}/{n_configs}] V={v_mb} MB  GS={gs_s:.0f} s"
                f"  XL={xl_mbps:.3g} Mbps",
                flush=True,
            )

            params = _make_params(v_bits, gs_s, xl_bps)
            tasks = [(params, relay_nodes) for _, _, _, relay_nodes in task_meta]

            t_cfg = time.perf_counter()
            completed = 0

            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                f.flush()

                for task_idx, result in iter_batch(
                    graphs, tasks, max_workers=MAX_WORKERS
                ):
                    completed += 1
                    k, combo_id, combo_str, _ = task_meta[task_idx]
                    metrics = compute_metrics(result, N_SATELLITES, v_bits)
                    writer.writerow(
                        {
                            "k": k,
                            "combo_id": combo_id,
                            "relay_subset": combo_str,
                            "n_satellites": N_SATELLITES,
                            "v_bits": v_bits,
                            "seed": SEED,
                            "gs_reconfig_s": gs_s,
                            "xl_rate_bps": xl_bps,
                            **metrics,
                        }
                    )
                    f.flush()
                    print(f"  {completed}/{n_tasks}", end="\r", flush=True)

            print(
                f"  {n_tasks}/{n_tasks}  done"
                f" ({time.perf_counter() - t_cfg:.1f} s)"
                f"  →  {out_path.name}"
            )

    print("\nAll grids complete.")


if __name__ == "__main__":
    run_experiment()
