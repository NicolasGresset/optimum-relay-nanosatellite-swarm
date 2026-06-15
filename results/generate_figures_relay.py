"""
generate_figures.py
-------------------
For each configured grid subdirectory, loads all CSV files and produces a
completion-rate figure.  Two figure styles are supported:

  - standard : mean ± std / min-max / box-plot  (fig_mean_std)
  - xl_sweep : same + optimal k* trajectory      (fig_xl_sweep)

Configuration: edit GRID_CONFIGS below.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RESULTS_DIR))

from style import (
    COLORS as _COLORS,
    LINESTYLES as _LINESTYLES,
    MARKERS as _MARKERS,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Grid figure configuration
# ---------------------------------------------------------------------------


@dataclass
class GridFigConfig:
    """Describes how to produce a figure for one results subdirectory.

    Args:
        subdir:   Subdirectory name under ``results/`` that contains the CSVs.
        fig_name: Output figure filename (saved alongside the CSVs).
        caption:  Callable that receives the first row of a CSV (a dict with
                  all numeric fields already cast) and returns the curve label.
        xl_sweep: When True, overlay the optimal k* trajectory (fig_xl_sweep).
    """

    subdir: str
    fig_name: str
    caption: Callable[[dict], str]
    legend_fontsize: float = 4.5
    xl_sweep: bool = False
    legend_loc: str | tuple[float, float] = "best"
    sort_key: Callable[[dict], float] | None = None


GRID_CONFIGS: list[GridFigConfig] = [
    GridFigConfig(
        subdir="XL_sweep",
        fig_name="figure_6.pdf",
        caption=lambda row: rf"$C_{{ISL}}$ = {float(row['xl_rate_bps']) / 1e6:.1f} Mbps",
        xl_sweep=True,
        legend_fontsize=6,
        sort_key=lambda row: row["xl_rate_bps"],
    ),
    GridFigConfig(
        subdir="Tr_sweep",
        fig_name="figure_7.pdf",
        caption=lambda row: rf"$T_r$ = {int(row['gs_reconfig_s'])} s",
        legend_fontsize=6,
        sort_key=lambda row: row["gs_reconfig_s"],
    ),
]

POLICY_LABELS: dict[str, str] = {
    "pure_round_robin": "Round robin",
    "precomputed_latest_first": "Latest-first",
    "precomputed_earliest_first": "Earliest-first",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["k"] = int(row["k"])
            row["n_satellites"] = int(row["n_satellites"])
            row["v_bits"] = int(float(row["v_bits"]))
            row["completion_ratio"] = float(row["completion_ratio"])
            row["total_delivered_bits"] = int(row["total_delivered_bits"])
            row["gs_reconfig_s"] = float(row["gs_reconfig_s"])
            row["xl_rate_bps"] = float(row["xl_rate_bps"])
            rows.append(row)
    return rows


def _group_by_k(rows: list[dict]) -> dict[int, list[float]]:
    groups: dict[int, list[float]] = {}
    for row in rows:
        groups.setdefault(row["k"], []).append(row["completion_ratio"])
    return {k: groups[k] for k in sorted(groups)}


def _k_star(rows: list[dict]) -> int:
    """Return the k whose average completion rate is maximal."""
    rates_by_k: dict[int, list[float]] = {}
    for row in rows:
        rates_by_k.setdefault(row["k"], []).append(row["completion_ratio"])
    avg = {k: sum(v) / len(v) for k, v in rates_by_k.items()}
    return max(avg, key=avg.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Core plotting primitive
# ---------------------------------------------------------------------------


def _draw_r_vs_k(
    ax: plt.Axes,
    datasets: list[tuple[str, list[dict]]],
    colors: list,
) -> None:
    """Draw completion ratio vs k onto *ax*. Does not add legend or colorbar."""
    all_ks = sorted({row["k"] for _, rows in datasets for row in rows if row["k"] > 0})

    for ds_idx, ((label, rows), color) in enumerate(zip(datasets, colors)):
        ls = _LINESTYLES[ds_idx % len(_LINESTYLES)]
        marker = _MARKERS[ds_idx % len(_MARKERS)]
        groups = _group_by_k(rows)
        relay_ks = [k for k in sorted(groups) if k > 0]
        means = np.array([np.mean(groups[k]) for k in relay_ks])
        stds = np.array([np.std(groups[k]) for k in relay_ks])
        lo = np.clip(means - stds, 0, 1)
        hi = np.clip(means + stds, 0, 1)
        yerr = np.vstack(
            [
                means - lo,
                hi - means,
            ]
        )

        ax.fill_between(relay_ks, lo, hi, alpha=0.12, color=color)
        ax.plot(
            relay_ks,
            means,
            linestyle=ls,
            marker=marker,
            color=color,
            linewidth=1,
            markersize=2,
            label=label,
        )
        ax.errorbar(
            relay_ks,
            means,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=0.5,
            capsize=2,
            alpha=0.6,
        )

    ax.set_xlabel(r"Number of relay satellites $k$")
    ax.set_ylabel(r"Completion ratio $r$")
    ax.tick_params(axis="both")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(all_ks)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure styles
# ---------------------------------------------------------------------------


def fig_mean_std(
    datasets: list[tuple[str, list[dict]]],
    *,
    out: Path,
    legend_loc: str | tuple[float, float] = "best",
    legend_fontsize: float = 4.5,
) -> None:
    """Standard figure: mean ± std per k, one curve per dataset."""
    all_ks = sorted({row["k"] for _, rows in datasets for row in rows if row["k"] > 0})
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    _draw_r_vs_k(ax, datasets, _COLORS)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


def fig_xl_sweep(
    datasets: list[tuple[str, list[dict]]],
    *,
    out: Path,
    legend_loc: str | tuple[float, float] = "best",
    legend_fontsize: float = 4.5,
) -> None:
    """XL-sweep figure: same as fig_mean_std plus the optimal k* trajectory."""
    all_ks = sorted({row["k"] for _, rows in datasets for row in rows if row["k"] > 0})
    # fig, ax = plt.subplots(figsize=(max(8, len(all_ks) * 0.8), 5))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))

    _draw_r_vs_k(ax, datasets, _COLORS)

    k_star_xs, k_star_ys = [], []
    for (_, rows), color in zip(datasets, _COLORS):
        ks = _k_star(rows)
        r_at_ks = np.mean(_group_by_k(rows).get(ks, [float("nan")]))
        k_star_xs.append(ks)
        k_star_ys.append(r_at_ks)
        ax.scatter(
            ks,
            r_at_ks,
            marker="*",
            s=30,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    ax.plot(
        k_star_xs,
        k_star_ys,
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.75,
        zorder=4,
        label=r"$k^*$",
    )

    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    policy_csv_dir = PROJECT_ROOT / "data" / "downlink_policy_sweep"
    policy_out = PROJECT_ROOT / "figures" / "figure_5.pdf"
    policy_csv_files = sorted(policy_csv_dir.glob("*.csv"))
    if not policy_csv_files:
        print("[downlink_policy_sweep] No CSV files found, skipping.")
    else:
        print(f"\n[downlink_policy_sweep] Loading {len(policy_csv_files)} file(s)…")
        datasets: list[tuple[str, list[dict]]] = []
        for path in policy_csv_files:
            rows = load_results(path)
            label = POLICY_LABELS.get(path.stem, path.stem)
            datasets.append((label, rows))
            print(f"  {path.name} → {label!r}")
        fig_mean_std(datasets, out=policy_out, legend_loc=(0.5, 0.2), legend_fontsize=6)

    for config in GRID_CONFIGS:
        csv_dir = PROJECT_ROOT / "data" / config.subdir
        out_path = PROJECT_ROOT / "figures" / config.fig_name

        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            print(f"[{config.subdir}] No CSV files found, skipping.")
            continue

        print(f"\n[{config.subdir}] Loading {len(csv_files)} file(s)…")
        datasets: list[tuple[str, list[dict]]] = []
        for path in csv_files:
            rows = load_results(path)
            label = config.caption(rows[0])
            datasets.append((label, rows))
            print(f"  {path.name} → {label!r}")

        if config.sort_key is not None:
            datasets.sort(key=lambda lr: config.sort_key(lr[1][0]))  # type: ignore[misc]

        if config.xl_sweep:
            fig_xl_sweep(
                datasets,
                out=out_path,
                legend_loc=config.legend_loc,
                legend_fontsize=config.legend_fontsize,
            )
        else:
            fig_mean_std(
                datasets,
                out=out_path,
                legend_loc=config.legend_loc,
                legend_fontsize=config.legend_fontsize,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
