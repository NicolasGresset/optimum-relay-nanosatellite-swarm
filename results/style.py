"""
Shared matplotlib style for all generate_figures_* scripts.

Import this module before any plt call (rcParams are applied at import time).
"""

from __future__ import annotations

import matplotlib as mpl

mpl.rcParams.update(
    {
        # Figure
        "figure.figsize": (3.3, 2.1),
        # Fonts
        "font.size": 8,
        "pdf.fonttype": 42,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 4.5,
        # Lines
        "lines.linewidth": 1,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.8,
        # Axes
        "axes.linewidth": 0.8,
        # Grid
        # "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Save
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

COLORS: list[str] = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]

LINESTYLES: list = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), "--", "-."]
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "h"]
