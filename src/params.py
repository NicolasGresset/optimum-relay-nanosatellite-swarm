"""
params.py
---------
Simulation parameter and result dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .radio import RadioParams


@dataclass
class SimulationParams:
    """Global simulation parameters."""

    duration_s: float
    dt_s: float
    initial_buffer_length_bits: int
    isl_radio: RadioParams
    downlink_radio: RadioParams
    deadline_s: float
    gs_reconfig_s: float = 70.0


@dataclass
class SimulationResult:
    """Aggregated results of a simulation run."""

    buffer_history: dict[str, list[float]]
