"""
formation.py
------------
Generators for orbital parameter sets describing groups of satellites
(swarm, train, and related orbital geometry helpers).
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .orbit import OrbitalParameters

SWARM_STDDEVS_30KM: dict = {
    "a": 0.0,
    "e": 0.0015,
    "inc": np.deg2rad(0.1),
    "Omega": np.deg2rad(0.1),
    "w": np.deg2rad(0.1),
    "M_0": np.deg2rad(0.0),
}

SWARM_STDDEVS_1000KM: dict = {
    "a": 0.0,
    "e": 0.05,
    "inc": np.deg2rad(3.3),
    "Omega": np.deg2rad(3.3),
    "w": np.deg2rad(3.3),
    "M_0": np.deg2rad(0.0),
}


def generate_swarm_random_orbital_parameters(
    n_satellites: int,
    initial_parameters: OrbitalParameters,
    stddevs: dict | None = None,
    seed=None,
) -> list[OrbitalParameters]:
    """
    Generates n_satellites orbital parameters according to a normal
    distribution N(mean_parameters, stddev), optionally using a random seed.

    Args:
        n_satellites: Number of satellites to generate.
        initial_parameters: Centre of the normal distribution.
        stddevs: Per-parameter standard deviations. Defaults to
            SWARM_STDDEVS_30KM. Use SWARM_STDDEVS_1000KM for a 1000 km ISL
            range with equivalent connectivity dynamics.
        seed: Optional seed for numpy's global RNG.
    """
    if stddevs is None:
        stddevs = SWARM_STDDEVS_30KM

    if seed is not None:
        np.random.seed(seed)
    swarm_orbital_parameters = []
    for _ in range(n_satellites):
        orbital_parameter = OrbitalParameters(
            a=np.random.randn() * stddevs.get("a", 0.0) + initial_parameters.a,
            e=np.random.randn() * stddevs.get("e", 0.0) + initial_parameters.e,
            inc=np.random.randn() * stddevs.get("inc", 0.0) + initial_parameters.inc,
            Omega=np.random.randn() * stddevs.get("Omega", 0.0)
            + initial_parameters.Omega,
            w=np.random.randn() * stddevs.get("w", 0.0) + initial_parameters.w,
            M_0=np.random.randn() * stddevs.get("M_0", 0.0) + initial_parameters.M_0,
        )
        swarm_orbital_parameters.append(orbital_parameter)
    return swarm_orbital_parameters


def generate_train_orbital_parameters(
    n_satellites: int, initial_parameters: OrbitalParameters, step_degree: float
) -> list[OrbitalParameters]:
    """
    Generates orbital parameters for a train of satellites evenly spaced
    along a shared orbit, separated by step_degree in mean anomaly.
    """
    train_orbital_parameters = []

    for i in range(n_satellites):
        parameters = copy.deepcopy(initial_parameters)
        parameters.M_0 += math.radians(i * step_degree)
        train_orbital_parameters.append(parameters)

    return train_orbital_parameters
