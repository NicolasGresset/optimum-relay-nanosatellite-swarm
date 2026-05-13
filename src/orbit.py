from dataclasses import dataclass
import math

import numpy as np

from . import constants


@dataclass
class OrbitalParameters:
    a: float  # semi-major axis [m]
    e: float  # excentricity
    inc: float  # inclination [rad]
    Omega: float  # longitude of the ascending node [rad]
    w: float  # argument of periapsis [rad]
    M_0: float  # position at t0 [rad]


def ground_station_ecef(lat, lon, alt):
    r = constants.R_EARTH + alt
    return np.array(
        [
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        ]
    )


def rotation_matrix_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def enu_matrix(lat, lon):
    return np.array(
        [
            [-math.sin(lon), math.cos(lon), 0],
            [
                -math.sin(lat) * math.cos(lon),
                -math.sin(lat) * math.sin(lon),
                math.cos(lat),
            ],
            [
                math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat),
            ],
        ]
    )


def compute_all_positions(
    orbital_params_list: list[OrbitalParameters],
    N_time: int,
    time_step_s: float,
) -> np.ndarray:
    """
    Vectorises position computation to (N_sat, N_time, 3) in a single numpy pass.

    Solves Kepler's equation via vectorised Newton-Raphson over the full
    (N_sat × N_time) array, then chains perifocal → ECI (Q per satellite)
    and ECI → ECEF (Rz(-θ) per timestep) without any Python loop.
    """
    times = np.arange(N_time) * time_step_s

    a = np.array([p.a for p in orbital_params_list])
    e = np.array([p.e for p in orbital_params_list])
    M_0 = np.array([p.M_0 for p in orbital_params_list])
    n = np.sqrt(constants.MU_EARTH / a**3)

    def _rx(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    Qs = np.stack(
        [
            rotation_matrix_z(p.Omega) @ _rx(p.inc) @ rotation_matrix_z(p.w)
            for p in orbital_params_list
        ]
    )  # (N_sat, 3, 3)

    M = (M_0[:, None] + n[:, None] * times[None, :]) % (2 * math.pi)

    E = M.copy()
    for _ in range(50):  # safety cap; converges in <5 iterations for e < 0.1
        dE = -(E - e[:, None] * np.sin(E) - M) / (1.0 - e[:, None] * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < 1e-10:  # ~0.06 mm precision at LEO altitudes
            break

    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e[:, None]) * np.sin(E / 2.0),
        np.sqrt(1.0 - e[:, None]) * np.cos(E / 2.0),
    )
    r = a[:, None] * (1.0 - e[:, None] * np.cos(E))

    r_pf = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(r)], axis=-1)

    r_eci = np.einsum("sij,stj->sti", Qs, r_pf)

    theta = constants.OMEGA_EARTH * times
    cos_t = np.cos(theta)[None, :]
    sin_t = np.sin(theta)[None, :]

    x = cos_t * r_eci[:, :, 0] + sin_t * r_eci[:, :, 1]
    y = -sin_t * r_eci[:, :, 0] + cos_t * r_eci[:, :, 1]
    z = r_eci[:, :, 2]

    return np.stack([x, y, z], axis=-1)
