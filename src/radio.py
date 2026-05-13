"""
radio.py
--------
Radio link model: Shannon capacity, free-space path loss, and graph
edge helpers for ISL / downlink capacity computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from .constants import SPEED_OF_LIGHT, BOLTZMANN


@dataclass
class RadioParams:
    """
    Radio link parameters.

    Construct with the factories:
      - RadioParams.fixed_rate(power_tx_w, power_rx_w, rate_bps)
      - RadioParams.shannon(frequency_hz, bandwidth_hz, power_tx_w, power_rx_w,
                            gain_tx_dbi, gain_rx_dbi, noise_temp_k=290.0)

    power_tx_w and power_rx_w are always required (energy accounting).
    Shannon parameters are only required when fixed_rate_bps is None.
    """

    power_tx_w: float
    power_rx_w: float
    fixed_rate_bps: float | None = None
    frequency_hz: float | None = None
    bandwidth_hz: float | None = None
    gain_tx_dbi: float | None = None
    gain_rx_dbi: float | None = None
    noise_temp_k: float = 290.0

    def __post_init__(self) -> None:
        if self.fixed_rate_bps is None:
            missing = [
                name
                for name, val in [
                    ("frequency_hz", self.frequency_hz),
                    ("bandwidth_hz", self.bandwidth_hz),
                    ("gain_tx_dbi", self.gain_tx_dbi),
                    ("gain_rx_dbi", self.gain_rx_dbi),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"Shannon mode: missing parameters: {missing}. "
                    "Provide fixed_rate_bps or all Shannon parameters."
                )

    @classmethod
    def fixed_rate(
        cls,
        power_tx_w: float,
        power_rx_w: float,
        rate_bps: float,
    ) -> "RadioParams":
        """Fixed-rate link — no RF parameters required."""
        return cls(
            power_tx_w=power_tx_w, power_rx_w=power_rx_w, fixed_rate_bps=rate_bps
        )

    @classmethod
    def shannon(
        cls,
        *,
        frequency_hz: float,
        bandwidth_hz: float,
        power_tx_w: float,
        power_rx_w: float,
        gain_tx_dbi: float,
        gain_rx_dbi: float,
        noise_temp_k: float = 290.0,
    ) -> "RadioParams":
        """Shannon-capacity link (Friis + log2(1+SNR))."""
        return cls(
            power_tx_w=power_tx_w,
            power_rx_w=power_rx_w,
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            gain_tx_dbi=gain_tx_dbi,
            gain_rx_dbi=gain_rx_dbi,
            noise_temp_k=noise_temp_k,
        )


# ---------------------------------------------------------------------------
# Radio calculations
# ---------------------------------------------------------------------------


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def _free_space_path_loss(distance_m: float, frequency_hz: float) -> float:
    """FSPL (linear, not in dB)."""
    if distance_m <= 0.0:
        return 1.0
    return (4.0 * math.pi * distance_m * frequency_hz / SPEED_OF_LIGHT) ** 2


def _snr(distance_m: float, radio: RadioParams) -> float:
    assert radio.frequency_hz is not None
    assert radio.bandwidth_hz is not None
    assert radio.gain_tx_dbi is not None
    assert radio.gain_rx_dbi is not None
    fspl = _free_space_path_loss(distance_m, radio.frequency_hz)
    g_tx = _db_to_linear(radio.gain_tx_dbi)
    g_rx = _db_to_linear(radio.gain_rx_dbi)
    noise_power = BOLTZMANN * radio.noise_temp_k * radio.bandwidth_hz
    return (radio.power_tx_w * g_tx * g_rx) / (fspl * noise_power)


def _link_capacity_bps(distance_m: float, radio: RadioParams) -> float:
    """Effective Shannon capacity [bits/s]."""
    if radio.fixed_rate_bps is not None:
        return radio.fixed_rate_bps
    snr = _snr(distance_m, radio)
    assert radio.bandwidth_hz is not None
    return radio.bandwidth_hz * math.log2(1.0 + snr)


def _distance_from_weight(weight: float) -> float:
    """Edge weight is d² [m²] → distance [m]."""
    return math.sqrt(max(weight, 0.0))


# ---------------------------------------------------------------------------
# RadioParams factory: target rate at a given distance
# ---------------------------------------------------------------------------


def radio_for_target(
    target_bps: float,
    at_distance_m: float,
    frequency_hz: float = 2.4e9,
    power_tx_w: float = 1.0,
    power_rx_w: float = 1.0,
    gain_tx_dbi: float = 0.0,
    gain_rx_dbi: float = 0.0,
    noise_temp_k: float = 290.0,
) -> RadioParams:
    """
    Build a RadioParams whose Shannon capacity equals target_bps at at_distance_m.

    Solves for the bandwidth B required to achieve the target rate, with all
    other physical parameters fixed. Uses bisection on the monotone equation
    C(B) = B · log₂(1 + SNR₀/B) = target_bps, where SNR₀ = P·Gt·Gr/(FSPL·k·T).

    As B → ∞ the capacity converges to SNR₀/ln(2) (wideband limit). If
    target_bps exceeds this limit, no finite bandwidth can reach the target
    and a ValueError is raised.

    Args:
        target_bps: Desired Shannon capacity [bits/s].
        at_distance_m: Reference distance [m] at which the capacity is matched.
        frequency_hz: Carrier frequency [Hz]. Default: 2.4 GHz (S-band).
        power_tx_w: Transmit power [W].
        power_rx_w: Receive power [W] (energy accounting only, not used in SNR).
        gain_tx_dbi: Transmit antenna gain [dBi].
        gain_rx_dbi: Receive antenna gain [dBi].
        noise_temp_k: System noise temperature [K].

    Returns:
        RadioParams configured for Shannon mode with the solved bandwidth.

    """
    fspl = _free_space_path_loss(at_distance_m, frequency_hz)
    g_tx = _db_to_linear(gain_tx_dbi)
    g_rx = _db_to_linear(gain_rx_dbi)
    snr0 = power_tx_w * g_tx * g_rx / (fspl * BOLTZMANN * noise_temp_k)
    c_max = snr0 / math.log(2)
    if target_bps >= c_max:
        raise ValueError(
            f"Target {target_bps:.3g} bps exceeds physical maximum "
            f"{c_max:.3g} bps at {at_distance_m:.0f} m "
            f"(increase power/gains or reduce distance)."
        )

    lo, hi = 1.0, 1e13  # [1 Hz, 10 THz] — spans all physically plausible bandwidths
    for _ in range(60):  # 60 bisections → precision < 1e13/2^60 ≈ 1e-5 Hz
        mid = (lo + hi) / 2.0
        if mid * math.log2(1.0 + snr0 / mid) < target_bps:
            lo = mid
        else:
            hi = mid

    return RadioParams.shannon(
        frequency_hz=frequency_hz,
        bandwidth_hz=(lo + hi) / 2.0,
        power_tx_w=power_tx_w,
        power_rx_w=power_rx_w,
        gain_tx_dbi=gain_tx_dbi,
        gain_rx_dbi=gain_rx_dbi,
        noise_temp_k=noise_temp_k,
    )


# ---------------------------------------------------------------------------
# Graph edge helpers
# ---------------------------------------------------------------------------


def _is_downlink_edge(u: str, v: str, stations: frozenset[str]) -> bool:
    return (u in stations) or (v in stations)


def link_capacity_for_edge(
    G: nx.Graph,
    u: str,
    v: str,
    stations: frozenset[str],
    isl_radio: RadioParams,
    downlink_radio: RadioParams,
) -> float:
    """Capacity [bits/s] of an edge based on its type (ISL or downlink)."""
    weight = G[u][v].get("weight", 0.0)
    distance = _distance_from_weight(weight)
    radio = downlink_radio if _is_downlink_edge(u, v, stations) else isl_radio
    return _link_capacity_bps(distance, radio)
