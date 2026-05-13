"""
config.py
---------
Shared simulation parameters used by generate_data_relay.py and
generate_figures_snaphot.py.  Edit here to change a parameter globally;
per-script overrides remain in each script.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orbit import OrbitalParameters

# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

SEED = 42
N_SATELLITES = 15

# Circular equatorial orbit at ~400 km altitude
BASE_ORBITAL_PARAMS = OrbitalParameters(
    6_770.5e3,          # semi-major axis [m]
    0.001095,           # eccentricity
    math.radians(0),    # inclination [rad]
    math.radians(0),    # RAAN [rad]
    math.radians(0),    # argument of perigee [rad]
    math.radians(100),  # mean anomaly at t=0 [rad]
)

DURATION_S = 28_500  # 6 orbital periods
DT_S = 1.0           # simulation time step [s]
R_MAX_M = 30_000     # maximum ISL range [m]
ELEV_MIN_DEG = 10.0  # minimum elevation angle for downlink visibility [deg]

# Single equatorial ground station (matches inclination=0 orbit)
STATIONS = [
    [math.radians(0.0), math.radians(0.0), 0.0],  # [lat, lon, alt_m]
]

# ---------------------------------------------------------------------------
# Downlink (X-band)
# ---------------------------------------------------------------------------

DL_RATE_BPS = 100e6        # target Shannon capacity at reference distance [bps]
DL_DISTANCE_M = 500_000.0  # reference distance for capacity matching [m]
DL_FREQ_HZ = 8.2e9         # carrier frequency [Hz]
DL_POWER_TX_W = 20.0       # transmit power [W]
DL_GAIN_TX_DBI = 6.0       # transmit antenna gain [dBi]
DL_GAIN_RX_DBI = 30.0      # receive antenna gain [dBi]

# ---------------------------------------------------------------------------
# Crosslink (S-band)
# ---------------------------------------------------------------------------

XL_DISTANCE_M = 30_000.0  # reference distance for capacity matching [m]
XL_FREQ_HZ = 2.4e9        # carrier frequency [Hz]
XL_POWER_TX_W = 10.0      # transmit power [W]
XL_GAIN_TX_DBI = 3.0      # transmit antenna gain [dBi]
XL_GAIN_RX_DBI = 3.0      # receive antenna gain [dBi]
