# Nanosatellite Swarm Downlink Simulation

Simulation framework for the paper **"On the Optimal Number of Relays for Nanosatellite Swarm Downlink under Ground Station Reconfiguration Overhead"**, submitted to LEO-NET 2026.

This repository contains the full source code and pre-computed data required to reproduce all simulation results and figures presented in the paper.

---

## Repository Structure

```
.
├── src/                        # Core simulation API
│   └── ...                     # Modules handling orbital trajectory generation,
│                               # connectivity graph computation, downlink scheduling,
│                               # and data transfer simulation
│
├── results/
│   ├── generate_data_relay.py  # Runs the simulation across parameter configurations
│   │                           # and stores results in data/
│   ├── generate_figures_relay.py  # Produces Figures 4–6 from pre-computed data
│   └── generate_figures_snapshot.py  # Produces Figures 2–3
│
├── data/                       # Pre-computed simulation outputs
│                               # (can be regenerated via generate_data_relay.py)
├── figures/                    # Output figures as referenced in the paper
├── requirements.txt            # Python dependencies
└── LICENSE
```

---

## Source modules (`src/`)

| Module | Role |
|---|---|
| `constants.py` | Physical constants (µ, R⊕, speed of light, Boltzmann…) |
| `orbit.py` | Kepler propagator — vectorised Newton-Raphson solver, ECEF position array |
| `radio.py` | Radio link model — free-space path loss, Shannon capacity, `RadioParams` factory |
| `formation.py` | Swarm generation — randomised orbital parameters around a reference orbit |
| `network.py` | Connectivity graph construction — ISL and downlink edges at each time step |
| `scheduling.py` | Pre-computed downlink schedule — visibility windows, slot budget policies, assignment |
| `routing.py` | ISL routing policies — relay selection and next-hop computation |
| `simulator.py` | Simulation engine — crosslink + downlink phases, buffer tracking (`Simulator`) |
| `framework.py` | Batch utilities — topology caching, parallel simulation, metric aggregation |
| `params.py` | Parameter dataclasses — `SimulationParams`, `SimulationResult` |

---

## Requirements

- Python 3.12.3
- Dependencies listed in `requirements.txt`

---

## Installation

It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing the Results

### Pre-computed data (recommended)

The `data/` directory already contains the simulation outputs used to generate the figures in the paper. To reproduce the figures directly:

```bash
python results/generate_figures_snapshot.py   # Figures 2–3
python results/generate_figures_relay.py      # Figures 4–6
```

Output figures will be saved in `figures/`.

### Recomputing from scratch

The simulation can be re-run from scratch using:

```bash
python results/generate_data_relay.py
```

> **Note:** This script takes approximately 30 minutes to complete on a 16-core CPU. It sweeps over relay count $k$, initial data volume $D$, crosslink capacity $C_\mathrm{ISL}$, and reconfiguration time $T_r$, and stores the results in `data/`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.