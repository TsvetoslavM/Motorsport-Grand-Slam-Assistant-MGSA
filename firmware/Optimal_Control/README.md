## Optimal_Control module

This folder contains the **CasADi/IPOPT‑based racing‑line optimizer** used to compute an optimal trajectory between track boundaries.

- `CasADi_IPOPT.py` – main script; loads `Track.csv`, builds an optimal line, and writes `optiline.csv` with the optimized trajectory.
- `constraints.py` – all physical and geometric constraints (track boundaries, traction circle, jerk, apex logic, chicanes, etc.).
- `solver_api.py` – callable API (`OptimizeOptions`, `optimize_trajectory_from_two_lines`) used by the MGSA server and tools.
- `visualization.py` – rich Matplotlib visualizations and summary for F1‑style analysis.
- `Track.csv` – example track centerline + widths.

### Running the optimizer locally

From the project root:

```bash
cd firmware/Optimal_Control
python CasADi_IPOPT.py
```

This will:

1. Load `Track.csv` (or generate a synthetic track if missing).
2. Build an optimal racing line subject to all constraints.
3. Save the result to `optiline.csv` in this directory.

### Dependencies

- Python 3.8+
- `casadi`
- `numpy`, `pandas`, `matplotlib`, `scipy`

Install (example):

```bash
pip install casadi numpy pandas matplotlib scipy
```

