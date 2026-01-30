## Motorsport Grand Slam Assistant (MGSA)

MGSA is an end‑to‑end toolkit for **recording, analysing, and optimizing race laps**:

- An **embedded runtime** on the car (buttons, GPS, IMU, LED HUD).
- A **FastAPI server** on laptop/PC for storing laps, managing tracks and visualizing comparisons.
- A **firmware toolkit** with curvature/segmentation tools and a full CasADi/IPOPT racing‑line optimizer.

This repo contains everything needed to go from raw GPS laps to optimal racing lines and visual feedback for the driver.

---

## Repository structure

- `server/` – FastAPI backend (`server.server:app`), APIs for:
  - recording laps, uploading boundaries, racing lines and comparisons,
  - websocket live status and map/compare UIs.
- `firmware/` – analysis/visualization toolkit:
  - `curves.py`, `curvature.py`, `segmentation.py`, `smoothing.py` – curvature & segmentation.
  - `visualization.py`, `track_coloring.py` – 2D/3D plots and HTML exports.
  - `vmax_raceline/` – simple vmax vs curvature model.
  - `Optimal_Control/` – CasADi/IPOPT optimal racing‑line solver (see its `README.md`).
- `hardware/` – embedded runtime layout (based on `diploma/`):
  - `diploma/runtime/app.py`, `state_machine.py`, `race_mode.py` – on‑car runtime + race feedback.
  - `diploma/services/button_daemon.py` – GPIO button handler.
  - `diploma/hud/led_strip_daemon.py` – LED HUD.
  - `diploma/config/mgsa.yaml` – main on‑car config.
- `templates/`, `server/static/` – HTML UIs for tracks, heatmaps, and driver vs optimal comparison.
- `diagrams/` – PlantUML diagrams and exported PNGs for system and data‑flow architecture.
- `tests/` – utilities and experiments for speed profiles, visualization, and server interaction.

See also:

- `server/README.md` – server details and routes.
- `firmware/README.md` – firmware utilities and commands.
- `firmware/Optimal_Control/README.md` – optimal‑control solver.
- `hardware/README.md` – embedded/runtime overview.

---

## Installation

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

Requirements (high‑level):

- Python 3.8+
- FastAPI + Uvicorn
- numpy, scipy, pandas, matplotlib, folium, plotly
- casadi
- pyyaml, gpiozero (for hardware runtime)

---

## Running the server

From the project root:

```bash
.\.venv\Scripts\activate
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m server.server
```

The server will create `./mgsa_data` and a SQLite DB, plus CSV laps and track artifacts.

---

## Firmware tools (offline analysis)

Most offline analysis lives in `firmware/`. A common entrypoint is:

```bash
python -m firmware.curves --points data/simple_track.csv --mad
```

Examples:

- Curvature/segmentation demo:

  ```bash
  python -m firmware.curves --points data/simple_track.csv --mad
  ```

- 2D heatmap:

  ```bash
  python -m firmware.curves --points data/simple_track.csv --heatmap
  ```

- Interactive web heatmap:

  ```bash
  python -m firmware.curves --points data/simple_track.csv --web templates/heatmap.html
  ```

- Outline + racing line overlay:

  ```bash
  python -m firmware.curves \
    --outline-csv data/simple_track.csv \
    --outline-web templates/outline.html \
    --raceline data/raceline.csv \
    --mad --factor 3
  ```

For the full CasADi/IPOPT optimizer, see `firmware/Optimal_Control/README.md`.

---

## Embedded / hardware runtime (high‑level)

On the device you typically run:

```bash
python -m diploma.runtime.app --config diploma/config/mgsa.yaml
python -m diploma.services.button_daemon &
python -m diploma.hud.led_strip_daemon &
```

The runtime:

- listens to GPS + IMU,
- records outer/inner/race laps,
- loads ideal trajectories from the server,
- computes driver vs optimal feedback and drives LEDs / HUD.

Configuration lives in `diploma/config/mgsa.yaml` (paths, GPIO, feedback timings, etc.).

---

## Quick start flows

- **Just play with curvature & heatmaps**:

  ```bash
  python -m firmware.curves --points data/simple_track.csv --mad
  python -m firmware.curves --points data/simple_track.csv --heatmap
  ```

- **Run the backend and compare driver vs optimal**:

  1. Start server: `uvicorn server.server:app --host 0.0.0.0 --port 8000`
  2. Record or upload laps via the API.
  3. Build boundaries and optimal line (`auto_pipeline.py` / UI).
  4. Open the compare UI under `server/static/compare.html` / appropriate route.

This README is intentionally high‑level; per‑module READMEs contain more details.
