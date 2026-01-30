# 🏁 Motorsport Grand Slam Assistant (MGSA)

**MGSA** is a full-stack motorsport analytics platform that turns **raw GPS laps** into
**optimal racing lines, data-driven insights, and real-time driver feedback**.

From the track → to algorithms → to visual guidance — all in one system.

---

## ✨ What problem does MGSA solve?

Most lap-analysis tools stop at *recording* data.
MGSA goes further:

* **Understands the track geometry**
* **Computes the optimal trajectory**
* **Compares the driver to the ideal line**
* **Closes the loop with real-time feedback in the car**

---

## 🧩 System Overview

MGSA consists of three tightly-integrated layers:

### 🚗 On-car runtime

* Physical buttons
* GPS + IMU acquisition
* LED / HUD driver feedback
* Lap recording (inner / outer / racing)

### 🧠 Analysis & Optimization

* Curvature & segmentation tools
* Heatmaps and speed profiles
* CasADi + IPOPT optimal racing-line solver

### 🌐 Server & Visualization

* FastAPI backend
* Track & lap storage
* Interactive maps, comparisons and dashboards

Everything lives in this repository.

---

## 📁 Repository structure

```
.
├── server/                 # FastAPI backend + APIs
│   ├── static/             # Web UIs (compare, maps, heatmaps)
│   └── templates/          # HTML exports
│
├── firmware/               # Offline analysis & visualization
│   ├── curves.py
│   ├── curvature.py
│   ├── segmentation.py
│   ├── visualization.py
│   ├── track_coloring.py
│   ├── vmax_raceline/      # Simple vmax vs curvature model
│   └── Optimal_Control/    # CasADi/IPOPT optimal control solver
│
├── hardware/               # Embedded runtime (on-car)
│   └── diploma/
│       ├── runtime/        # Main execution loop & state machine
│       ├── services/       # Button daemon
│       ├── hud/            # LED / HUD logic
│       └── config/         # mgsa.yaml configuration
│
├── diagrams/               # PlantUML + exported PNGs
├── tests/                  # Analysis experiments & utilities
└── mgsa_data/              # Auto-generated runtime data (gitignored)
```

📌 Each major folder contains its own README with deeper details.

---

## 🚀 Installation

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
```

### Requirements (high-level)

* Python **3.8+**
* FastAPI + Uvicorn
* numpy, scipy, pandas
* matplotlib, folium, plotly
* casadi
* pyyaml
* gpiozero *(for embedded runtime)*

---

## 🌐 Running the server

```bash
.\.venv\Scripts\activate
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

or

```bash
python -m server.server
```

The server will automatically create:

* `./mgsa_data/`
* a SQLite database
* CSV artifacts for laps, boundaries and racing lines

---

## 🧪 Firmware tools (offline analysis)

Most offline experiments live in `firmware/`.

### Curvature & segmentation

```bash
python -m firmware.curves --points data/simple_track.csv --mad
```

### Heatmaps

```bash
python -m firmware.curves --points data/simple_track.csv --heatmap
```

### Interactive web visualization

```bash
python -m firmware.curves \
  --points data/simple_track.csv \
  --web templates/heatmap.html
```

### Outline + racing line overlay

```bash
python -m firmware.curves \
  --outline-csv data/simple_track.csv \
  --outline-web templates/outline.html \
  --raceline data/raceline.csv \
  --mad --factor 3
```

👉 For the full optimal-control pipeline, see
`firmware/Optimal_Control/README.md`

---

## 🏎️ Embedded / on-car runtime

Typical startup sequence on the device:

```bash
python -m diploma.runtime.app --config diploma/config/mgsa.yaml
python -m diploma.services.button_daemon &
python -m diploma.hud.led_strip_daemon &
```

### Runtime responsibilities

* Reads **GPS + IMU**
* Detects laps & states (idle / record / race)
* Sends data to the server
* Receives optimal trajectories
* Computes driver vs optimal deviation
* Drives LED / HUD feedback in real time

All hardware logic is configured via:

```
diploma/config/mgsa.yaml
```

---

## ⚡ Quick start workflows

### 🔍 Just explore track geometry

```bash
python -m firmware.curves --points data/simple_track.csv --mad
python -m firmware.curves --points data/simple_track.csv --heatmap
```

### 🆚 Driver vs Optimal comparison

1. Start the server
2. Record or upload laps
3. Build boundaries & optimal line
4. Open the comparison UI (`server/static/compare.html`)

---

## 📚 Philosophy

MGSA is designed to be:

* **Engineering-first**, not marketing-first
* **Research-friendly**, not locked-down
* **Modular**, not monolithic
* **Executable**, not just theoretical

This README is intentionally high-level.
Each subsystem is documented where it lives.
