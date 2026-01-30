# Motorsport Grand Slam Assistant (MGSA)
Embedded device that records the car's behavior during laps on a race track and analyzes the most optimal trajectory. It shows the pilot a real-time visual indication (via a HUD display) of the ideal line of travel relative to the car's current position. The system has a web-based interface where you can monitor important parameters.

---

# Motorsport Curvature & Segmentation Toolkit

This module provides **curvature calculation, segmentation, and visualization tools** for motorsport track analysis.
It supports **MATLAB-style heatmaps, interactive Plotly visualizations, track segmentation with Median + MAD**, and **track outline rendering**.

---

## Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/your-repo/firmware.git
cd firmware
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install numpy matplotlib plotly
```

---

## Usage

Run the script directly with:

```bash
python -m firmware.curves [OPTIONS]
```

### Input Data

* `--points <path>` : CSV or JSON file with track points (`x,y` format).
* Alternatively, set via environment variable:

  ```bash
  export TRACK_POINTS_FILE=data/simple_track.csv
  ```

---

## Example Data Format - time_s,x_m,y_m,w_tr_right_m,w_tr_left_m,pitch_deg,roll_deg

### Track Points (CSV)

```csv
x,y
0,0
1,2
2,3
3,5
```

### Track Outline (CSV with widths)

```csv
x,y,left,right
0,0,3,3
1,2,2.5,2.5
2,3,2,2
3,5,3,3
```

---

## Requirements

* Python 3.8+
* Libraries:

  * `numpy`
  * `matplotlib` (for heatmaps)
  * `plotly` (for interactive HTML exports)

Install all dependencies with:

```bash
pip install numpy matplotlib plotly
```

---

## Quick Start

1. Run segmentation on the included track:

   ```bash
   python -m firmware.curves --points data/simple_track.csv --mad
   ```

2. View curvature heatmap:

   ```bash
   python -m firmware.curves --points data/simple_track.csv --heatmap
   ```

3. Export interactive web heatmap:

   ```bash
   python -m firmware.curves --points data/simple_track.csv --web out_heatmap.html
   ```

4. Try the default demo:

   ```bash
   python -m firmware.curves
   ```

4. Try full raceline with traight and turns:

   ```bash
   python -m firmware.curves --outline-csv data/simple_track.csv --outline-web templates/outline.html --raceline data/raceline.csv --mad --factor 3
   ```
---