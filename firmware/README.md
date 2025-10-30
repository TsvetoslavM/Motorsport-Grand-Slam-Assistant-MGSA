## 🎯 Commands

### 1. Segmentation (Median + MAD)

Run segmentation on points:

```bash
python -m firmware.curves --points data/simple_track.csv --mad
```

Change sensitivity with factor (default = `3.0`):

```bash
python -m firmware.curves --points data/simple_track.csv --mad --factor 2.5
```

Output results as JSON:

```bash
python -m firmware.curves --points data/simple_track.csv --mad --print-json
```

---

### 2. Heatmaps (Matplotlib)

2D curvature heatmap:

```bash
python -m firmware.curves --points data/simple_track.csv --heatmap
```

3D curvature heatmap:

```bash
python -m firmware.curves --points data/simple_track.csv --heatmap3d
```

---

### 3. Interactive Web Visualizations (Plotly)

Export interactive 2D HTML heatmap:

```bash
python -m firmware.curves --points data/simple_track.csv --web templates/heatmap.html
```

Overlay a racing line:

```bash
python -m firmware.curves --points data/simple_track.csv --web templates/heatmap.html --raceline data/raceline.csv
```

Export interactive 3D HTML heatmap:

```bash
python -m firmware.curves --points data/simple_track.csv --web3d templates/heatmap3d.html
```

---

### 4. Track Outline Rendering

Render track outline with widths (`x,y,left,right`):

```bash
python -m firmware.curves --outline-csv data/simple_track.csv --outline-web templates/outline.html
```

With racing line overlay:

```bash
python -m firmware.curves --outline-csv data/simple_track.csv --outline-web templates/outline.html --raceline data/raceline.csv
```

---

### 5. Default Demo

If no arguments are given, the script runs a **demo visualization**:

```bash
python -m firmware.curves
```

---

### 6. Visualization compare of lines

```bash
python firmware/Optimal_Control/compare_racing_lines_altair.py --inputs templates/outline_raceline.csv "Optimla line" firmware/Optimal_Control/optiline.csv "My line" --track_csv firmware/Optimal_Control/Track.csv  --output_html comparison_altair.html
```

---