import csv
import math
import statistics
import sys
from pathlib import Path


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def summarize_csv(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty.")
        return

    numeric_columns = {}
    for row in rows:
        for key, value in row.items():
            num = _to_float(value)
            if num is not None:
                numeric_columns.setdefault(key, []).append(num)

    print(f"Rows: {len(rows)}")
    print()

    wanted = [
        "lap_time",
        "noise_m",
        "bias",
        "global_offset_m",
        "v_straight_kmh",
        "v_corner_kmh",
        "speed_noise_kmh",
        "stats_mean_deviation_m",
        "stats_max_deviation_m",
        "stats_rms_deviation_m",
    ]

    shown_any = False
    for col in wanted:
        values = numeric_columns.get(col)
        if not values:
            continue
        shown_any = True
        print(f"{col}:")
        print(f"  mean = {statistics.mean(values):.6f}")
        print(f"  min  = {min(values):.6f}")
        print(f"  max  = {max(values):.6f}")
        if len(values) > 1:
            print(f"  std  = {statistics.pstdev(values):.6f}")
        print()

    if not shown_any:
        print("No expected numeric summary columns were found.")
        print("Available numeric columns:")
        for col in sorted(numeric_columns.keys()):
            print(f"  {col}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_mc.py mc_results_full_cl_fast.csv")
        sys.exit(1)

    summarize_csv(sys.argv[1])