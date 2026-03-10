import csv
import sys
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v):
    try:
        return float(v)
    except:
        return None


def plot_results(rows):

    lap_times = []
    noise = []
    offset = []
    v_straight = []

    for r in rows:
        lt = to_float(r.get("lap_time"))
        n = to_float(r.get("noise_m"))
        off = to_float(r.get("global_offset_m"))
        vs = to_float(r.get("v_straight_kmh"))

        if lt is not None:
            lap_times.append(lt)

        if n is not None:
            noise.append(n)

        if off is not None:
            offset.append(off)

        if vs is not None:
            v_straight.append(vs)

    runs = list(range(1, len(lap_times) + 1))

    # --- Graph 1: Lap time per run ---
    plt.figure()
    plt.plot(runs, lap_times, marker="o")
    plt.title("Lap Time per Monte Carlo Run")
    plt.xlabel("Run")
    plt.ylabel("Lap Time (s)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python plot_mc_results.py mc_results_full_cl_fast.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    rows = load_csv(csv_file)

    print(f"Loaded {len(rows)} runs")

    plot_results(rows)