import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d

try:
    from .constraints import add_constraints
    from .visualization import plot_f1_results, print_summary
except Exception:
    try:
        from firmware.Optimal_Control.constraints import add_constraints
        from firmware.Optimal_Control.visualization import plot_f1_results, print_summary
    except Exception:
        import os, sys
        sys.path.append(os.path.dirname(__file__))
        from constraints import add_constraints
        from visualization import plot_f1_results, print_summary

@dataclass
class VehicleParams:
    mass_kg: float = 798.0
    mu_friction: float = 1.8
    gravity: float = 9.81
    rho_air: float = 1.225
    cL_downforce: float = 3.0
    cD_drag: float = 1.2
    frontal_area_m2: float = 1.5
    engine_power_watts: float = 750000.0
    brake_power_watts: float = 2500000.0
    a_accel_max: float = 12.0
    a_brake_max: float = 45.0
    a_lat_max: float = 60.0
    c_rr: float = 0.02
    wheelbase_m: float = 3.6
    v_min: float = 15.0
    v_max: float = 100.0

    def k_aero(self) -> float:
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / self.mass_kg

    def k_drag(self) -> float:
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2

vehicle = VehicleParams()

# ═══════════════════════════════════════════════════════════════
# 📊 LOAD TRACK DATA
# ═══════════════════════════════════════════════════════════════
try:
    track = pd.read_csv("Turn_1.csv", comment="#", names=["x_m","y_m","w_tr_right_m","w_tr_left_m"])
except:
    print("Warning: Could not find Turn_1.csv, using synthetic track")
    theta = np.linspace(0, 2*np.pi, 100)
    r = 200 + 50*np.sin(3*theta)
    track = pd.DataFrame({
        'x_m': r * np.cos(theta),
        'y_m': r * np.sin(theta),
        'w_tr_right_m': np.ones(100) * 8,
        'w_tr_left_m': np.ones(100) * 8
    })

x_center = track["x_m"].values
y_center = track["y_m"].values
w_right = track["w_tr_right_m"].values
w_left = track["w_tr_left_m"].values
N_raw = len(x_center)

# ═══════════════════════════════════════════════════════════════
# 🔹 A. ADAPTIVE STEP SIZE (Δs)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🔹 ADAPTIVE DISCRETIZATION")
print("="*60)

# Calculate initial curvature on raw track
curvature_raw = np.zeros(N_raw)
for i in range(N_raw):
    i_prev = (i - 1) % N_raw
    i_next = (i + 1) % N_raw
    x0, y0 = x_center[i_prev], y_center[i_prev]
    x1, y1 = x_center[i], y_center[i]
    x2, y2 = x_center[i_next], y_center[i_next]
    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x1, y2 - y1
    cross = dx1*dy2 - dy1*dx2
    ds1 = np.sqrt(dx1**2 + dy1**2) + 1e-9
    ds2 = np.sqrt(dx2**2 + dy2**2) + 1e-9
    curvature_raw[i] = 2 * cross / (ds1 * ds2 * (ds1 + ds2) + 1e-9)

curvature_raw = gaussian_filter1d(curvature_raw, sigma=2, mode='wrap')

# Adaptive resampling
CURVATURE_THRESHOLD = 0.01
DS_STRAIGHT = 5.0
DS_CORNER = 1.5

new_x, new_y, new_w_right, new_w_left = [], [], [], []
i = 0
while i < N_raw:
    new_x.append(x_center[i])
    new_y.append(y_center[i])
    new_w_right.append(w_right[i])
    new_w_left.append(w_left[i])
    
    abs_curv = abs(curvature_raw[i])
    if abs_curv > CURVATURE_THRESHOLD:
        ds_target = DS_CORNER
    else:
        ds_target = DS_STRAIGHT
    
    i_next = (i + 1) % N_raw
    dx = x_center[i_next] - x_center[i]
    dy = y_center[i_next] - y_center[i]
    ds_actual = np.sqrt(dx**2 + dy**2)
    
    skip = max(1, int(ds_target / (ds_actual + 1e-6)))
    i += skip
    
    if i >= N_raw and len(new_x) > 10:
        break

x_center = np.array(new_x)
y_center = np.array(new_y)
w_right = np.array(new_w_right)
w_left = np.array(new_w_left)
N = len(x_center)

print(f"Original segments: {N_raw}")
print(f"Adaptive segments: {N}")
print(f"Reduction: {100*(1-N/N_raw):.1f}%")

# Recalculate segment lengths
ds_array = np.zeros(N)
for i in range(N):
    i_next = (i + 1) % N
    dx = x_center[i_next] - x_center[i]
    dy = y_center[i_next] - y_center[i]
    ds_array[i] = np.sqrt(dx**2 + dy**2)

ds_avg = np.mean(ds_array)
ds_min = np.min(ds_array)
ds_max = np.max(ds_array)
print(f"Δs: avg={ds_avg:.2f}m, min={ds_min:.2f}m, max={ds_max:.2f}m")

# ═══════════════════════════════════════════════════════════════
# 📐 GEOMETRY CALCULATIONS
# ═══════════════════════════════════════════════════════════════
normals = np.zeros((N, 2))
curvature = np.zeros(N)

for i in range(N):
    i_prev = (i - 1) % N
    i_next = (i + 1) % N
    dx = x_center[i_next] - x_center[i_prev]
    dy = y_center[i_next] - y_center[i_prev]
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    normals[i, 0] = -dy / norm
    normals[i, 1] = dx / norm

for i in range(N):
    i_prev = (i - 1) % N
    i_next = (i + 1) % N
    x0, y0 = x_center[i_prev], y_center[i_prev]
    x1, y1 = x_center[i], y_center[i]
    x2, y2 = x_center[i_next], y_center[i_next]
    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x1, y2 - y1
    cross = dx1*dy2 - dy1*dx2
    ds1 = np.sqrt(dx1**2 + dy1**2) + 1e-9
    ds2 = np.sqrt(dx2**2 + dy2**2) + 1e-9
    curvature[i] = 2 * cross / (ds1 * ds2 * (ds1 + ds2) + 1e-9)

curvature = gaussian_filter1d(curvature, sigma=2, mode='wrap')

# ═══════════════════════════════════════════════════════════════
# 🎯 OPTIMIZATION SETUP
# ═══════════════════════════════════════════════════════════════
opti = ca.Opti()
n = opti.variable(N)
v = opti.variable(N)
a_lon = opti.variable(N)
slack_power = opti.variable(N)

opti.set_initial(slack_power, 0)
for i in range(N):
    opti.subject_to(slack_power[i] >= 0)
    opti.subject_to(slack_power[i] <= 50000)

# ═══════════════════════════════════════════════════════════════
# 🔹 B. MULTI-OBJECTIVE COST FUNCTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🔹 MULTI-OBJECTIVE COST FUNCTION")
print("="*60)

lap_time = ca.sum1(ds_array / v)
a_lon_squared = ca.sum1(a_lon**2)
tire_load_penalty = 0.001 * a_lon_squared

path_diffs = [ca.sqrt((n[i+1] - n[i])**2 + ds_array[i]**2) for i in range(N-1)]
path_length_penalty = 0.01 * ca.sum1(ca.vertcat(*path_diffs))

n_diff = [(n[(i+1)%N] - n[i])**2 for i in range(N)]
trajectory_smoothness = 0.0001 * ca.sum1(ca.vertcat(*n_diff))

# a_lon_changes = [(a_lon[(i+1)%N] - a_lon[i])**2 for i in range(N)]
# energy_smoothness = 0.0002 * ca.sum1(ca.vertcat(*a_lon_changes))

reg_n = 1e-8 * ca.sum1(n**2)
reg_a = 1e-6 * ca.sum1(a_lon**2)
reg_slack = 1e3 * ca.sum1(slack_power**2)

total_cost = (
    lap_time +
    tire_load_penalty +
    path_length_penalty +
    trajectory_smoothness +
    # energy_smoothness +
    reg_n + reg_a + reg_slack
)

opti.minimize(total_cost)

print("Cost components:")
print("  ✓ Lap time (primary)")
print("  ✓ Tire load (G-force squared)")
print("  ✓ Path smoothness")
print("  ✓ Trajectory smoothness")
print("  ✓ Energy efficiency (jerk)")

# ═══════════════════════════════════════════════════════════════
# 🚧 CONSTRAINTS
# ═══════════════════════════════════════════════════════════════
vehicle.mu_friction = 2.0
vehicle.cL_downforce = 3.0
vehicle.a_accel_max = 12.0
vehicle.a_brake_max = 45.0

a_lat = add_constraints(
    opti=opti,
    vehicle=vehicle,
    n=n,
    v=v,
    a_lon=a_lon,
    slack_power=slack_power,
    w_left=w_left,
    w_right=w_right,
    ds_array=ds_array,
    curvature=curvature,
)

# ═══════════════════════════════════════════════════════════════
# 🎬 INITIALIZATION
# ═══════════════════════════════════════════════════════════════
n_init = np.zeros(N)
for i in range(N):
    if curvature[i] > 0.001:
        n_init[i] = w_left[i] * 0.8
    elif curvature[i] < -0.001:
        n_init[i] = -w_right[i] * 0.8
    else:
        n_init[i] = 0

v_init = np.zeros(N)
for i in range(N):
    abs_curv = abs(curvature[i])
    if abs_curv < 1e-4:
        v_init[i] = 120.0
    else:
        a_lat_available = vehicle.a_lat_max
        v_corner = np.sqrt(a_lat_available / (abs_curv + 1e-6))
        v_init[i] = min(v_corner, 80.0)

v_init = gaussian_filter1d(v_init, sigma=5, mode='wrap')
v_init = np.clip(v_init, vehicle.v_min, vehicle.v_max)
opti.set_initial(v, v_init)

a_init = np.zeros(N)
for i in range(N):
    i_next = (i + 1) % N
    dv = v_init[i_next] - v_init[i]
    dt = ds_array[i] / (v_init[i] + 1e-3)
    a_init[i] = dv / dt

a_init = np.clip(a_init, -vehicle.a_brake_max, vehicle.a_accel_max)
opti.set_initial(n, n_init)
opti.set_initial(a_lon, a_init)

# ═══════════════════════════════════════════════════════════════
# ⚙️ SOLVER CONFIGURATION
# ═══════════════════════════════════════════════════════════════
opts = {
    'ipopt.max_iter': 3000,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_iter': 20,
    'ipopt.constr_viol_tol': 1e-3,
    'ipopt.print_level': 5,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.warm_start_bound_push': 1e-6,
    'ipopt.warm_start_mult_bound_push': 1e-6,
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.adaptive_mu_globalization': 'kkt-error',
    'ipopt.linear_solver': 'mumps',
    'ipopt.hessian_approximation': 'exact',
    'ipopt.limited_memory_max_history': 50,
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.line_search_method': 'filter',
    'ipopt.alpha_for_y': 'primal',
    'error_on_fail': False
}

opti.solver('ipopt', opts)

# ═══════════════════════════════════════════════════════════════
# 🚀 SOLVE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🚀 Starting F1 lap time optimization...")
print("="*60)

try:
    sol = opti.solve()
    print("\nOptimization converged successfully!")
except Exception as e:
    print(f"\nSolver didn't fully converge: {e}")
    print("Using best solution found...")
    sol = opti.debug

# ═══════════════════════════════════════════════════════════════
# 📊 EXTRACT RESULTS
# ═══════════════════════════════════════════════════════════════
n_opt = sol.value(n)
v_opt = sol.value(v)
a_lon_opt = sol.value(a_lon)
a_lat_opt = sol.value(a_lat)

lap_time_seconds = float(sol.value(lap_time))
track_length = np.sum(ds_array)

# ═══════════════════════════════════════════════════════════════
# 💾 SAVE OPTIMAL TRAJECTORY TO CSV
# ═══════════════════════════════════════════════════════════════
# Compute absolute coordinates of the optimal trajectory
x_opt = x_center + n_opt * normals[:, 0]
y_opt = y_center + n_opt * normals[:, 1]

# Compute cumulative time at the start of each segment (t[0] = 0)
eps = 1e-6
time_s = np.cumsum(np.concatenate(([0.0], ds_array[:-1] / np.maximum(v_opt[:-1], eps))))

# Convert speed to km/h
speed_kmh = v_opt * 3.6

# Create dataframe with only required columns
opt_df = pd.DataFrame({
    "x_m": x_opt,
    "y_m": y_opt,
    "speed_kmh": speed_kmh,
    "time_s": time_s,
})

# Save to CSV
opt_df.to_csv("optiline.csv", index=False)
print("\n✅ Optimal trajectory saved as 'optiline.csv'!")

# ═══════════════════════════════════════════════════════════════
# 📋 PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════
print_summary(
    v_opt=v_opt,
    a_lon_opt=a_lon_opt,
    a_lat_opt=a_lat_opt,
    vehicle=vehicle,
    lap_time_seconds=lap_time_seconds,
    track_length=track_length
)

# ═══════════════════════════════════════════════════════════════
# 📈 VISUALIZATION
# ═══════════════════════════════════════════════════════════════
fig = plot_f1_results(
    x_center=x_center,
    y_center=y_center,
    w_left=w_left,
    w_right=w_right,
    normals=normals,
    n_opt=n_opt,
    v_opt=v_opt,
    a_lon_opt=a_lon_opt,
    a_lat_opt=a_lat_opt,
    ds_array=ds_array,
    vehicle=vehicle,
    lap_time_seconds=lap_time_seconds,
    track_length=track_length,
    N=N
)

plt.show()