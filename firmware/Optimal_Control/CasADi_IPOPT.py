import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
try:
    from .constraints import add_constraints
except Exception:
    try:
        from firmware.Optimal_Control.constraints import add_constraints
    except Exception:
        import os, sys
        sys.path.append(os.path.dirname(__file__))
        from constraints import add_constraints

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
N = len(x_center)

ds_array = np.zeros(N)
for i in range(N):
    i_next = (i + 1) % N
    dx = x_center[i_next] - x_center[i]
    dy = y_center[i_next] - y_center[i]
    ds_array[i] = np.sqrt(dx**2 + dy**2)

ds = np.mean(ds_array)
print(f"Track segments: {N}, Average segment length: {ds:.2f}m")

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

from scipy.ndimage import gaussian_filter1d
curvature = gaussian_filter1d(curvature, sigma=2, mode='wrap')

opti = ca.Opti()
n = opti.variable(N)
v = opti.variable(N)
a_lon = opti.variable(N)
slack_power = opti.variable(N)
opti.set_initial(slack_power, 0)
for i in range(N):
    opti.subject_to(slack_power[i] >= 0)
    opti.subject_to(slack_power[i] <= 50000)

lap_time = ca.sum1(ds_array / v)
path_diffs = [ca.sqrt((n[i+1] - n[i])**2 + ds_array[i]**2) for i in range(N-1)]
path_length_penalty = 0.01 * ca.sum1(ca.vertcat(*path_diffs))
reg_n = 1e-8 * ca.sum1(n**2)
reg_a = 1e-6 * ca.sum1(a_lon**2)
reg_slack = 1e3 * ca.sum1(slack_power**2)

opti.minimize(lap_time + path_length_penalty + reg_n + reg_a + reg_slack)

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
        v_init[i] = 70.0
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

print("\n" + "="*60)
print("Starting F1 lap time optimization...")
print("="*60)

try:
    sol = opti.solve()
    print("\n✅ Optimization converged successfully!")
except Exception as e:
    print(f"\n⚠️ Solver didn't fully converge: {e}")
    print("Using best solution found...")
    sol = opti.debug

n_opt = sol.value(n)
v_opt = sol.value(v)
a_lon_opt = sol.value(a_lon)
a_lat_opt = sol.value(a_lat)

x_opt = x_center + n_opt * normals[:, 0]
y_opt = y_center + n_opt * normals[:, 1]

lap_time_seconds = float(sol.value(lap_time))
track_length = np.sum(ds_array)

s = np.cumsum(np.concatenate([[0], ds_array[:-1]]))
a_total = np.sqrt(a_lon_opt**2 + a_lat_opt**2)

F_normal = vehicle.mass_kg * (vehicle.gravity + vehicle.k_aero() * v_opt**2)
downforce_g = vehicle.k_aero() * v_opt**2 / vehicle.gravity
F_drag = vehicle.k_drag() * v_opt**2
power_used = vehicle.mass_kg * a_lon_opt * v_opt + F_drag * v_opt

# Calculate sector data
sectors = 3
sector_times = []
sector_speeds = []
for sector in range(sectors):
    start_idx = int(sector * N / sectors)
    end_idx = int((sector + 1) * N / sectors)
    sector_time = sum(ds_array[start_idx:end_idx] / v_opt[start_idx:end_idx])
    sector_avg_speed = np.mean(v_opt[start_idx:end_idx])
    sector_times.append(sector_time)
    sector_speeds.append(sector_avg_speed * 3.6)

# ===== COMBINED FIGURE: 3 rows x 4 columns =====
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)

x_left_bound = x_center - w_left * normals[:, 0]
y_left_bound = y_center - w_left * normals[:, 1]
x_right_bound = x_center + w_right * normals[:, 0]
y_right_bound = y_center + w_right * normals[:, 1]

# ===== ROW 1-2, COL 1-2: F1 OPTIMAL RACING LINE (LARGE) =====
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.fill(np.concatenate([x_left_bound, x_right_bound[::-1]]),
         np.concatenate([y_left_bound, y_right_bound[::-1]]),
         color='#1a1a1a', edgecolor='white', linewidth=2, alpha=0.8)

points = np.array([x_opt, y_opt]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='plasma', linewidth=4)
lc.set_array(v_opt[:-1] * 3.6)
line = ax1.add_collection(lc)
cbar = plt.colorbar(line, ax=ax1, pad=0.02)
cbar.set_label('Speed [km/h]', rotation=270, labelpad=20, fontsize=11)

ax1.scatter(x_opt[0], y_opt[0], c='lime', s=300, marker='s', 
           edgecolors='white', linewidth=2, zorder=10, label='Start/Finish')

arrow_spacing = max(N // 12, 1)
for i in range(0, N, arrow_spacing):
    if i < N-1:
        dx = x_opt[(i+1)%N] - x_opt[i]
        dy = y_opt[(i+1)%N] - y_opt[i]
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        ax1.arrow(x_opt[i], y_opt[i], 3*dx/norm, 3*dy/norm, 
                 head_width=2, head_length=2.5, fc='cyan', ec='cyan', 
                 alpha=0.6, linewidth=1.5)

ax1.set_xlabel("X [m]", fontsize=13, fontweight='bold')
ax1.set_ylabel("Y [m]", fontsize=13, fontweight='bold')
ax1.axis('equal')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_title("F1 Optimal Racing Line", fontsize=15, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.2, linestyle='--')

# ===== ROW 1-2, COL 3: F1 PERFORMANCE SUMMARY =====
ax_stats = fig.add_subplot(gs[0:2, 2])
ax_stats.axis('off')

stats_text = f"""F1 PERFORMANCE
SUMMARY

LAP TIME
{lap_time_seconds:.2f} sec

SPEED
Avg: {np.mean(v_opt)*3.6:.1f} km/h
Max: {np.max(v_opt)*3.6:.1f} km/h
Min: {np.min(v_opt)*3.6:.1f} km/h

G-FORCES
Long: {np.max(np.abs(a_lon_opt))/vehicle.gravity:.2f}G
Lat: {np.max(a_lat_opt)/vehicle.gravity:.2f}G
Total: {np.max(a_total)/vehicle.gravity:.2f}G

FORCES
Downforce: {np.max(downforce_g):.2f}G
Track: {track_length:.1f}m

SPECS
Mass: {vehicle.mass_kg:.0f}kg
Power: {vehicle.engine_power_watts/1000:.0f}kW
μ: {vehicle.mu_friction:.2f}
C_L: {vehicle.cL_downforce:.2f}
C_D: {vehicle.cD_drag:.2f}"""

ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                      edgecolor='cyan', linewidth=2, pad=1))

# ===== ROW 1, COL 4: SECTOR TIMES =====
ax5 = fig.add_subplot(gs[2, 3])
colors = ['gold', 'silver', '#CD7F32']
bars = ax5.bar(range(1, sectors+1), sector_times, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax5.set_xlabel("Sector", fontsize=11, fontweight='bold')
ax5.set_ylabel("Time [s]", fontsize=11, fontweight='bold')
ax5.set_title("Sector Times", fontsize=13, fontweight='bold')
ax5.set_xticks(range(1, sectors+1))

for bar, time, speed in zip(bars, sector_times, sector_speeds):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.2f}s\n{speed:.0f}km/h',
            ha='center', va='bottom', fontsize=9)
ax5.set_ylim([20, 45])
ax5.grid(True, alpha=0.2, axis='y')

# ===== ROW 2, COL 4: DOWNFORCE =====
ax6 = fig.add_subplot(gs[1, 3])
ax6.fill_between(s, 0, downforce_g, color='purple', alpha=0.4)
ax6.plot(s, downforce_g, color='purple', linewidth=2.5)
ax6.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax6.set_ylabel("Downforce [G]", fontsize=11, fontweight='bold')
ax6.set_title("Aerodynamic Downforce", fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.2)

# ===== ROW 3, COL 1: SPEED PROFILE =====
ax2 = fig.add_subplot(gs[2, 0])
ax2.fill_between(s, 0, v_opt*3.6, color='cyan', alpha=0.3)
ax2.plot(s, v_opt*3.6, color='cyan', linewidth=2.5)
ax2.axhline(np.mean(v_opt)*3.6, color='yellow', linestyle='--', 
           linewidth=1.5, alpha=0.7, label=f'Avg: {np.mean(v_opt)*3.6:.1f}')
ax2.set_xlabel("Distance [m]", fontsize=10, fontweight='bold')
ax2.set_ylabel("Speed [km/h]", fontsize=10, fontweight='bold')
ax2.set_title("Speed Profile", fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)
ax2.set_ylim([0, max(v_opt*3.6)*1.1])

# ===== ROW 3, COL 2: G-FORCE =====
ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(s, a_lon_opt/vehicle.gravity, color='orange', linewidth=2, 
         label='Longitudinal', alpha=0.8)
ax3.plot(s, (a_lat_opt/vehicle.gravity)/100, color='magenta', linewidth=2, 
         label='Lateral', alpha=0.8)
ax3.plot(s, (a_total/vehicle.gravity)/100, color='red', linewidth=2.5, 
         label='Total', linestyle='--')
ax3.axhline(vehicle.mu_friction, color='lime', linestyle=':', 
           linewidth=2, alpha=0.6, label='Grip')
ax3.set_xlabel("Distance [m]", fontsize=10, fontweight='bold')
ax3.set_ylabel("G-Force", fontsize=10, fontweight='bold')
ax3.set_title("G-Force Profile", fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(True, alpha=0.2)
ax3.set_ylim([-6, 6])

# ===== ROW 3, COL 3: POWER USAGE =====
ax4 = fig.add_subplot(gs[2, 2])
ax4.fill_between(s, 0, np.maximum(power_used/1000, 0), 
                 color='green', alpha=0.4, label='Accel')
ax4.fill_between(s, 0, np.minimum(power_used/1000, 0), 
                 color='red', alpha=0.4, label='Brake')
ax4.plot(s, power_used/1000, color='white', linewidth=2)
ax4.axhline(vehicle.engine_power_watts/1000, color='green', 
           linestyle='--', alpha=0.7, linewidth=1.5)
ax4.axhline(-vehicle.brake_power_watts/1000, color='red', 
           linestyle='--', alpha=0.7, linewidth=1.5)
ax4.set_xlabel("Distance [m]", fontsize=10, fontweight='bold')
ax4.set_ylabel("Power [kW]", fontsize=10, fontweight='bold')
ax4.set_title("Power Usage", fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.2)

# ===== ROW 3, COL 4: TRACK INFO (NEW) =====
ax7 = fig.add_subplot(gs[0, 3])
ax7.axis('off')

track_info = f"""TRACK INFO

Length: {track_length:.1f}m
Segments: {N}
Avg Δs: {ds:.2f}m

Max Curvature:
{np.max(np.abs(curvature)):.4f} m⁻¹"""

ax7.text(0.1, 0.5, track_info, transform=ax7.transAxes,
         fontsize=10, family='monospace', verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                  edgecolor='orange', linewidth=2, pad=1))

plt.suptitle("F1 LAP TIME OPTIMIZATION - COMPLETE ANALYSIS", 
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'='*60}")
print(f"F1 LAP TIME OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"LAP TIME:       {lap_time_seconds:.2f} seconds")
print(f"Average speed:  {np.mean(v_opt):.2f} m/s ({np.mean(v_opt)*3.6:.1f} km/h)")
print(f"Top speed:      {np.max(v_opt):.2f} m/s ({np.max(v_opt)*3.6:.1f} km/h)")
print(f"Min speed:      {np.min(v_opt):.2f} m/s ({np.min(v_opt)*3.6:.1f} km/h)")
print(f"{'='*60}")
print(f"Max acceleration:  {np.max(a_lon_opt):.2f} m/s² ({np.max(a_lon_opt)/vehicle.gravity:.2f}G)")
print(f"Max braking:       {abs(np.min(a_lon_opt)):.2f} m/s² ({abs(np.min(a_lon_opt))/vehicle.gravity:.2f}G)")
print(f"Max lateral G:     {np.max(a_lat_opt)/vehicle.gravity:.2f}G")
print(f"Max combined G:    {np.max(a_total)/vehicle.gravity:.2f}G")
print(f"Max downforce:     {np.max(downforce_g):.2f}G")
print(f"Track length:      {track_length:.1f}m")
print(f"{'='*60}\n")