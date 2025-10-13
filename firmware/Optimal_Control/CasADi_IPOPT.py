import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
try:
    from .constraints import add_constraints
except Exception:
    try:
        # Try absolute package path from project root
        from firmware.Optimal_Control.constraints import add_constraints
    except Exception:
        # Last resort: running as a script from this directory
        import os, sys
        sys.path.append(os.path.dirname(__file__))
        from constraints import add_constraints

# --- Vehicle Parameters Dataclass (Realistic F1 specs) ---
@dataclass
class VehicleParams:
    # Modern F1 car specifications (2024 regulations)
    mass_kg: float = 798.0  # Minimum weight with driver
    mu_friction: float = 1.8  # Peak friction with warm slicks + downforce
    gravity: float = 9.81
    rho_air: float = 1.225
    
    # Aerodynamics - Modern F1 car
    cL_downforce: float = 3.0  # Realistic F1 downforce coefficient
    cD_drag: float = 1.2  # F1 drag coefficient
    frontal_area_m2: float = 1.5
    
    # Power and braking
    engine_power_watts: float = 750000.0  # 750 kW (~1000 hp) - F1 power unit
    brake_power_watts: float = 2500000.0  # 2.5 MW - F1 braking capability
    
    # Acceleration limits based on F1 telemetry data
    a_accel_max: float = 12.0  # Max longitudinal acceleration ~2.5G
    a_brake_max: float = 45.0  # Max braking ~5G
    a_lat_max: float = 60.0  # Max lateral ~5G
    
    # Other parameters
    c_rr: float = 0.02  # Rolling resistance for F1 tires
    wheelbase_m: float = 3.6  # F1 wheelbase
    
    # Speed limits
    v_min: float = 15.0  # Minimum speed to maintain (m/s) ~54 km/h
    v_max: float = 100.0  # Maximum speed (m/s) ~360 km/h

    def k_aero(self) -> float:
        """Downforce coefficient"""
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / self.mass_kg

    def k_drag(self) -> float:
        """Drag force coefficient"""
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2

# Initialize vehicle parameters
vehicle = VehicleParams()

# --- Read Track Data ---
try:
    track = pd.read_csv("Turn_1.csv", comment="#", names=["x_m","y_m","w_tr_right_m","w_tr_left_m"])
except:
    print("Warning: Could not find Turn_1.csv, using synthetic track")
    # Create a simple test track if file not found
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

# Calculate segment lengths more accurately
ds_array = np.zeros(N)
for i in range(N):
    i_next = (i + 1) % N
    dx = x_center[i_next] - x_center[i]
    dy = y_center[i_next] - y_center[i]
    ds_array[i] = np.sqrt(dx**2 + dy**2)

ds = np.mean(ds_array)  # Average segment length
print(f"Track segments: {N}, Average segment length: {ds:.2f}m")

# --- Pre-compute track properties ---
normals = np.zeros((N, 2))
curvature = np.zeros(N)

# Calculate normals
for i in range(N):
    i_prev = (i - 1) % N
    i_next = (i + 1) % N
    
    dx = x_center[i_next] - x_center[i_prev]
    dy = y_center[i_next] - y_center[i_prev]
    
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    normals[i, 0] = -dy / norm
    normals[i, 1] = dx / norm

# Calculate curvature with smoothing
for i in range(N):
    i_prev = (i - 1) % N
    i_next = (i + 1) % N
    
    x0, y0 = x_center[i_prev], y_center[i_prev]
    x1, y1 = x_center[i], y_center[i]
    x2, y2 = x_center[i_next], y_center[i_next]
    
    # Use circle fitting for curvature
    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x1, y2 - y1
    
    # Cross product for signed curvature
    cross = dx1*dy2 - dy1*dx2
    
    # Distances
    ds1 = np.sqrt(dx1**2 + dy1**2) + 1e-9
    ds2 = np.sqrt(dx2**2 + dy2**2) + 1e-9
    
    # Menger curvature formula
    curvature[i] = 2 * cross / (ds1 * ds2 * (ds1 + ds2) + 1e-9)

# Smooth curvature to avoid numerical issues
from scipy.ndimage import gaussian_filter1d
curvature = gaussian_filter1d(curvature, sigma=2, mode='wrap')

# --- CasADi Problem Setup ---
opti = ca.Opti()

# Decision variables
n = opti.variable(N)  # Normal offset from centerline
v = opti.variable(N)  # Velocity at each point
a_lon = opti.variable(N)  # Longitudinal acceleration

# Slack variables for soft constraints (helps convergence)
slack_power = opti.variable(N)
opti.set_initial(slack_power, 0)
for i in range(N):
    opti.subject_to(slack_power[i] >= 0)
    opti.subject_to(slack_power[i] <= 50000)  # reduce from 100000 to 50k


# --- Objective: Minimize lap time with regularization ---
lap_time = ca.sum1(ds_array / v)
path_diffs = [ca.sqrt((n[i+1] - n[i])**2 + ds_array[i]**2) for i in range(N-1)]
path_length_penalty = 0.01 * ca.sum1(ca.vertcat(*path_diffs))


# Regularization terms (small weights to help convergence)
reg_n = 1e-8 * ca.sum1(n**2)  # Prefer staying near centerline
reg_a = 1e-6 * ca.sum1(a_lon**2)  # Smooth acceleration
reg_slack = 1e3 * ca.sum1(slack_power**2)  # Penalize slack usage

opti.minimize(lap_time + path_length_penalty + reg_n + reg_a + reg_slack)

"""Constraints moved to dedicated module and imported above."""
vehicle.mu_friction = 2.2
vehicle.cL_downforce = 3.0
vehicle.a_accel_max = 12.0
vehicle.a_brake_max = 45.0

# Add all constraints and get a_lat back for plotting/stats
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

# --- Initial guess (CRITICAL for convergence) ---
# Start with a reasonable racing line and speed profile

# Initial position: slight racing line based on curvature
n_init = np.zeros(N)
for i in range(N):
    if curvature[i] > 0.001:  # Ляв завой
        n_init[i] = w_left[i] * 0.8  # Започни от 80% вътре
    elif curvature[i] < -0.001:  # Десен завой
        n_init[i] = -w_right[i] * 0.8
    else:
        n_init[i] = 0

# Initial velocity: vary based on curvature
v_init = np.zeros(N)
for i in range(N):
    abs_curv = abs(curvature[i])
    if abs_curv < 1e-4:  # Straight
        v_init[i] = 70.0  # 80 m/s on straights
    else:
        # Estimate corner speed from curvature
        # v_corner = sqrt(a_lat_max / curvature)
        a_lat_available = vehicle.a_lat_max
        v_corner = np.sqrt(a_lat_available / (abs_curv + 1e-6))
        v_init[i] = min(v_corner, 80.0)

# Smooth the velocity profile
v_init = gaussian_filter1d(v_init, sigma=5, mode='wrap')
v_init = np.clip(v_init, vehicle.v_min, vehicle.v_max)
opti.set_initial(v, v_init)

# Initial acceleration based on velocity gradient
a_init = np.zeros(N)
for i in range(N):
    i_next = (i + 1) % N
    dv = v_init[i_next] - v_init[i]
    dt = ds_array[i] / (v_init[i] + 1e-3)
    a_init[i] = dv / dt

a_init = np.clip(a_init, -vehicle.a_brake_max, vehicle.a_accel_max)

# Set initial values
opti.set_initial(n, n_init)
opti.set_initial(a_lon, a_init)

# --- Solver options ---
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
    'ipopt.hessian_approximation': 'exact',  # Use L-BFGS for better convergence
    'ipopt.limited_memory_max_history': 50,
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.line_search_method': 'filter',
    'ipopt.alpha_for_y': 'primal',
    'error_on_fail': False
}

opti.solver('ipopt', opts)

# --- Solve ---
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

# --- Extract solution ---
n_opt = sol.value(n)
v_opt = sol.value(v)
a_lon_opt = sol.value(a_lon)
a_lat_opt = sol.value(a_lat)

# Reconstruct trajectory
x_opt = x_center + n_opt * normals[:, 0]
y_opt = y_center + n_opt * normals[:, 1]

# Calculate lap time and other metrics
lap_time_seconds = float(sol.value(lap_time))
track_length = np.sum(ds_array)

# Additional calculations
s = np.cumsum(np.concatenate([[0], ds_array[:-1]]))
a_total = np.sqrt(a_lon_opt**2 + a_lat_opt**2)

# Forces and power
F_normal = vehicle.mass_kg * (vehicle.gravity + vehicle.k_aero() * v_opt**2)
downforce_g = vehicle.k_aero() * v_opt**2 / vehicle.gravity
F_drag = vehicle.k_drag() * v_opt**2
power_used = vehicle.mass_kg * a_lon_opt * v_opt + F_drag * v_opt

# --- Visualization ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Track boundaries
x_left_bound = x_center - w_left * normals[:, 0]
y_left_bound = y_center - w_left * normals[:, 1]
x_right_bound = x_center + w_right * normals[:, 0]
y_right_bound = y_center + w_right * normals[:, 1]

# 1. Main track view
ax1 = fig.add_subplot(gs[:2, 0:2])
ax1.fill(np.concatenate([x_left_bound, x_right_bound[::-1]]),
         np.concatenate([y_left_bound, y_right_bound[::-1]]),
         color='#1a1a1a', edgecolor='white', linewidth=2, alpha=0.8)

# Racing line colored by speed
points = np.array([x_opt, y_opt]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='plasma', linewidth=4)
lc.set_array(v_opt[:-1] * 3.6)  # Convert to km/h for display
line = ax1.add_collection(lc)
cbar = plt.colorbar(line, ax=ax1, pad=0.02)
cbar.set_label('Speed [km/h]', rotation=270, labelpad=20, fontsize=12)

# Start/finish line
ax1.scatter(x_opt[0], y_opt[0], c='lime', s=300, marker='s', 
           edgecolors='white', linewidth=2, zorder=10, label='Start/Finish')

# Direction arrows
arrow_spacing = max(N // 12, 1)
for i in range(0, N, arrow_spacing):
    if i < N-1:
        dx = x_opt[(i+1)%N] - x_opt[i]
        dy = y_opt[(i+1)%N] - y_opt[i]
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        ax1.arrow(x_opt[i], y_opt[i], 3*dx/norm, 3*dy/norm, 
                 head_width=2, head_length=2.5, fc='cyan', ec='cyan', 
                 alpha=0.6, linewidth=1.5)

ax1.set_xlabel("X [m]", fontsize=14, fontweight='bold')
ax1.set_ylabel("Y [m]", fontsize=14, fontweight='bold')
ax1.axis('equal')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_title("F1 Optimal Racing Line", fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.2, linestyle='--')

# 2. Speed profile
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(s, 0, v_opt*3.6, color='cyan', alpha=0.3)
ax2.plot(s, v_opt*3.6, color='cyan', linewidth=2.5)
ax2.axhline(np.mean(v_opt)*3.6, color='yellow', linestyle='--', 
           linewidth=1.5, alpha=0.7, label=f'Avg: {np.mean(v_opt)*3.6:.1f} km/h')
ax2.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax2.set_ylabel("Speed [km/h]", fontsize=11, fontweight='bold')
ax2.set_title("Speed Profile", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim([0, max(v_opt*3.6)*1.1])

# 3. G-forces
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(s, a_lon_opt/vehicle.gravity, color='orange', linewidth=2, 
         label='Longitudinal', alpha=0.8)
ax3.plot(s, a_lat_opt/vehicle.gravity, color='magenta', linewidth=2, 
         label='Lateral', alpha=0.8)
ax3.plot(s, a_total/vehicle.gravity, color='red', linewidth=2.5, 
         label='Total', linestyle='--')
ax3.axhline(vehicle.mu_friction, color='lime', linestyle=':', 
           linewidth=2, alpha=0.6, label='Max grip')
ax3.axhline(-vehicle.a_brake_max/vehicle.gravity, color='lime', 
           linestyle=':', linewidth=2, alpha=0.6)
ax3.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax3.set_ylabel("G-Force", fontsize=11, fontweight='bold')
ax3.set_title("G-Force Profile", fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.2)
ax3.set_ylim([-6, 6])

# 4. Power usage
ax4 = fig.add_subplot(gs[2, 0])
ax4.fill_between(s, 0, np.maximum(power_used/1000, 0), 
                 color='green', alpha=0.4, label='Acceleration')
ax4.fill_between(s, 0, np.minimum(power_used/1000, 0), 
                 color='red', alpha=0.4, label='Braking')
ax4.plot(s, power_used/1000, color='white', linewidth=2)
ax4.axhline(vehicle.engine_power_watts/1000, color='green', 
           linestyle='--', alpha=0.7, label='Engine limit')
ax4.axhline(-vehicle.brake_power_watts/1000, color='red', 
           linestyle='--', alpha=0.7, label='Brake limit')
ax4.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax4.set_ylabel("Power [kW]", fontsize=11, fontweight='bold')
ax4.set_title("Power Usage", fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2)

# 5. Downforce
ax5 = fig.add_subplot(gs[2, 1])
ax5.fill_between(s, 0, downforce_g, color='purple', alpha=0.4)
ax5.plot(s, downforce_g, color='purple', linewidth=2.5)
ax5.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax5.set_ylabel("Downforce [G]", fontsize=11, fontweight='bold')
ax5.set_title("Aerodynamic Downforce", fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.2)

# 6. Sector times
ax6 = fig.add_subplot(gs[2, 2])
sectors = 3
sector_length = track_length / sectors
sector_times = []
sector_speeds = []
for sector in range(sectors):
    start_idx = int(sector * N / sectors)
    end_idx = int((sector + 1) * N / sectors)
    sector_time = sum(ds_array[start_idx:end_idx] / v_opt[start_idx:end_idx])
    sector_avg_speed = np.mean(v_opt[start_idx:end_idx])
    sector_times.append(sector_time)
    sector_speeds.append(sector_avg_speed * 3.6)

colors = ['gold', 'silver', '#CD7F32']
bars = ax6.bar(range(1, sectors+1), sector_times, color=colors, alpha=0.8)
ax6.set_xlabel("Sector", fontsize=11, fontweight='bold')
ax6.set_ylabel("Time [s]", fontsize=11, fontweight='bold')
ax6.set_title("Sector Times", fontsize=13, fontweight='bold')
ax6.set_xticks(range(1, sectors+1))

for i, (bar, time, speed) in enumerate(zip(bars, sector_times, sector_speeds)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.2f}s\n{speed:.1f}km/h',
            ha='center', va='bottom', fontsize=10)

ax6.grid(True, alpha=0.2, axis='y')

# Performance stats
stats_text = f"""
╔════════════════════════════════════╗
║      F1 PERFORMANCE SUMMARY         ║
╠════════════════════════════════════╣
║ LAP TIME:      {lap_time_seconds:7.2f} s         ║
║ AVG SPEED:     {np.mean(v_opt)*3.6:7.1f} km/h      ║
║ TOP SPEED:     {np.max(v_opt)*3.6:7.1f} km/h      ║
║ MIN SPEED:     {np.min(v_opt)*3.6:7.1f} km/h      ║
╠════════════════════════════════════╣
║ MAX G-FORCE:                        ║
║   Longitudinal: {np.max(np.abs(a_lon_opt))/vehicle.gravity:5.2f} G          ║
║   Lateral:      {np.max(a_lat_opt)/vehicle.gravity:5.2f} G          ║
║   Combined:     {np.max(a_total)/vehicle.gravity:5.2f} G          ║
╠════════════════════════════════════╣
║ MAX DOWNFORCE: {np.max(downforce_g):6.2f} G         ║
║ TRACK LENGTH:  {track_length:7.1f} m        ║
╠════════════════════════════════════╣
║ VEHICLE SPECS:                      ║
║   Mass:        {vehicle.mass_kg:6.0f} kg         ║
║   Power:       {vehicle.engine_power_watts/1000:6.0f} kW         ║
║   Peak μ:      {vehicle.mu_friction:6.2f}            ║
║   C_L:         {vehicle.cL_downforce:6.2f}            ║
║   C_D:         {vehicle.cD_drag:6.2f}            ║
╚════════════════════════════════════╝
"""

fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#2a2a2a', 
                  edgecolor='cyan', linewidth=2),
         verticalalignment='bottom')

plt.suptitle("🏎️  F1 LAP TIME OPTIMIZATION  🏁", 
            fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'='*60}")
print(f"🏁 F1 LAP TIME OPTIMIZATION COMPLETE 🏁")
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