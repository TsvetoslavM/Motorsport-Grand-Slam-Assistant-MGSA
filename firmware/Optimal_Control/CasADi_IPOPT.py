import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- Vehicle Parameters Dataclass ---
@dataclass
class VehicleParams:
    mass_kg: float = 798.0
    mu_friction: float = 2.0
    gravity: float = 9.81
    rho_air: float = 1.225
    cL_downforce: float = 4.0
    frontal_area_m2: float = 1.6
    engine_power_watts: float = 735000.0
    a_brake_max: float = 54.0
    a_accel_cap: float = 20.0
    cD_drag: float = 1.0
    c_rr: float = 0.004
    safety_speed_margin: float = 1.00
    brake_power_watts: float = 1200000.0
    wheelbase_m: float = 2.5

    def k_aero(self) -> float:
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / max(self.mass_kg, 1e-9)

    def k_drag(self) -> float:
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2

# Initialize vehicle parameters
vehicle = VehicleParams()

# --- Read CSV ---
track = pd.read_csv("Turn_1.csv", comment="#", names=["x_m","y_m","w_tr_right_m","w_tr_left_m"])
x_center = track["x_m"].values
y_center = track["y_m"].values
w_right = track["w_tr_right_m"].values
w_left = track["w_tr_left_m"].values

N = len(x_center)
ds = 1.0

# --- Extract vehicle parameters ---
L = vehicle.wheelbase_m
mu = vehicle.mu_friction
g = vehicle.gravity
mass = vehicle.mass_kg
k_aero = vehicle.k_aero()
k_drag = vehicle.k_drag()
c_rr = vehicle.c_rr
P_engine = vehicle.engine_power_watts * vehicle.safety_speed_margin
P_brake = vehicle.brake_power_watts
a_brake_max = vehicle.a_brake_max
a_accel_cap = vehicle.a_accel_cap
v_min, v_max = 1.0, 100.0

# --- Pre-compute track normals ---
normals = np.zeros((N, 2))
for i in range(N):
    if i == 0:
        dx = x_center[1] - x_center[0]
        dy = y_center[1] - y_center[0]
    elif i == N-1:
        dx = x_center[-1] - x_center[-2]
        dy = y_center[-1] - y_center[-2]
    else:
        dx = x_center[i+1] - x_center[i-1]
        dy = y_center[i+1] - y_center[i-1]
    
    norm = np.sqrt(dx**2 + dy**2)
    normals[i, 0] = -dy / norm
    normals[i, 1] = dx / norm

# --- CasADi variables ---
n = ca.MX.sym('n', N)  # Normal offset from centerline
v = ca.MX.sym('v', N)  # Velocity
a = ca.MX.sym('a', N)  # Acceleration

# --- Objective: minimize time ---
J = ca.sum1(ds / v)

# --- Constraints ---
g_dyn = []
lbg = []
ubg = []

# 1. Simple kinematic constraints (velocity dynamics only)
for i in range(N-1):
    dt = ds / v[i]
    g_dyn.append(v[i] + a[i] * dt - v[i+1])
    lbg.append(0.0)
    ubg.append(0.0)

# 2. Track boundaries
for i in range(N):
    g_dyn.append(n[i])
    lbg.append(-w_left[i])
    ubg.append(w_right[i])

# 3. Simplified friction circle with downforce
a_normal_max = g + k_aero * v_max**2
friction_limit_sq = (mu * a_normal_max)**2

for i in range(N):
    kappa = 0.001
    a_lat = kappa * v[i]**2 if i < N-1 else 0.0
    
    g_dyn.append(a_lat**2 + a[i]**2)
    lbg.append(0.0)
    ubg.append(float(friction_limit_sq))

# 4. Power constraints
for i in range(N):
    g_dyn.append(a[i] * v[i])
    lbg.append(-P_brake / mass)
    ubg.append(P_engine / mass)

# 5. Acceleration limits
for i in range(N):
    g_dyn.append(a[i])
    lbg.append(-a_brake_max)
    ubg.append(a_accel_cap)

# 6. Smoothness: limit change in offset
for i in range(N-1):
    dn = n[i+1] - n[i]
    g_dyn.append(dn)
    lbg.append(-0.5)
    ubg.append(0.5)

# 7. Smoothness: limit second derivative
for i in range(N-2):
    d2n = n[i+2] - 2*n[i+1] + n[i]
    g_dyn.append(d2n)
    lbg.append(-0.1)
    ubg.append(0.1)

# 8. Periodic boundary conditions
g_dyn.append(n[0] - n[-1])
g_dyn.append(v[0] - v[-1])
lbg += [0.0, 0.0]
ubg += [0.0, 0.0]

# --- Variable bounds ---
lbx, ubx = [], []
for i in range(N):
    lbx += [-w_left[i], v_min, -10]
    ubx += [w_right[i], v_max, 10]

# --- Formulate NLP ---
opt_vars = ca.vertcat(n, v, a)
nlp = {'x': opt_vars, 'f': J, 'g': ca.vertcat(*g_dyn)}

solver = ca.nlpsol('solver', 'ipopt', nlp, {
    'ipopt.max_iter': 3000,
    'ipopt.tol': 1e-4,
    'ipopt.print_level': 5,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.constr_viol_tol': 1e-3
})

# --- Initialization ---
x0 = np.zeros(opt_vars.size()[0])
x0[0:N] = 0.0
x0[N:2*N] = 20.0
x0[2*N:3*N] = 0.0

# --- Solve ---
lbx = np.array(lbx, dtype=float)
ubx = np.array(ubx, dtype=float)
lbg = np.array(lbg, dtype=float)
ubg = np.array(ubg, dtype=float)

sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
sol_x = np.array(sol['x']).flatten()

# --- Extract results ---
n_opt = sol_x[0:N]
v_opt = sol_x[N:2*N]
a_opt = sol_x[2*N:3*N]

# Reconstruct actual trajectory
x_opt = x_center + n_opt * normals[:, 0]
y_opt = y_center + n_opt * normals[:, 1]

# Calculate additional metrics
s = np.cumsum(np.concatenate([[0], np.ones(N-1) * ds]))

# Calculate lateral acceleration from path curvature
a_lat = np.zeros(N)
for i in range(1, N-1):
    dx1 = x_opt[i] - x_opt[i-1]
    dy1 = y_opt[i] - y_opt[i-1]
    dx2 = x_opt[i+1] - x_opt[i]
    dy2 = y_opt[i+1] - y_opt[i]
    
    norm1 = np.sqrt(dx1**2 + dy1**2) + 1e-9
    norm2 = np.sqrt(dx2**2 + dy2**2) + 1e-9
    
    cos_angle = (dx1*dx2 + dy1*dy2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_change = np.arccos(cos_angle)
    
    kappa = angle_change / ds
    a_lat[i] = kappa * v_opt[i]**2

# Calculate forces and downforce
F_normal = mass * g + k_aero * mass * v_opt**2
F_drag = k_drag * v_opt**2
F_roll = c_rr * F_normal
downforce_g = (F_normal - mass * g) / (mass * g)

a_total = np.sqrt(a_lat**2 + a_opt**2)

# --- Cool Visualization ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Calculate track boundaries
x_left_bound = x_center - w_left * normals[:, 0]
y_left_bound = y_center - w_left * normals[:, 1]
x_right_bound = x_center + w_right * normals[:, 0]
y_right_bound = y_center + w_right * normals[:, 1]

# 1. Main track view with velocity coloring
ax1 = fig.add_subplot(gs[:2, 0:2])
ax1.fill(np.concatenate([x_left_bound, x_right_bound[::-1]]),
         np.concatenate([y_left_bound, y_right_bound[::-1]]),
         color='#1a1a1a', edgecolor='white', linewidth=2, alpha=0.8)

points = np.array([x_opt, y_opt]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap='plasma', linewidth=4)
lc.set_array(v_opt[:-1])
line = ax1.add_collection(lc)
cbar = plt.colorbar(line, ax=ax1, pad=0.02)
cbar.set_label('Velocity [m/s]', rotation=270, labelpad=20, fontsize=12)

ax1.scatter(x_opt[0], y_opt[0], c='lime', s=300, marker='s', 
           edgecolors='white', linewidth=2, zorder=10, label='Start/Finish')

# Add arrows showing direction
arrow_spacing = max(N // 8, 1)
for i in range(0, N, arrow_spacing):
    if i < N-1:
        dx = x_opt[i+1] - x_opt[i]
        dy = y_opt[i+1] - y_opt[i]
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        dx = 3 * dx / norm
        dy = 3 * dy / norm
        ax1.arrow(x_opt[i], y_opt[i], dx, dy, head_width=1.5, 
                 head_length=2, fc='cyan', ec='cyan', alpha=0.6, linewidth=1.5)

ax1.set_xlabel("X [m]", fontsize=14, fontweight='bold')
ax1.set_ylabel("Y [m]", fontsize=14, fontweight='bold')
ax1.axis('equal')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_title("Optimal Racing Line", fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.2, linestyle='--')

# 2. Velocity profile
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(s, 0, v_opt, color='cyan', alpha=0.3)
ax2.plot(s, v_opt, color='cyan', linewidth=2.5)
ax2.axhline(np.mean(v_opt), color='yellow', linestyle='--', 
           linewidth=1.5, alpha=0.7, label=f'Avg: {np.mean(v_opt):.1f} m/s')
ax2.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax2.set_ylabel("Velocity [m/s]", fontsize=11, fontweight='bold')
ax2.set_title("Velocity Profile", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# 3. Acceleration profile
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(s, a_opt, color='orange', linewidth=2, label='Longitudinal', alpha=0.8)
ax3.plot(s, a_lat, color='magenta', linewidth=2, label='Lateral', alpha=0.8)
ax3.plot(s, a_total, color='red', linewidth=2.5, label='Total', linestyle='--')
max_g = mu * a_normal_max
ax3.axhline(max_g, color='lime', linestyle=':', linewidth=2, alpha=0.6, label='Friction limit')
ax3.axhline(-max_g, color='lime', linestyle=':', linewidth=2, alpha=0.6)
ax3.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax3.set_ylabel("Acceleration [m/s²]", fontsize=11, fontweight='bold')
ax3.set_title("Acceleration Profile", fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.2)

# 4. Downforce profile
ax4 = fig.add_subplot(gs[2, 0])
ax4.fill_between(s, 0, downforce_g, color='purple', alpha=0.4)
ax4.plot(s, downforce_g, color='purple', linewidth=2.5)
ax4.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax4.set_ylabel("Downforce [G]", fontsize=11, fontweight='bold')
ax4.set_title("Aerodynamic Downforce", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.2)

# 5. Drag forces
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(s, F_drag/1000, color='red', linewidth=2, label='Aero Drag')
ax5.plot(s, F_roll/1000, color='orange', linewidth=2, label='Rolling Resistance')
ax5.plot(s, (F_drag + F_roll)/1000, color='yellow', linewidth=2.5, 
        linestyle='--', label='Total Resistance')
ax5.set_xlabel("Distance [m]", fontsize=11, fontweight='bold')
ax5.set_ylabel("Force [kN]", fontsize=11, fontweight='bold')
ax5.set_title("Resistance Forces", fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.2)

# Add performance statistics box
avg_power_kw = np.mean(mass * a_opt * v_opt) / 1000
max_downforce_g = np.max(downforce_g)
stats_text = f"""
╔════════════════════════════════════╗
║      VEHICLE PERFORMANCE STATS      ║
╠════════════════════════════════════╣
║ Lap Time:      {float(sol['f']):6.2f} s          ║
║ Avg Speed:     {np.mean(v_opt):6.2f} m/s        ║
║                {np.mean(v_opt)*3.6:6.1f} km/h       ║
║ Max Speed:     {np.max(v_opt):6.2f} m/s        ║
║                {np.max(v_opt)*3.6:6.1f} km/h       ║
║ Max Accel:     {np.max(a_total):6.2f} m/s²      ║
║ Max Brake:     {abs(np.min(a_opt)):6.2f} m/s²      ║
║ Max Downforce: {max_downforce_g:6.2f} G          ║
║ Avg Power:     {avg_power_kw:6.1f} kW         ║
║ Track Length:  {s[-1]:6.1f} m          ║
╠════════════════════════════════════╣
║ Mass:          {mass:6.1f} kg         ║
║ Power:         {P_engine/1000:6.1f} kW         ║
║ Friction μ:    {mu:6.2f}             ║
║ Downforce C_L: {vehicle.cL_downforce:6.2f}             ║
╚════════════════════════════════════╝
"""
fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='cyan', linewidth=2),
         verticalalignment='bottom')

plt.suptitle("🏎️  Race Line Optimization Analysis  🏁", 
            fontsize=20, fontweight='bold', y=0.98)

plt.show()

print(f"\n{'='*60}")
print(f"🏁 RACE LINE OPTIMIZATION COMPLETE 🏁")
print(f"{'='*60}")
print(f"Vehicle: {vehicle.mass_kg}kg, {vehicle.engine_power_watts/1000:.0f}kW, μ={vehicle.mu_friction}")
print(f"{'='*60}")
print(f"Lap time:       {float(sol['f']):.2f} seconds")
print(f"Average speed:  {np.mean(v_opt):.2f} m/s ({np.mean(v_opt)*3.6:.1f} km/h)")
print(f"Max velocity:   {np.max(v_opt):.2f} m/s ({np.max(v_opt)*3.6:.1f} km/h)")
print(f"Min velocity:   {np.min(v_opt):.2f} m/s ({np.min(v_opt)*3.6:.1f} km/h)")
print(f"Max downforce:  {max_downforce_g:.2f} G")
print(f"Max accel:      {np.max(a_total):.2f} m/s²")
print(f"Max brake:      {abs(np.min(a_opt)):.2f} m/s²")
print(f"Track length:   {s[-1]:.1f} m")
print(f"{'='*60}\n")