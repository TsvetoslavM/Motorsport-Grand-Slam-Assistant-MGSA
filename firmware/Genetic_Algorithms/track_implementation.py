import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import math

# ===== Vehicle Parameters =====
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

    def k_aero(self):
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / max(self.mass_kg, 1e-9)

    def k_drag(self):
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2

vehicle = VehicleParams()

# ===== Load centerline =====
def load_track_with_boundaries(filename):
    """
    Load track from CSV with columns:
    x_m, y_m, w_tr_right_m, w_tr_left_m
    Returns: center, out_boundary, in_boundary as numpy arrays
    """
    df = pd.read_csv(filename)
    required_cols = ['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    center = df[['x_m', 'y_m']].to_numpy()
    
    # Compute direction vectors along track
    tangents = np.gradient(center, axis=0)
    normals = np.stack([-tangents[:,1], tangents[:,0]], axis=1)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True)+1e-9)
    
    # Compute outer and inner boundaries from width offsets
    out_boundary = center + normals * df['w_tr_right_m'].to_numpy()[:,None]
    in_boundary = center - normals * df['w_tr_left_m'].to_numpy()[:,None]
    
    return center, out_boundary, in_boundary

# ===== Build trajectory =====
def build_traj(out, inn, lam):
    return (1-lam[:,None])*out + lam[:,None]*inn

# ===== Curvature =====
def curvatures(pts):
    a = pts[:-2]
    b = pts[1:-1]
    c = pts[2:]
    ab = b - a
    bc = c - b
    area2 = np.cross(ab, bc)
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(c-a, axis=1) + 1e-9
    k = np.zeros(len(pts))
    k[1:-1] = 2*np.abs(area2)/denom
    k[0], k[-1] = k[1], k[-2]
    return k

# ===== Speed profile (forward/backward) =====
def speed_profile(traj, vehicle: VehicleParams, max_iter=50, tol=0.01):
    n = len(traj)
    ds = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    ds = np.append(ds, ds[-1])  # last segment
    s = np.cumsum(np.concatenate([[0], ds[:-1]]))
    
    kappa = curvatures(traj)
    
    # Initial lateral-speed limit
    mu, g = vehicle.mu_friction, vehicle.gravity
    k_aero = vehicle.k_aero()
    v_lat = np.sqrt(np.maximum((mu * g) / (np.abs(kappa)+1e-9), 0))
    
    # Initialize speed
    v = v_lat.copy()
    
    for _ in range(max_iter):
        v_prev = v.copy()
        
        # Backward (braking)
        for i in range(n-2, -1, -1):
            a_lat = (v[i+1]**2) * np.abs(kappa[i])
            a_total = mu * (g + k_aero * v[i+1]**2)
            a_long = math.sqrt(max(a_total**2 - a_lat**2, 0.0))
            a_long = min(a_long, vehicle.a_brake_max)
            v[i] = min(v[i], math.sqrt(v[i+1]**2 + 2*a_long*ds[i]))
        
        # Forward (acceleration)
        for i in range(n-1):
            a_lat = (v[i]**2) * np.abs(kappa[i])
            a_total = mu * (g + k_aero * v[i]**2)
            a_long_friction = math.sqrt(max(a_total**2 - a_lat**2,0.0))
            
            # Drag and rolling resistance
            F_drag = vehicle.k_drag() * v[i]**2
            F_rr = vehicle.c_rr * vehicle.mass_kg * g
            a_power = max(vehicle.engine_power_watts - F_drag*v[i] - F_rr*v[i],0)/(vehicle.mass_kg*max(v[i],1e-3))
            
            a_long = min(a_long_friction, a_power, vehicle.a_accel_cap)
            v[i+1] = min(v[i+1], math.sqrt(v[i]**2 + 2*a_long*ds[i]))
        
        if np.max(np.abs(v - v_prev)) < tol:
            break
    
    return s, kappa, v_lat, v


# ===== Cost function (lap time) =====
def cost_function(ctrl, out, inn, vehicle: VehicleParams, w_smooth=0.1):
    n = len(out)
    lam = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(ctrl)), ctrl)
    lam = np.clip(lam,0,1)
    traj = build_traj(out, inn, lam)
    s, kappa, v_lat, v = speed_profile(traj, vehicle)
    T = np.sum(np.diff(s)/np.maximum(v[:-1],1e-6))
    smooth_pen = w_smooth*np.sum(np.diff(ctrl)**2)
    return T + smooth_pen

# ===== Genetic Algorithm =====
def run_ga(out, inn, vehicle, pop_size=30, gens=600, elite=2, mut_rate=0.4, mut_sigma=0.1, n_ctrl=12):
    pop = np.random.rand(pop_size, n_ctrl)
    scores = np.array([cost_function(ind, out, inn, vehicle) for ind in pop])
    
    for g in range(gens):
        new_pop = []
        elite_idx = np.argsort(scores)[:elite]
        for i in elite_idx:
            new_pop.append(pop[i].copy())
        while len(new_pop)<pop_size:
            i1,i2 = np.random.randint(0,pop_size,2)
            p1 = pop[i1] if scores[i1]<scores[i2] else pop[i2]
            i3,i4 = np.random.randint(0,pop_size,2)
            p2 = pop[i3] if scores[i3]<scores[i4] else pop[i4]
            mask = np.random.rand(n_ctrl)<0.5
            child = np.where(mask,p1,p2)
            if np.random.rand()<mut_rate:
                idxs = np.random.choice(n_ctrl,1)
                child[idxs] += mut_sigma*np.random.randn()
                child = np.clip(child,0,1)
            new_pop.append(child)
        pop = np.array(new_pop)
        scores = np.array([cost_function(ind, out, inn, vehicle) for ind in pop])
    return pop[np.argmin(scores)]

# ===== Main: full lap =====
center, out, inn  = load_track_with_boundaries('test.csv')

best_ctrl = run_ga(out, inn, vehicle, n_ctrl=12, pop_size=50, gens=400)
best_lam = np.interp(np.linspace(0,1,len(out)), np.linspace(0,1,len(best_ctrl)), best_ctrl)
best_traj = build_traj(out, inn, best_lam)

# Compute speed profile
s, kappa, v_lat, v_profile = speed_profile(best_traj, vehicle)
lap_time = np.sum(np.diff(s)/np.maximum(v_profile[:-1],1e-6))
print(f"Optimized full lap time: {lap_time:.2f} s")

# ===== Visualization =====
fig = go.Figure()
fig.add_trace(go.Scatter(x=out[:,0], y=out[:,1], mode="lines",
                         line=dict(color="black", width=15), showlegend=False))
fig.add_trace(go.Scatter(x=inn[:,0], y=inn[:,1], mode="lines",
                         line=dict(color="black", width=8), showlegend=False))
fig.add_trace(go.Scatter(x=best_traj[:,0], y=best_traj[:,1], mode="lines",
                         line=dict(color="green", width=3), name="Optimal Line"))
fig.update_layout(title=f"Full Lap GA Optimized Line ({lap_time:.2f}s)", yaxis_scaleratio=1, plot_bgcolor="white")
fig.write_html("optimized_full_lap.html", include_plotlyjs="cdn", full_html=True, auto_open=True)
