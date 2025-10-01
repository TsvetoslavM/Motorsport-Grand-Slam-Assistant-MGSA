import numpy as np
import pandas as pd
import plotly.graph_objects as go

class VehicleParams:
    mass_kg: float = 798.0
    mu_friction: float = 2.0
    gravity: float = 9.81
    rho_air: float = 1.225
    cL_downforce: float = 4.0
    frontal_area_m2: float = 1.6
    engine_power_watts: float = 735000.0
    a_brake_max: float = 54.0  # m/s^2, hard cap for braking decel (system limit)
    a_accel_cap: float = 20.0  # m/s^2, optional hard cap for longitudinal accel
    cD_drag: float = 1.0       # aerodynamic drag coefficient
    c_rr: float = 0.004        # rolling resistance coefficient (F1-like)
    safety_speed_margin: float = 1.00  # multiplier for global power cap (≤ 1.0)
    brake_power_watts: float = 1200000.0  # optional power dissipation limit for braking

    def k_aero(self) -> float:
        # k_aero = (0.5 * rho * C_L * A) / m
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / max(self.mass_kg, 1e-9)

    def k_drag(self) -> float:
        # k_drag = 0.5 * rho * C_D * A
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2

# ==== 1. Load centerline from CSV ==== 
def load_center_from_csv(filename):
    df = pd.read_csv(filename)
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV file must contain 'x' and 'y' columns")
    return df[['x','y']].to_numpy()

# ==== 2. Generate track boundaries ==== 
def generate_boundaries(center, width=8.0):
    tangents = np.gradient(center, axis=0)
    normals = np.stack([-tangents[:,1], tangents[:,0]], axis=1)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True)+1e-9)
    
    out = center + normals*(width/2.0)
    inn = center - normals*(width/2.0)
    return out, inn

# ==== 3. Build trajectory from lambda ==== 
def build_traj(out, inn, lam):
    return (1-lam[:,None])*out + lam[:,None]*inn

# ==== 4. Curvature ==== 
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

# ==== 5. Cost function ==== 
def cost_function(ctrl, out, inn, vehicle: VehicleParams, n=None, w_smooth=0.1):
    """
    Compute lap time using realistic car physics.
    
    ctrl: GA control points (0=outer, 1=inner)
    out, inn: track boundaries
    vehicle: instance of VehicleParams
    n: number of points along trajectory
    w_smooth: penalty for non-smooth control points
    """
    if n is None:
        n = len(out)
    
    # Interpolate lambda
    lam = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(ctrl)), ctrl)
    lam = np.clip(lam, 0, 1)
    traj = build_traj(out, inn, lam)

    # Segment lengths
    ds = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    k = curvatures(traj)  # curvature
    
    # Max lateral speed from tire friction + downforce
    # a_lat = mu * g + k_aero * v^2 → v_max = sqrt(mu*g / k)
    k_aero = vehicle.k_aero()
    v_lat_max = np.sqrt((vehicle.mu_friction * vehicle.gravity) / (np.abs(k)+1e-9))
    
    # Initialize speed profile
    v = np.zeros_like(v_lat_max)
    v[0] = min(v_lat_max[0], np.sqrt(vehicle.engine_power_watts / max(vehicle.mass_kg,1e-9)))

    # Forward pass (acceleration limited by engine, aero, drag)
    for i in range(1, len(v)):
        # longitudinal limit due to friction circle
        if k[i-1] < 1e-6:
            a_long_max = vehicle.a_accel_cap
        else:
            a_long_max = np.sqrt(max(vehicle.a_accel_cap**2 - (v[i-1]**2 * k[i-1])**2, 0))
        
        # aerodynamic downforce + drag
        f_aero = k_aero * v[i-1]**2  # extra lateral grip from downforce
        f_drag = vehicle.k_drag() * v[i-1]**2
        f_rr = vehicle.c_rr * vehicle.mass_kg * vehicle.gravity
        
        # net longitudinal accel possible
        a_net = min(a_long_max, vehicle.engine_power_watts / max(vehicle.mass_kg*v[i-1], 1e-9))
        a_net = max(a_net - f_drag/vehicle.mass_kg - f_rr/vehicle.mass_kg, 0)
        
        dv = np.sqrt(v[i-1]**2 + 2 * a_net * ds[i-1])
        v[i] = min(dv, v_lat_max[i])

    # Backward pass (braking)
    for i in range(len(v)-2, -1, -1):
        if k[i] < 1e-6:
            a_long_max = vehicle.a_brake_max
        else:
            a_long_max = np.sqrt(max(vehicle.a_brake_max**2 - (v[i+1]**2 * k[i])**2, 0))
        
        dv = np.sqrt(v[i+1]**2 + 2 * a_long_max * ds[i])
        v[i] = min(v[i], dv, v_lat_max[i])

    # Segment average speed
    v_seg = (v[:-1] + v[1:]) / 2
    T = np.sum(ds / np.maximum(v_seg, 1e-6))  # total lap time

    # Smoothness penalty
    smooth_pen = w_smooth * np.sum((np.diff(ctrl))**2)

    return T + smooth_pen



# ==== 6. GA ==== 
def run_ga(out, inn, vehicle: VehicleParams, pop_size=30, gens=600, elite=2, mut_rate=0.4, mut_sigma=0.1, n_ctrl=6):
    pop = np.random.rand(pop_size, n_ctrl)
    scores = np.array([cost_function(ind, out, inn, vehicle) for ind in pop])

    for g in range(gens):
        new_pop = []
        elite_idx = np.argsort(scores)[:elite]
        for i in elite_idx:
            new_pop.append(pop[i].copy())

        while len(new_pop) < pop_size:
            i1, i2 = np.random.randint(0, pop_size, 2)
            p1 = pop[i1] if scores[i1] < scores[i2] else pop[i2]
            i3, i4 = np.random.randint(0, pop_size, 2)
            p2 = pop[i3] if scores[i3] < scores[i4] else pop[i4]
            mask = np.random.rand(n_ctrl) < 0.5
            child = np.where(mask, p1, p2)
            if np.random.rand() < mut_rate:
                idxs = np.random.choice(n_ctrl, 1)
                child[idxs] += mut_sigma*np.random.randn()
                child = np.clip(child,0,1)
            new_pop.append(child)

        pop = np.array(new_pop)
        scores = np.array([cost_function(ind, out, inn, vehicle) for ind in pop])

    return pop[np.argmin(scores)]


# ==== 7. Run GA ==== 
center = load_center_from_csv('test.csv')
out, inn = generate_boundaries(center)

vehicle = VehicleParams()

# ==== Run GA with vehicle passed in ====
best_ctrl = run_ga(out, inn, pop_size=30, gens=600, n_ctrl=6, vehicle=vehicle)

best_lam = np.interp(np.linspace(0,1,len(out)), np.linspace(0,1,len(best_ctrl)), best_ctrl)
best_traj = build_traj(out, inn, best_lam)

# ==== 8. Plotly visualization with markers ==== 
fig = go.Figure()

# Outer boundary thick black line
fig.add_trace(go.Scatter(x=out[:,0], y=out[:,1], mode="lines",
                         line=dict(color="black", width=15),
                         hoverinfo="skip", showlegend=False))
# Inner boundary white line
fig.add_trace(go.Scatter(x=inn[:,0], y=inn[:,1], mode="lines",
                         line=dict(color="black", width=8),
                         hoverinfo="skip", showlegend=False))
# Track points with black border
# Optimized racing line
fig.add_trace(go.Scatter(x=best_traj[:,0], y=best_traj[:,1], mode="lines",
                         line=dict(color="green", width=3),
                         name="Optimal Line"))

fig.update_layout(title="Минимално време през завоя (GA Optimized)",
                  yaxis_scaleratio=1, plot_bgcolor="white")

fig.write_html("optimized_track.html", include_plotlyjs="cdn", full_html=True, auto_open=True)
