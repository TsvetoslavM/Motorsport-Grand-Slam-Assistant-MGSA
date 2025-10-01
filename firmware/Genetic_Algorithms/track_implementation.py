import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
def cost_function(ctrl, out, inn, n=None, mu=1.7, g=9.81, vmax=80.0, w_smooth=0.1, a_max=8.0):
    """
    More accurate lap time computation using friction circle:
    - ctrl: control points (0=outer, 1=inner)
    - out, inn: track boundaries
    - mu: tire friction coefficient
    - g: gravity
    - vmax: max speed on straights
    - w_smooth: smoothness penalty
    - a_max: maximum longitudinal acceleration/braking (m/s^2)
    """
    if n is None:
        n = len(out)
    
    # Interpolate lambda to full trajectory
    lam = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(ctrl)), ctrl)
    lam = np.clip(lam,0,1)
    traj = build_traj(out, inn, lam)

    # Segment lengths
    ds = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    # Curvature
    k = curvatures(traj)

    # Maximum lateral speed (v_lat) from mu*g = v^2 / R
    v_lat_max = np.sqrt(mu * g / (np.abs(k) + 1e-6))
    v_lat_max = np.minimum(v_lat_max, vmax)

    # Initialize speed profile
    v = np.zeros_like(v_lat_max)
    v[0] = min(v_lat_max[0], vmax)

    # Forward pass: accelerate respecting friction circle
    for i in range(1, len(v)):
        # Max longitudinal acceleration limited by friction circle
        if k[i-1] < 1e-6:  # nearly straight
            a_long_max = a_max
        else:
            a_long_max = np.sqrt(max(a_max**2 - (v[i-1]**2 * k[i-1])**2, 0))
        dv = np.sqrt(v[i-1]**2 + 2 * a_long_max * ds[i-1])
        v[i] = min(dv, v_lat_max[i])

    # Backward pass: braking respecting friction circle
    for i in range(len(v)-2, -1, -1):
        if k[i] < 1e-6:
            a_long_max = a_max
        else:
            a_long_max = np.sqrt(max(a_max**2 - (v[i+1]**2 * k[i])**2, 0))
        dv = np.sqrt(v[i+1]**2 + 2 * a_long_max * ds[i])
        v[i] = min(v[i], dv, v_lat_max[i])

    # Compute total lap time
    v_seg = (v[:-1] + v[1:]) / 2
    T = np.sum(ds / v_seg)

    # Smoothness penalty for GA
    smooth_pen = w_smooth*np.sum((np.diff(ctrl))**2)

    return T + smooth_pen



# ==== 6. GA ==== 
def run_ga(out, inn, pop_size=30, gens=5000, elite=2, mut_rate=0.4, mut_sigma=0.1, n_ctrl=6):
    pop = np.random.rand(pop_size, n_ctrl)
    scores = np.array([cost_function(ind, out, inn) for ind in pop])

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
        scores = np.array([cost_function(ind, out, inn) for ind in pop])

    return pop[np.argmin(scores)]

# ==== 7. Run GA ==== 
center = load_center_from_csv('test.csv')
out, inn = generate_boundaries(center)

best_ctrl = run_ga(out, inn, n_ctrl=6, pop_size=30, gens=600)
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
