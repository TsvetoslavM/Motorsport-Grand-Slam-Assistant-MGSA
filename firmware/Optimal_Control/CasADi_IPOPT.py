import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from firmware.vehicle import VehicleParams
from .constraints import (
    add_constraints,
    create_objective_with_racing_line,
    initialize_with_proper_racing_line,
)

try:
    from .visualization import plot_f1_results, print_summary
except Exception:
    print("Warning: visualization module not found, skipping plots")

# ═══════════════════════════════════════════════════════════════
# 🔧 SHAREABLE FUNCTIONS FOR OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def compute_curvature_closed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute curvature for a closed loop using three-point method.
    
    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        
    Returns:
        Array of curvature values
    """
    N = len(x)
    curvature = np.zeros(N, dtype=float)
    for i in range(N):
        i_prev = (i - 1) % N
        i_next = (i + 1) % N
        x0, y0 = x[i_prev], y[i_prev]
        x1, y1 = x[i], y[i]
        x2, y2 = x[i_next], y[i_next]
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1
        cross = dx1 * dy2 - dy1 * dx2
        ds1 = np.sqrt(dx1**2 + dy1**2) + 1e-9
        ds2 = np.sqrt(dx2**2 + dy2**2) + 1e-9
        curvature[i] = 2 * cross / (ds1 * ds2 * (ds1 + ds2) + 1e-9)
    return curvature


def smooth_curvature_closed(curvature: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Smooth curvature using Gaussian filter with wrap-around for closed loops.
    
    Args:
        curvature: Array of curvature values
        sigma: Gaussian filter standard deviation
        
    Returns:
        Smoothed curvature array
    """
    return gaussian_filter1d(curvature, sigma=sigma, mode='wrap')


def adaptive_path_discretization(
    x: np.ndarray,
    y: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    curvature_threshold: float = 0.01,
    ds_straight: float = 5.0,
    ds_corner: float = 1.5,
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptively resample a closed track path based on curvature.
    
    Uses finer discretization in corners and coarser in straights.
    
    Args:
        x: Array of x coordinates (closed loop)
        y: Array of y coordinates (closed loop)
        w_left: Array of left widths
        w_right: Array of right widths
        curvature_threshold: Curvature threshold to distinguish corners from straights
        ds_straight: Target segment length for straights (meters)
        ds_corner: Target segment length for corners (meters)
        smooth_sigma: Gaussian smoothing sigma for curvature calculation
        
    Returns:
        Tuple of (x_resampled, y_resampled, w_left_resampled, w_right_resampled, ds_array)
    """
    N_raw = len(x)
    if N_raw < 3:
        # Too few points, return as-is
        ds_array = np.zeros(N_raw)
        for i in range(N_raw):
            i_next = (i + 1) % N_raw
            dx = x[i_next] - x[i]
            dy = y[i_next] - y[i]
            ds_array[i] = np.sqrt(dx**2 + dy**2)
        return x, y, w_left, w_right, ds_array
    
    # Calculate initial curvature on raw track
    curvature_raw = compute_curvature_closed(x, y)
    curvature_raw = smooth_curvature_closed(curvature_raw, sigma=smooth_sigma)
    
    # Adaptive resampling
    new_x, new_y, new_w_right, new_w_left = [], [], [], []
    i = 0
    while i < N_raw:
        new_x.append(x[i])
        new_y.append(y[i])
        new_w_right.append(w_right[i])
        new_w_left.append(w_left[i])
        
        abs_curv = abs(curvature_raw[i])
        if abs_curv > curvature_threshold:
            ds_target = ds_corner
        else:
            ds_target = ds_straight
        
        i_next = (i + 1) % N_raw
        dx = x[i_next] - x[i]
        dy = y[i_next] - y[i]
        ds_actual = np.sqrt(dx**2 + dy**2)
        
        skip = max(1, int(ds_target / (ds_actual + 1e-6)))
        i += skip
        
        if i >= N_raw and len(new_x) > 10:
            break
    
    x_resampled = np.array(new_x)
    y_resampled = np.array(new_y)
    w_right_resampled = np.array(new_w_right)
    w_left_resampled = np.array(new_w_left)
    N = len(x_resampled)
    
    # Recalculate segment lengths
    ds_array = np.zeros(N)
    for i in range(N):
        i_next = (i + 1) % N
        dx = x_resampled[i_next] - x_resampled[i]
        dy = y_resampled[i_next] - y_resampled[i]
        ds_array[i] = np.sqrt(dx**2 + dy**2)
    
    return x_resampled, y_resampled, w_left_resampled, w_right_resampled, ds_array


def get_advanced_ipopt_options(
    max_iter: int = 3000,
    tol: float = 1e-4,
    acceptable_tol: float = 1e-3,
    print_level: int = 5,
    linear_solver: str = "mumps",
    **overrides
) -> dict:
    """
    Get advanced IPOPT solver options with best practices for trajectory optimization.
    
    Args:
        max_iter: Maximum number of iterations
        tol: Optimality tolerance
        acceptable_tol: Acceptable tolerance for early termination
        print_level: IPOPT print level (0=silent, 5=verbose)
        linear_solver: Linear solver to use ('mumps', 'ma27', 'ma57', etc.)
        **overrides: Additional options to override defaults
        
    Returns:
        Dictionary of IPOPT options
    """
    opts = {
        'ipopt.max_iter': max_iter,
        'ipopt.tol': tol,
        'ipopt.acceptable_tol': acceptable_tol,
        'ipopt.acceptable_iter': 20,
        'ipopt.constr_viol_tol': 1e-3,
        'ipopt.print_level': print_level,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-6,
        'ipopt.warm_start_mult_bound_push': 1e-6,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.adaptive_mu_globalization': 'kkt-error',
        'ipopt.linear_solver': linear_solver,
        'ipopt.hessian_approximation': 'exact',
        'ipopt.limited_memory_max_history': 50,
        'ipopt.nlp_scaling_method': 'gradient-based',
        'ipopt.line_search_method': 'filter',
        'ipopt.alpha_for_y': 'primal',
        'error_on_fail': False
    }
    
    # Apply overrides
    opts.update(overrides)
    
    return opts


# ═══════════════════════════════════════════════════════════════
# 📜 MAIN SCRIPT (only runs when executed directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    vehicle = VehicleParams()

    # ═══════════════════════════════════════════════════════════════
    # 📊 LOAD TRACK DATA
    # ═══════════════════════════════════════════════════════════════
    try:
        track = pd.read_csv("Track.csv", comment="#", names=["x_m","y_m","w_tr_right_m","w_tr_left_m"])
    except:
        print("Warning: Could not find Track.csv, using synthetic track")
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

    # Use shareable adaptive discretization function
    x_center, y_center, w_left, w_right, ds_array = adaptive_path_discretization(
        x_center, y_center, w_left, w_right,
        curvature_threshold=0.01,
        ds_straight=5.0,
        ds_corner=1.5,
        smooth_sigma=2.0
    )
    N = len(x_center)

    print(f"Original segments: {N_raw}")
    print(f"Adaptive segments: {N}")
    print(f"Reduction: {100*(1-N/N_raw):.1f}%")

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

    # Use shareable curvature computation and smoothing functions
    curvature = compute_curvature_closed(x_center, y_center)
    curvature = smooth_curvature_closed(curvature, sigma=2.0)

    # ═══════════════════════════════════════════════════════════════
    # 🎯 OPTIMIZATION SETUP
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🎯 OPTIMIZATION SETUP WITH RACING LINE")
    print("="*60)

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
    # 🎬 INITIALIZATION WITH RACING LINE
    # ═══════════════════════════════════════════════════════════════
    print("\nInitializing with proper racing line geometry...")
    corner_types, corner_phases = initialize_with_proper_racing_line(
        opti, n, v, a_lon, curvature, ds_array, w_left, w_right, vehicle, N
    )

    print(f"\nCorner classification:")
    for corner_type, indices in corner_types.items():
        if len(indices) > 0:
            print(f"  {corner_type}: {len(indices)} segments")

    # ═══════════════════════════════════════════════════════════════
    # 🚧 CONSTRAINTS WITH RACING LINE AWARENESS
    # ═══════════════════════════════════════════════════════════════
    print("\nAdding constraints with racing line awareness...")
    vehicle.mu_friction = 2.0
    vehicle.cL_downforce = 3.0
    vehicle.a_accel_max = 12.0
    vehicle.a_brake_max = 45.0

    a_lat, corner_types, corner_phases = add_constraints(
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
    # 🔹 OBJECTIVE FUNCTION WITH RACING LINE
    # ═══════════════════════════════════════════════════════════════
    print("\nCreating objective function with racing line cost...")
    total_cost = create_objective_with_racing_line(
        opti, v, a_lon, slack_power, n, curvature, w_left, w_right,
        corner_phases, ds_array, N, vehicle
    )

    # ═══════════════════════════════════════════════════════════════
    # ⚙️ SOLVER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    opts = get_advanced_ipopt_options(
        max_iter=3000,
        tol=1e-4,
        acceptable_tol=1e-3,
        print_level=5,
        linear_solver='mumps'
    )

    opti.solver('ipopt', opts)

    # ═══════════════════════════════════════════════════════════════
    # 🚀 SOLVE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🚀 Starting F1 lap time optimization...")
    print("="*60)

    try:
        sol = opti.solve()
        print("\n✅ Optimization converged successfully!")
    except Exception as e:
        print(f"\n⚠️ Solver didn't fully converge: {e}")
        print("Using best solution found...")
        sol = opti.debug

    # ═══════════════════════════════════════════════════════════════
    # 📊 EXTRACT RESULTS
    # ═══════════════════════════════════════════════════════════════
    n_opt = sol.value(n)
    v_opt = sol.value(v)
    a_lon_opt = sol.value(a_lon)
    a_lat_opt = sol.value(a_lat)

    # Calculate lap time
    lap_time = 0
    for i in range(N):
        i_next = (i + 1) % N
        v_avg = 0.5 * (v_opt[i] + v_opt[i_next])
        dt = ds_array[i] / (v_avg + 1e-3)
        lap_time += dt

    lap_time_seconds = float(lap_time)
    track_length = np.sum(ds_array)

    # ═══════════════════════════════════════════════════════════════
    # 💾 SAVE OPTIMAL TRAJECTORY TO CSV
    # ═══════════════════════════════════════════════════════════════
    x_opt = x_center + n_opt * normals[:, 0]
    y_opt = y_center + n_opt * normals[:, 1]

    eps = 1e-6
    time_s = np.cumsum(np.concatenate(([0.0], ds_array[:-1] / np.maximum(v_opt[:-1], eps))))
    speed_kmh = v_opt * 3.6

    opt_df = pd.DataFrame({
        "x_m": x_opt,
        "y_m": y_opt,
        "speed_kmh": speed_kmh,
        "time_s": time_s,
    })

    opt_df.to_csv("optiline.csv", index=False)
    print("\n✅ Optimal trajectory saved as 'optiline.csv'!")

    # ═══════════════════════════════════════════════════════════════
    # 📋 PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📋 OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Lap time: {lap_time_seconds:.3f} seconds")
    print(f"Track length: {track_length:.1f} meters")
    print(f"Average speed: {track_length/lap_time_seconds*3.6:.1f} km/h")
    print(f"Max speed: {np.max(v_opt)*3.6:.1f} km/h")
    print(f"Min speed: {np.min(v_opt)*3.6:.1f} km/h")
    print(f"Max longitudinal accel: {np.max(a_lon_opt):.2f} m/s²")
    print(f"Max braking: {np.min(a_lon_opt):.2f} m/s²")
    print(f"Max lateral accel: {np.max(np.abs(a_lat_opt)):.2f} m/s²")

    try:
        print_summary(
            v_opt=v_opt,
            a_lon_opt=a_lon_opt,
            a_lat_opt=a_lat_opt,
            vehicle=vehicle,
            lap_time_seconds=lap_time_seconds,
            track_length=track_length
        )
    except:
        pass

    # ═══════════════════════════════════════════════════════════════
    # 📈 VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    try:
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
    except Exception as e:
        print(f"\nVisualization error: {e}")
        print("Continuing without plots...")

    print("\n" + "="*60)
    print("🏁 OPTIMIZATION COMPLETE!")
    print("="*60)