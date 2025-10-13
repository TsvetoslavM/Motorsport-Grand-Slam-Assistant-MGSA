import casadi as ca


def add_constraints(
    opti,
    vehicle,
    n,
    v,
    a_lon,
    slack_power,
    w_left,
    w_right,
    ds_array,
    curvature,
):
    """Add all problem constraints to the Opti instance.

    Returns:
        a_lat (ca.MX): Vector of lateral acceleration terms used in post-processing.
    """
    N = v.shape[0]

    # 1) Track boundaries
    boundary_margin = 0.1
    for i in range(N):
        opti.subject_to(n[i] >= -w_left[i] + boundary_margin)
        opti.subject_to(n[i] <= w_right[i] - boundary_margin)

    # 2) Velocity continuity (trapezoidal integration for dt)
    for i in range(N):
        i_next = (i + 1) % N
        v_avg = 0.5 * (v[i] + v[i_next])
        dt = ds_array[i] / (v_avg + 1e-3)
        opti.subject_to(v[i_next] == v[i] + a_lon[i] * dt)

    # 3) Velocity bounds
    for i in range(N):
        opti.subject_to(v[i] >= vehicle.v_min)
        opti.subject_to(v[i] <= vehicle.v_max)

    # 4) Lateral acceleration from path curvature
    a_lat = ca.MX.zeros(N)
    for i in range(N):
        kappa_center = curvature[i]
        denom = 1.0 - kappa_center * n[i]
        denom = ca.fmax(denom, 1e-3)
        kappa_path = kappa_center / denom
        a_lat[i] = (v[i] ** 4) * (kappa_path ** 2)

    # 5) Combined traction circle (ellipse)
    for i in range(N):
        F_normal = vehicle.mass_kg * (vehicle.gravity + vehicle.k_aero() * v[i] ** 2)
        a_max_total = vehicle.mu_friction * F_normal / vehicle.mass_kg
        a_max_total_sq = a_max_total ** 2
        opti.subject_to(a_lon[i] ** 2 + a_lat[i] <= a_max_total_sq)

    # 6) Power constraints with slack
    for i in range(N):
        F_drag = vehicle.k_drag() * v[i] ** 2
        F_rolling = vehicle.c_rr * vehicle.mass_kg * vehicle.gravity
        F_resistance = F_drag + F_rolling
        P_required = vehicle.mass_kg * a_lon[i] * v[i] + F_resistance * v[i]
        opti.subject_to(P_required <= vehicle.engine_power_watts + slack_power[i])
        opti.subject_to(P_required >= -vehicle.brake_power_watts - slack_power[i])

    # 7) Longitudinal acceleration bounds
    for i in range(N):
        opti.subject_to(a_lon[i] >= -vehicle.a_brake_max)
        opti.subject_to(a_lon[i] <= vehicle.a_accel_max)

    # 8) Path smoothness (limit change in lateral position and curvature)
    # Limit per-step lateral offset change to avoid zig-zags
    max_dn = 1.0
    for i in range(N):
        i_next = (i + 1) % N
        dn = n[i_next] - n[i]
        opti.subject_to(dn >= -max_dn)
        opti.subject_to(dn <= max_dn)

    # Additionally bound second difference to enforce smooth curvature of path
    max_d2n = 0.5
    for i in range(N):
        i_prev = (i - 1) % N
        i_next = (i + 1) % N
        d2n = n[i_next] - 2 * n[i] + n[i_prev]
        opti.subject_to(d2n >= -max_d2n)
        opti.subject_to(d2n <= max_d2n)

    # 9) Jerk limits (change in acceleration)
    max_jerk = 30.0
    for i in range(N):
        i_next = (i + 1) % N
        v_avg = 0.5 * (v[i] + v[i_next])
        dt = ds_array[i] / (v_avg + 1e-3)
        jerk = (a_lon[i_next] - a_lon[i]) / dt
        opti.subject_to(jerk >= -max_jerk)
        opti.subject_to(jerk <= max_jerk)

   # 10) Apex clipping heuristic
    # for i in range(N):
    #     if abs(float(curvature[i])) > 0.01:
    #         if float(curvature[i]) > 0:
    #             opti.subject_to(n[i] >= w_left[i] * 0.5)
    #         else:
    #             opti.subject_to(n[i] <= -w_right[i] * 0.5)
    return a_lat


