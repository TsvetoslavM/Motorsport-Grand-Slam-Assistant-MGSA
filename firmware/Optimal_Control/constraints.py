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

    # 10) Apex constraints at curvature peaks (inner-side clipping only at apex)
    # Detect local maxima of |curvature| above a threshold and constrain n there
    try:
        abs_curv = [abs(float(curvature[i])) for i in range(N)]
        curv_thresh = 0.01
        min_peak_separation = 5  # in discretization steps, to avoid clustered peaks
        candidate_indices = []
        for i in range(N):
            i_prev = (i - 1) % N
            i_next = (i + 1) % N
            if abs_curv[i] > curv_thresh and abs_curv[i] >= abs_curv[i_prev] and abs_curv[i] >= abs_curv[i_next]:
                candidate_indices.append(i)

        # Thin peaks to ensure separation
        apex_indices = []
        for idx in candidate_indices:
            if not apex_indices:
                apex_indices.append(idx)
                continue
            if all(min((idx - j) % N, (j - idx) % N) >= min_peak_separation for j in apex_indices):
                apex_indices.append(idx)

        # Apply inner-side constraint at apex only
        apex_inside_fraction = 0.6  # n must be at least 60% toward inside at apex
        for i in apex_indices:
            k = float(curvature[i])
            if k > 0:
                # Left-hand corner: positive curvature -> inside is to the left
                opti.subject_to(n[i] >= apex_inside_fraction * w_left[i])
            elif k < 0:
                # Right-hand corner: negative curvature -> inside is to the right (negative n)
                opti.subject_to(n[i] <= -apex_inside_fraction * w_right[i])
            # if k == 0, skip
    except Exception:
        # If curvature is symbolic or detection fails, skip apex constraints gracefully
        pass

    # 11) Delay outside positioning in long corners (entry window constraint)
    try:
        abs_curv = [abs(float(curvature[i])) for i in range(N)]
        curv_thresh = 0.01
        is_corner = [1 if c > curv_thresh else 0 for c in abs_curv]

        # Find contiguous segments of corner samples
        segments = []
        start = None
        for i in range(N):
            if is_corner[i] and start is None:
                start = i
            elif not is_corner[i] and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, N - 1))

        # Apply entry constraints for sufficiently long corners
        min_long_len = 20
        entry_fraction = 0.25
        outside_fraction_require = 0.35  # required fraction toward outside during entry window

        for (s0, s1) in segments:
            seg_indices = list(range(s0, s1 + 1))
            seg_len = len(seg_indices)
            if seg_len < min_long_len:
                continue

            # Determine dominant curvature sign within the segment
            seg_sign_sum = sum(float(curvature[i]) for i in seg_indices)
            if seg_sign_sum == 0:
                continue
            sign_k = 1.0 if seg_sign_sum > 0 else -1.0

            entry_len = max(3, int(entry_fraction * seg_len))
            entry_len = min(entry_len, 30)
            entry_indices = seg_indices[:entry_len]

            for i in entry_indices:
                if sign_k > 0:
                    # Left-hand corner: outside is to the right (negative n)
                    opti.subject_to(n[i] <= -outside_fraction_require * w_right[i])
                else:
                    # Right-hand corner: outside is to the left (positive n)
                    opti.subject_to(n[i] >= outside_fraction_require * w_left[i])
    except Exception:
        # If detection fails (e.g., symbolic curvature), skip gracefully
        pass

    # 12) Force wide exit if a long straight follows the corner (exit window)
    try:
        abs_curv = [abs(float(curvature[i])) for i in range(N)]
        curv_thresh = 0.01
        is_corner = [1 if c > curv_thresh else 0 for c in abs_curv]

        # contiguous corner segments
        segments = []
        start = None
        for i in range(N):
            if is_corner[i] and start is None:
                start = i
            elif not is_corner[i] and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, N - 1))

        # Detect long straight following a segment using curvature and ds
        # We approximate straight length by counting samples with very low curvature
        straight_thresh = 0.002
        min_straight_len = 20  # samples of very low curvature after corner end
        exit_fraction = 0.35
        outside_fraction_require = 0.95

        for (s0, s1) in segments:
            # determine sign of corner
            seg_indices = list(range(s0, s1 + 1))
            seg_sign_sum = sum(float(curvature[i]) for i in seg_indices)
            if seg_sign_sum == 0:
                continue
            sign_k = 1.0 if seg_sign_sum > 0 else -1.0

            # scan ahead from s1 to see if a long straight follows
            count_straight = 0
            j = (s1 + 1) % N
            while count_straight < min_straight_len:
                if abs(float(curvature[j])) < straight_thresh:
                    count_straight += 1
                    j = (j + 1) % N
                    if j == s0:
                        break
                else:
                    break

            if count_straight >= min_straight_len:
                # enforce wide exit over the last portion of the corner
                seg_len = len(seg_indices)
                exit_len = max(3, int(exit_fraction * seg_len))
                exit_len = min(exit_len, 30)
                exit_indices = seg_indices[-exit_len:]

                for i in exit_indices:
                    if sign_k > 0:
                        # Left-hand corner -> outside is right (negative n)
                        opti.subject_to(n[i] <= -outside_fraction_require * w_right[i])
                    else:
                        # Right-hand corner -> outside is left (positive n)
                        opti.subject_to(n[i] >= outside_fraction_require * w_left[i])
    except Exception:
        pass
    return a_lat


