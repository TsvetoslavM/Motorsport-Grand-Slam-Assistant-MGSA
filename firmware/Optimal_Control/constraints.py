import casadi as ca
import numpy as np


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
        a_lat[i] = v[i] ** 2 * kappa_path

    # 5) Combined traction circle (ellipse)
    for i in range(N):
        F_normal = vehicle.mass_kg * (vehicle.gravity + vehicle.k_aero() * v[i] ** 2)
        a_max_total = vehicle.mu_friction * F_normal / vehicle.mass_kg
        a_max_total_sq = a_max_total ** 2
        opti.subject_to(a_lon[i]**2 + a_lat[i]**2 <= a_max_total_sq)

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

    # ============================================================================
    # Detect corners and apply different strategies based on corner angle
    # ============================================================================
    try:
        # Convert curvature to numeric for analysis
        abs_curv = [abs(float(curvature[i])) for i in range(N)]
        curv_signed = [float(curvature[i]) for i in range(N)]
        
        # Detect corner segments
        curv_thresh = 0.01
        is_corner = [1 if c > curv_thresh else 0 for c in abs_curv]
        
        # Find contiguous corner segments
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
        
        min_corner_length = 5
        
        # ========================================================================
        # CHICANE DETECTION: Detect S-curves (rapid direction reversals)
        # ========================================================================
        chicanes = []
        for i in range(len(segments)):
            if i + 1 < len(segments):
                seg1_start, seg1_end = segments[i]
                seg2_start, seg2_end = segments[i + 1]
                
                # Check if segments are close together
                gap = (seg2_start - seg1_end - 1) % N
                if gap < 15:  # Less than 15 points between corners
                    # Check if they have opposite curvature signs
                    seg1_curv_avg = sum(curv_signed[j] for j in range(seg1_start, seg1_end + 1)) / max(1, seg1_end - seg1_start + 1)
                    seg2_curv_avg = sum(curv_signed[j] for j in range(seg2_start, seg2_end + 1)) / max(1, seg2_end - seg2_start + 1)
                    
                    if seg1_curv_avg * seg2_curv_avg < 0:  # Opposite signs
                        chicanes.append((i, i + 1))  # Store segment indices that form chicane
        
        for (s0, s1) in segments:
            seg_indices = list(range(s0, s1 + 1))
            seg_len = len(seg_indices)
            
            if seg_len < min_corner_length:
                continue
            
            # Check if this segment is part of a chicane
            is_chicane = False
            chicane_idx = None
            for chi_idx, (chi_seg1, chi_seg2) in enumerate(chicanes):
                seg_idx = segments.index((s0, s1))
                if seg_idx == chi_seg1 or seg_idx == chi_seg2:
                    is_chicane = True
                    chicane_idx = chi_idx
                    break
            
            # Calculate total corner angle by integrating curvature over distance
            total_angle = 0.0
            for i in seg_indices:
                total_angle += abs(float(curvature[i])) * float(ds_array[i])
            
            # Convert to degrees
            total_angle_deg = abs(total_angle) * 180.0 / 3.14159265359
            
            # Determine corner direction
            seg_curv_sum = sum(float(curvature[i]) for i in seg_indices)
            if seg_curv_sum == 0:
                continue
            
            is_left_turn = seg_curv_sum > 0
            
            # ========================================================================
            # CHICANE HANDLING: Special strategy for S-curves
            # ========================================================================
            # CHICANE HANDLING: Smooth S-turns
            if is_chicane:
                chi_seg1, chi_seg2 = chicanes[chicane_idx]
                seg_idx = segments.index((s0, s1))

                # Get both corner segments and indices
                seg1_start, seg1_end = segments[chi_seg1]
                seg2_start, seg2_end = segments[chi_seg2]
                seg1_indices = list(range(seg1_start, seg1_end + 1))
                seg2_indices = list(range(seg2_start, seg2_end + 1))

                # Apex detection (index in global track indexing)
                apex_idx_1 = max(seg1_indices, key=lambda i: abs_curv[i])
                apex_idx_2 = max(seg2_indices, key=lambda i: abs_curv[i])

                apex_pos1 = seg1_indices.index(apex_idx_1)
                apex_pos2 = seg2_indices.index(apex_idx_2)

                # Average curvature sign -> corner direction
                seg1_curv_avg = sum(curv_signed[i] for i in seg1_indices) / len(seg1_indices)
                seg2_curv_avg = sum(curv_signed[i] for i in seg2_indices) / len(seg2_indices)
                is_first_left = seg1_curv_avg > 0
                is_second_left = seg2_curv_avg > 0

                # Define numeric lateral targets (tweak these scalars if needed)
                outside_mag = 0.90   # how far to start from the outside
                apex_inside_mag = 0.60  # how deep to the inside at apex
                exit_wide_mag = 0.95  # how wide on exit of second corner

                # Determine signed target numbers: positive => left-side targets, negative => right-side targets
                def signed_outside(is_left):
                    return -outside_mag if is_left else outside_mag

                def signed_apex_inside(is_left):
                    return apex_inside_mag if is_left else -apex_inside_mag

                def signed_exit_outside(is_left):
                    return exit_wide_mag if not is_left else -exit_wide_mag  # exit is opposite sign to inside

                # ---------- FIRST SEGMENT (entry -> apex -> exit toward straight)
                for idx, i in enumerate(seg1_indices):
                    # progress 0..1 relative to this segment
                    progress = idx / max(1, len(seg1_indices) - 1)

                    if idx <= apex_pos1:
                        # ENTRY: move from outside to inside at apex
                        entry_progress = idx / max(1, apex_pos1)
                        start_t = signed_outside(is_first_left)       # outside starting position
                        end_t = signed_apex_inside(is_first_left)     # inside at apex
                        target = start_t + (end_t - start_t) * entry_progress
                    else:
                        # EXIT from apex -> aim toward the straightline connection (intermediate diagonal)
                        exit_progress = (idx - apex_pos1) / max(1, len(seg1_indices) - 1 - apex_pos1)
                        # we aim to move from inside (apex) back toward a neutral/diagonal approach value (~0)
                        # but will compute exact connection later for transition indices; here bias toward center
                        target = signed_apex_inside(is_first_left) + (0.0 - signed_apex_inside(is_first_left)) * exit_progress * 0.9

                    # apply using same mapping as your code (positive -> w_left lower bound, negative -> w_right upper bound)
                    if target > 0:
                        opti.subject_to(n[i] >= target * w_left[i])
                    else:
                        opti.subject_to(n[i] <= target * w_right[i])

                # ---------- SECOND SEGMENT (approach -> apex -> track out)
                for idx, i in enumerate(seg2_indices):
                    progress = idx / max(1, len(seg2_indices) - 1)
                    # For second corner we want to approach apex along the straightline; use a softer entry toward apex
                    if idx <= apex_pos2:
                        # entry from the straightline toward apex
                        entry_progress = idx / max(1, apex_pos2)
                        # entry start will be filled by the straightline interpolation (computed lower)
                        # default start value before interpolation: signed_outside(second) (safe fallback)
                        start_t = signed_outside(is_second_left)
                        end_t = signed_apex_inside(is_second_left)
                        target = start_t + (end_t - start_t) * (0.2 + 0.8 * entry_progress)  # bias slightly inside early
                    else:
                        # exit: from apex to full outside for good exit
                        exit_progress = (idx - apex_pos2) / max(1, len(seg2_indices) - 1 - apex_pos2)
                        start_t = signed_apex_inside(is_second_left)
                        end_t = signed_exit_outside(is_second_left)
                        target = start_t + (end_t - start_t) * (exit_progress ** 1.15)  # slightly nonlinear opening

                    if target > 0:
                        opti.subject_to(n[i] >= target * w_left[i])
                    else:
                        opti.subject_to(n[i] <= target * w_right[i])

                # ---------- STRAIGHT LINE CONNECTION BETWEEN APEX1 AND APEX2 (REVISED)
                # Force a nearly geometric straight line across transition
                transition_start = (seg1_end + 1) % N
                transition_end = (seg2_start - 1) % N

                inter_indices = []
                j = transition_start
                while True:
                    if j == seg2_start or j == (seg2_start % N):
                        break
                    inter_indices.append(j)
                    j = (j + 1) % N
                    if len(inter_indices) > N:
                        break

                if len(inter_indices) > 0:
                    # compute pure linear interpolation of lateral offset between apex1 and apex2
                    # regardless of curvature in that zone
                    start_val = signed_apex_inside(is_first_left) * 0.35
                    end_val   = signed_apex_inside(is_second_left) * 0.35

                    for tpos, idx_global in enumerate(inter_indices):
                        frac = tpos / max(1, len(inter_indices) - 1)

                        # strict linear interpolation (straight path)
                        interp_target = start_val + (end_val - start_val) * frac

                        # ignore curvature (do NOT damp by curvature anymore)
                        # small smoothing near edges to avoid jumps
                        if frac < 0.15:
                            interp_target = start_val + (interp_target - start_val) * (frac / 0.15)
                        elif frac > 0.85:
                            interp_target = interp_target + (end_val - interp_target) * ((frac - 0.85) / 0.15)

                        if interp_target > 0:
                            opti.subject_to(n[idx_global] >= interp_target * w_left[idx_global])
                        else:
                            opti.subject_to(n[idx_global] <= interp_target * w_right[idx_global])


                # ---------- SPEED MANAGEMENT BETWEEN APEXES (lenient to favor connecting line)
                # Allow slightly higher corner speeds for shallow S-turns to favor momentum
                max_corner_curv = max(abs_curv[i] for i in seg1_indices + seg2_indices)
                if max_corner_curv > 0.012:
                    for i in (seg1_indices + seg2_indices):
                        k = abs(float(curvature[i]))
                        if k > 0.005:
                            F_normal_est = vehicle.mass_kg * vehicle.gravity
                            a_lat_max = 0.95 * vehicle.mu_friction * F_normal_est / vehicle.mass_kg
                            R = 1.0 / (k + 1e-6)
                            v_corner_max = ca.sqrt(a_lat_max * R)
                            opti.subject_to(v[i] <= v_corner_max * 1.25)


            
            # ========================================================================
            # If corner is MORE than 90 degrees: Use progressive racing line
            # ========================================================================
            elif total_angle_deg > 90.0:
                # Find apex (maximum curvature point)
                apex_idx = max(seg_indices, key=lambda i: abs_curv[i])
                apex_position = seg_indices.index(apex_idx)
                
                # Progressive racing line through the corner
                for idx, i in enumerate(seg_indices):
                    progress = idx / max(1, seg_len - 1)
                    
                    if idx <= apex_position:
                        # Entry phase: outside → inside
                        progress_to_apex = idx / max(1, apex_position)
                        target_fraction = 0.95 - 1.75 * progress_to_apex
                    else:
                        # Exit phase: inside → outside
                        progress_from_apex = (idx - apex_position) / max(1, seg_len - 1 - apex_position)
                        target_fraction = -0.95 + 1.75 * progress_from_apex
                    
                    # Apply constraint based on corner direction
                    if is_left_turn:
                        if target_fraction > 0:
                            opti.subject_to(n[i] >= target_fraction * w_left[i])
                        else:
                            opti.subject_to(n[i] <= target_fraction * w_right[i])
                    else:
                        if target_fraction > 0:
                            opti.subject_to(n[i] >= target_fraction * w_left[i])
                        else:
                            opti.subject_to(n[i] <= target_fraction * w_right[i])
                
                # Pre-corner positioning
                lookahead = min(10, seg_len // 2)
                for j in range(lookahead):
                    pre_idx = (s0 - lookahead + j) % N
                    if not is_corner[pre_idx]:
                        pre_progress = j / max(1, lookahead - 1)
                        pre_target = 0.3 + 0.5 * pre_progress
                        
                        if is_left_turn:
                            opti.subject_to(n[pre_idx] <= -pre_target * w_right[pre_idx])
                        else:
                            opti.subject_to(n[pre_idx] >= pre_target * w_left[pre_idx])
                
                # Speed reduction for sharp hairpins
                max_corner_curv = max(abs_curv[i] for i in seg_indices)
                if max_corner_curv > 0.015:
                    for i in seg_indices:
                        k = abs(float(curvature[i]))
                        if k > 0.01:
                            F_normal_est = vehicle.mass_kg * vehicle.gravity
                            a_lat_max = 0.90 * vehicle.mu_friction * F_normal_est / vehicle.mass_kg
                            R = 1.0 / (k + 1e-6)
                            v_corner_max = ca.sqrt(a_lat_max * R)
                            opti.subject_to(v[i] <= v_corner_max * 1.15)
            
            # ========================================================================
            # If corner is LESS than or equal to 120 degrees: Use original apex logic
            # ========================================================================
            else:
                # Find apex
                apex_idx = max(seg_indices, key=lambda i: abs_curv[i])
                
                # Apply inner-side constraint at apex only
                apex_inside_fraction = 0.6
                k = float(curvature[apex_idx])
                if k > 0:
                    # Left-hand corner: inside is to the left
                    opti.subject_to(n[apex_idx] >= apex_inside_fraction * w_left[apex_idx])
                elif k < 0:
                    # Right-hand corner: inside is to the right
                    opti.subject_to(n[apex_idx] <= -apex_inside_fraction * w_right[apex_idx])
                
                # Entry window constraint for longer corners
                if seg_len >= 20:
                    entry_fraction = 0.25
                    outside_fraction_require = 0.35
                    entry_len = max(3, int(entry_fraction * seg_len))
                    entry_len = min(entry_len, 30)
                    entry_indices = seg_indices[:entry_len]
                    
                    for i in entry_indices:
                        if is_left_turn:
                            # Outside is to the right
                            opti.subject_to(n[i] <= -outside_fraction_require * w_right[i])
                        else:
                            # Outside is to the left
                            opti.subject_to(n[i] >= outside_fraction_require * w_left[i])
                
                # Exit window constraint if straight follows
                straight_thresh = 0.002
                min_straight_len = 20
                exit_fraction = 0.35
                outside_fraction_require = 0.95
                
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
                    exit_len = max(3, int(exit_fraction * seg_len))
                    exit_len = min(exit_len, 30)
                    exit_indices = seg_indices[-exit_len:]
                    
                    for i in exit_indices:
                        if is_left_turn:
                            opti.subject_to(n[i] <= -outside_fraction_require * w_right[i])
                        else:
                            opti.subject_to(n[i] >= outside_fraction_require * w_left[i])
    
    except Exception as e:
        print(f"Warning: Corner detection skipped due to: {e}")
        pass
    
    return a_lat