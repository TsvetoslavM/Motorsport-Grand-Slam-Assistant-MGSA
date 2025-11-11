import casadi as ca
import numpy as np
from scipy.ndimage import gaussian_filter1d


def add_chicane_racing_line_cost(opti, n, curvature, w_left, w_right, chicane_sequences, ds_array, N):
    """
    Форсира траектория през шикана: outside → apex1 → apex2 → outside
    """
    chicane_cost = 0.0
    
    for chicane in chicane_sequences:
        corner1_indices = chicane['corner1']
        corner2_indices = chicane['corner2']
        dir1 = chicane['direction1']
        dir2 = chicane['direction2']
        
        if not corner1_indices or not corner2_indices:
            continue
        
        # 1️⃣ Намери apex1
        apex1_idx = max(corner1_indices, key=lambda i: abs(curvature[i])) + 1
        
        # 2️⃣ Намери apex2
        apex2_idx = min(corner2_indices) + 1
        
        # 3️⃣ Entry point (по-рано преди първия завой)
        entry_idx = (min(corner1_indices) - 3) % N  # 🔥 увеличено от 3 на 5
        
        # 4️⃣ Exit zone (удължена зона след apex2)
        # Вместо една точка, направи постепенен изход
        corner2_end = max(corner2_indices)
        
        # Намери колко дълъг трябва да бъде изходът (докато кривината падне)
        exit_end_idx = corner2_end
        for offset in range(1, 50):
            check_idx = (corner2_end + offset) % N
            # Спри когато:
            # 1) Кривината е много малка (почти права)
            # 2) Или кривината сменя посока (нов завой)
            if abs(curvature[check_idx]) < 0.01:
                exit_end_idx = check_idx
                break
            if offset > 5 and np.sign(curvature[check_idx]) != np.sign(curvature[apex2_idx]):
                exit_end_idx = (corner2_end + offset - 1) % N
                break
        
        # Създай пълна exit зона от apex2 до exit_end
        if exit_end_idx > apex2_idx:
            full_exit_indices = list(range(apex2_idx + 1, exit_end_idx + 1))
        else:
            full_exit_indices = list(range(apex2_idx + 1, N)) + list(range(0, exit_end_idx + 1))
                
        # 5️⃣ ENTRY: outside
        if dir1 == 'left':
            entry_target = -w_right[entry_idx] * 0.95
        else:
            entry_target = w_left[entry_idx] * 0.95
        
        chicane_cost += (n[entry_idx] - entry_target) ** 2 * 5000.0
        
        # 6️⃣ APEX1
        if dir1 == 'left':
            apex1_target = w_left[apex1_idx]
        else:
            apex1_target = -w_right[apex1_idx]
        
        chicane_cost += (n[apex1_idx] - apex1_target) ** 2 * 10000.0
        
        # 7️⃣ TRANSITION
        if apex2_idx > apex1_idx:
            transition_indices = list(range(apex1_idx, apex2_idx))
        else:
            transition_indices = list(range(apex1_idx, N)) + list(range(0, apex2_idx))
        
        if transition_indices:
            apex2_target = w_left[apex2_idx] * 1.02 if dir2 == 'left' else -w_right[apex2_idx] * 1.02
            for idx, trans_idx in enumerate(transition_indices):
                t = (idx + 2) / len(transition_indices)
                target = (1.5 - t) * apex1_target + (1 - t) * apex2_target
                chicane_cost += (n[trans_idx] - target) ** 2 * 3000.0
        
        # 8️⃣ APEX2
        if dir2 == 'left':
            apex2_target = w_left[apex2_idx] * 0.98
        else:
            apex2_target = -w_right[apex2_idx] * 0.98
        
        chicane_cost += (n[apex2_idx] - apex2_target) ** 2 * 10000.0
        
        if full_exit_indices:
            num_exit = len(full_exit_indices)
            
            # Определи финалната outside позиция
            if dir2 == 'left':
                final_outside = -w_right[exit_end_idx] * 0.99
            else:
                final_outside = w_left[exit_end_idx] * 0.99
            
            buffer_ratio = 0.6
            buffer_count = max(1, int(buffer_ratio * num_exit))
            blend_power = 0.001   # степен на забавяне (можеш да пробваш и 4)

            for idx, exit_idx in enumerate(full_exit_indices):
                t = idx / max(1, num_exit - 1)

                if idx < buffer_count:
                    # още по-бавно начало
                    blend = (idx / buffer_count) ** blend_power * 0.08
                else:
                    t_adj = (idx - buffer_count) / max(1, num_exit - buffer_count - 1)
                    # плавен преход към финалната линия
                    blend = 0.08 + t_adj ** blend_power

                target = (1 - blend) * apex2_target + blend * final_outside

                weight = 10000.0 * (1 - t) + 4000.0 * t
                chicane_cost += (n[exit_idx] - target) ** 2 * weight

                if idx > 0:
                    prev_exit_idx = full_exit_indices[idx - 1]
                    dn = n[exit_idx] - n[prev_exit_idx]
                    chicane_cost += dn ** 2 * 2000.0
    
    return chicane_cost

def classify_corner_types(curvature, ds_array, N):
    """Classify each segment into corner types based on curvature analysis.
    
    Returns:
        corner_types: dict with keys 'hairpin', 'chicane', 'sweeper', 'tight', 
                     'kink', 'decreasing', 'increasing', 'complex'
        Each value is a list of segment indices belonging to that type.
    """
    corner_types = {
        'hairpin': [],
        'chicane': [],
        'sweeper': [],
        'tight': [],
        'kink': [],
        'decreasing': [],
        'increasing': [],
        'complex': [],
        'straight': []
    }
    
    # Calculate curvature derivatives
    curv_derivative = np.zeros(N)
    for i in range(N):
        i_next = (i + 1) % N
        curv_derivative[i] = (curvature[i_next] - curvature[i]) / ds_array[i]
    
    # Smooth derivatives
    curv_derivative = gaussian_filter1d(curv_derivative, sigma=3, mode='wrap')
    
    # Threshold values for F1
    HAIRPIN_CURV = 0.08  # Very tight radius (~12.5m)
    TIGHT_CURV = 0.05    # Tight 90-degree turn
    SWEEPER_CURV = 0.02  # Fast sweeper
    KINK_CURV = 0.005    # Barely noticeable
    
    DECREASING_THRESHOLD = 0.001  # Curvature increasing (radius decreasing)
    INCREASING_THRESHOLD = -0.001  # Curvature decreasing (radius increasing)
    
    i = 0
    while i < N:
        abs_curv = abs(curvature[i])
        
        # STRAIGHT
        if abs_curv < KINK_CURV:
            corner_types['straight'].append(i)
            i += 1
            continue
        
        # KINK - very slight corner
        if abs_curv < SWEEPER_CURV:
            corner_types['kink'].append(i)
            i += 1
            continue
        
        # Find corner extent
        corner_start = i
        corner_sign = np.sign(curvature[i])
        
        # Continue while same direction and significant curvature
        while i < N and np.sign(curvature[i]) == corner_sign and abs(curvature[i]) >= KINK_CURV:
            i += 1
        
        corner_end = i - 1
        corner_length = (corner_end - corner_start + 1) % N
        
        if corner_length < 2:
            i += 1
            continue
        
        # Get corner statistics
        corner_indices = [(corner_start + j) % N for j in range(corner_length)]
        corner_curvatures = [abs(curvature[idx]) for idx in corner_indices]
        max_curv = max(corner_curvatures)
        avg_curv = np.mean(corner_curvatures)
        
        # Check for radius change
        corner_derivatives = [curv_derivative[idx] for idx in corner_indices]
        avg_derivative = np.mean(corner_derivatives)
        
        # HAIRPIN - very tight, ~180 degrees
        if max_curv > HAIRPIN_CURV:
            corner_types['hairpin'].extend(corner_indices)
        
        # TIGHT CORNER - 90 degree
        elif max_curv > TIGHT_CURV:
            # Check if radius is decreasing or increasing
            if avg_derivative > DECREASING_THRESHOLD:
                corner_types['decreasing'].extend(corner_indices)
            elif avg_derivative < INCREASING_THRESHOLD:
                corner_types['increasing'].extend(corner_indices)
            else:
                corner_types['tight'].extend(corner_indices)
        
        # SWEEPER - fast corner
        elif max_curv > SWEEPER_CURV:
            corner_types['sweeper'].extend(corner_indices)
        
    # Detect CHICANES - alternating direction corners close together
    chicane_sequences = []  # НОВО: пази последователности от шикани
    chicane_window = 50  # meters
    
    for i in range(N):
        if curvature[i] == 0 or abs(curvature[i]) < SWEEPER_CURV:
            continue
        
        # Look ahead for opposite curvature
        distance = 0
        for j in range(1, 50):
            idx = (i + j) % N
            distance += ds_array[(i + j - 1) % N]
            
            if distance > chicane_window:
                break
            
            if np.sign(curvature[idx]) == -np.sign(curvature[i]) and abs(curvature[idx]) > SWEEPER_CURV:
                # Found chicane - пази като двойка завои
                corner1_indices = []
                corner2_indices = []
                
                # Намери пълния обхват на първия завой
                for k in range(i-5, i+10):
                    k_idx = k % N
                    if abs(curvature[k_idx]) > SWEEPER_CURV and np.sign(curvature[k_idx]) == np.sign(curvature[i]):
                        corner1_indices.append(k_idx)
                
                # Намери пълния обхват на втория завой
                for k in range(idx-5, idx+10):
                    k_idx = k % N
                    if abs(curvature[k_idx]) > SWEEPER_CURV and np.sign(curvature[k_idx]) == np.sign(curvature[idx]):
                        corner2_indices.append(k_idx)
                
                if corner1_indices and corner2_indices:
                    chicane_sequences.append({
                        'corner1': corner1_indices,
                        'corner2': corner2_indices,
                        'direction1': 'left' if curvature[i] > 0 else 'right',
                        'direction2': 'left' if curvature[idx] > 0 else 'right'
                    })
                    
                    # Маркирай индексите като chicane
                    for k_idx in corner1_indices + corner2_indices:
                        if k_idx not in corner_types['chicane']:
                            corner_types['chicane'].append(k_idx)
                break
    
    corner_types['chicane_sequences'] = chicane_sequences  # НОВО
    # Detect COMPLEX - multiple corners in sequence
    complex_window = 100  # meters
    for i in range(N):
        if abs(curvature[i]) < SWEEPER_CURV:
            continue
        
        corner_count = 0
        distance = 0
        
        for j in range(1, 50):
            idx = (i + j) % N
            distance += ds_array[(i + j - 1) % N]
            
            if distance > complex_window:
                break
            
            if abs(curvature[idx]) > SWEEPER_CURV:
                corner_count += 1
        
        if corner_count > 3:
            for k in range(min(50, N)):
                complex_idx = (i + k) % N
                if abs(curvature[complex_idx]) > SWEEPER_CURV:
                    if complex_idx not in corner_types['complex']:
                        corner_types['complex'].append(complex_idx)
    
    return corner_types


def find_corner_phases(curvature, ds_array, N):
    """Identify entry, apex, and exit phases of corners for proper racing line.
    
    Returns:
        corner_phases: dict with 'entry', 'apex', 'exit' lists
    """
    corner_phases = {
        'entry': [],
        'apex': [],
        'exit': []
    }
    
    # Smooth curvature to find peaks
    curv_smooth = gaussian_filter1d(np.abs(curvature), sigma=5, mode='wrap')
    
    CORNER_THRESHOLD = 0.01
    
    i = 0
    while i < N:
        if abs(curvature[i]) < CORNER_THRESHOLD:
            i += 1
            continue
        
        # Found a corner, find its extent
        corner_start = i
        corner_sign = np.sign(curvature[i])
        
        # Find corner end
        while i < N and np.sign(curvature[i]) == corner_sign and abs(curvature[i]) >= CORNER_THRESHOLD * 0.5:
            i += 1
        
        corner_end = (i - 1) % N
        corner_length = (corner_end - corner_start + 1) % N
        
        if corner_length < 3:
            continue
        
        # Find apex (max curvature point)
        corner_indices = [(corner_start + j) % N for j in range(corner_length)]
        corner_curvatures = [abs(curvature[idx]) for idx in corner_indices]
        apex_local_idx = np.argmax(corner_curvatures)
        apex_idx = corner_indices[apex_local_idx]
        
        # Define phases
        entry_length = max(1, apex_local_idx)
        exit_length = max(1, corner_length - apex_local_idx - 1)
        
        # Entry: from corner start to 2 segments before apex
        for j in range(max(0, apex_local_idx - 2)):
            entry_idx = (corner_start + j) % N
            corner_phases['entry'].append(entry_idx)
        
        # Apex: apex and 1-2 segments around it
        for j in range(max(0, apex_local_idx - 1), min(corner_length, apex_local_idx + 2)):
            apex_idx_current = (corner_start + j) % N
            corner_phases['apex'].append(apex_idx_current)
        
        # Exit: from 2 segments after apex to corner end
        for j in range(min(corner_length, apex_local_idx + 2), corner_length):
            exit_idx = (corner_start + j) % N
            corner_phases['exit'].append(exit_idx)
    
    return corner_phases


def add_apex_constraints(opti, n, curvature, w_left, w_right, corner_phases, N):
    tol = 0.03  # 5 cm tolerance (примерно) — настрой според размерите на w_left/w_right
    slack_apex = opti.variable(len(corner_phases['apex']))  # ако искаш slack
    opti.set_initial(slack_apex, 0.0)
    opti.subject_to(slack_apex >= 0)
    slack_weight = 1000.0  # penalize slack in objective (ако го ползваш)

    single_apices = extract_single_apex_indices(curvature, corner_phases)
    for idx_i, apex_idx in enumerate(single_apices):
        if abs(curvature[apex_idx]) < 0.01:
            continue

        if curvature[apex_idx] > 0:  # left turn
            apex_target = w_left[apex_idx] * 0.95
        else:
            apex_target = -w_right[apex_idx] * 0.95

        # жестко но с толеранс: apex_target +/- tol, допускаме slack_apex[idx_i]
        opti.subject_to(n[apex_idx] >= apex_target - tol - slack_apex[idx_i])
        opti.subject_to(n[apex_idx] <= apex_target + tol + slack_apex[idx_i])



def add_racing_line_geometry_cost(opti, n, curvature, w_left, w_right, corner_phases, ds_array, N, corner_types=None):
    """
    🔥 Добави corner_types като параметър
    """
    racing_line_cost = 0.0

    KINK_CURV = 0.005
    MIN_CORNER_LEN = 5
    STRAIGHT_WEIGHT = 1000.0

    # Helper to find corner segments
    corner_segments = []
    in_corner = False
    start_idx = 0
    for i in range(N):
        if abs(curvature[i]) >= KINK_CURV and not in_corner:
            in_corner = True
            start_idx = i
        elif abs(curvature[i]) < KINK_CURV and in_corner:
            in_corner = False
            if i - start_idx >= MIN_CORNER_LEN:
                corner_segments.append((start_idx, i))
    if in_corner:
        corner_segments.append((start_idx, N - 1))

    # Compute corner directions (left/right)
    corner_info = []
    for s, e in corner_segments:
        avg_sign = np.sign(np.sum(curvature[s:e+1]))
        direction = "left" if avg_sign > 0 else "right"
        corner_info.append((s, e, direction))

    # 1️⃣ Regular corner geometry penalty (keeps apex inside)
    for (s, e, direction) in corner_info:
        seg_indices = list(range(s, e + 1))
        abs_curv = [abs(float(curvature[i])) for i in seg_indices]
        apex_idx = seg_indices[np.argmax(abs_curv)]
        apex_position = seg_indices.index(apex_idx)

        for idx, i in enumerate(seg_indices):
            progress = idx / max(1, len(seg_indices) - 1)
            if direction == "left":
                target = (
                    -w_right[i] * (1 - progress)
                    + w_left[i] * progress
                )
            else:
                target = (
                    w_left[i] * (1 - progress)
                    - w_right[i] * progress
                )
            racing_line_cost += (n[i] - target) ** 2 * 1000.0

    # 2️⃣ Enhanced smooth gradual transition between corners with chicane logic
    def is_corner_in_chicane(corner_start, corner_end, corner_types):
        """Check if a corner segment is part of any chicane sequence."""
        if corner_types is None or 'chicane_sequences' not in corner_types:
            return False
        
        # Create set of indices for the corner segment (handling wrap-around)
        if corner_end >= corner_start:
            corner_indices = set(range(corner_start, corner_end + 1))
        else:
            corner_indices = set(list(range(corner_start, N)) + list(range(0, corner_end + 1)))
        
        # Check each chicane sequence
        for chicane in corner_types['chicane_sequences']:
            chicane_indices = set()
            # Add all indices from corner1
            for idx in chicane['corner1']:
                chicane_indices.add(idx)
            # Add all indices from corner2
            for idx in chicane['corner2']:
                chicane_indices.add(idx)
            
            # If there's any overlap, this corner is part of a chicane
            if corner_indices & chicane_indices:  # Set intersection
                return True
        
        return False

    for i in range(len(corner_info) - 1):
        exit_start, exit_end, exit_dir = corner_info[i]
        next_start, next_end, next_dir = corner_info[i + 1]

        mid_indices = list(range(exit_end, next_start))
        if len(mid_indices) < 3:
            continue

        if exit_dir == "left":
            exit_side = w_left[exit_end]
        else:
            exit_side = w_right[exit_end]

        # Enhanced chicane detection for the next corner
        is_next_a_chicane = is_corner_in_chicane(next_start, next_end, corner_types)
        
        # Calculate entry_side based on chicane status
        if is_next_a_chicane:
            # For chicane corners: outside-to-inside approach for optimal chicane entry
            if next_dir == "left":
                entry_side = w_right[next_start]  # Approach from right (positive) for left chicane turn
            else:
                entry_side = -w_left[next_start]  # Approach from left (negative) for right chicane turn
        else:
            # Normal corner entry logic: outside approach
            if next_dir == "left":
                entry_side = -w_right[next_start]  # Approach from right (negative) for left turn
            else:
                entry_side = w_left[next_start]    # Approach from left (positive) for right turn

        # Apply smooth transition
        for idx, i_mid in enumerate(mid_indices):
            t = idx / max(1, len(mid_indices) - 1)
            blend = 1.0 - 1.0 * np.cos(np.pi * t)
            target = (1 - blend) * exit_side + blend * entry_side
            racing_line_cost += (n[i_mid] - target) ** 2 * STRAIGHT_WEIGHT

    # 3️⃣ Path length penalty to keep centerline reasonable
    path_length_cost = 0.0
    for i in range(N):
        if abs(curvature[i]) < KINK_CURV:
            path_length_cost += n[i] ** 2

    racing_line_cost += path_length_cost * 0.000001

    # 🆕 4️⃣ Добави chicane cost (АКО corner_types е подаден)
    if corner_types is not None and 'chicane_sequences' in corner_types:
        chicane_cost = add_chicane_racing_line_cost(
            opti, n, curvature, w_left, w_right, 
            corner_types['chicane_sequences'], ds_array, N
        )
        racing_line_cost += chicane_cost

    return racing_line_cost


def extract_single_apex_indices(curvature, corner_phases):
    # Връща един apex index за всеки групиран апекс сегмент (максимална крива)
    apex_list = sorted(set(corner_phases.get('apex', [])))
    if not apex_list:
        return []

    N = len(curvature)
    groups = []
    current = [apex_list[0]]
    for idx in apex_list[1:]:
        prev = current[-1]
        if idx == (prev + 1) % N:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    apices = []
    for g in groups:
        max_idx = max(g, key=lambda i: abs(curvature[i]))
        apices.append(max_idx)
    return apices


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
    """Add all problem constraints to the Opti instance with corner-type awareness.

    Returns:
        a_lat (ca.MX): Vector of lateral acceleration terms used in post-processing.
        corner_types (dict): Classification of corner types for analysis.
    """
    N = v.shape[0]

    # Classify corner types and phases
    corner_types = classify_corner_types(curvature, ds_array, N)
    corner_phases = find_corner_phases(curvature, ds_array, N)

    # 1) Track boundaries
    boundary_margin_default = 0.1
    boundary_margin_apex = 0.0  # позволи достигане до кърба на апекса
    single_apices = extract_single_apex_indices(curvature, corner_phases)

    for i in range(N):
        if i in single_apices:
            bm = boundary_margin_apex
        else:
            bm = boundary_margin_default
        opti.subject_to(n[i] >= -w_left[i] + bm)
        opti.subject_to(n[i] <= w_right[i] - bm)


    # 1b) 🔥🔥🔥 Add EXTREME apex constraints to force MAXIMUM inside trajectory
    add_apex_constraints(opti, n, curvature, w_left, w_right, corner_phases, N)

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
        eps = 1e-3
        denom = 1.0 - kappa_center * n[i]
        denom = ca.sign(denom) * ca.fmax(ca.fabs(denom), eps)
        kappa_path = kappa_center / denom
        lateral_factor = 0.8  # Reduce effective curvature
        a_lat[i] = v[i] ** 2 * kappa_path * lateral_factor

    # 5) Combined traction circle (ellipse) - RELAXED
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
        opti.subject_to(P_required <= vehicle.engine_power_watts * 1.1 + slack_power[i])
        opti.subject_to(P_required >= -vehicle.brake_power_watts - slack_power[i])

    # 7) Longitudinal acceleration bounds
    for i in range(N):
        opti.subject_to(a_lon[i] >= -vehicle.a_brake_max)
        opti.subject_to(a_lon[i] <= vehicle.a_accel_max)

    # 8) Path smoothness with adaptive limits
    # Path smoothness adaptive around apex
    single_apices = extract_single_apex_indices(curvature, corner_phases)

    for i in range(N):
        i_next = (i + 1) % N
        dn = n[i_next] - n[i]

        if i in single_apices or i_next in single_apices:
            max_dn = 4.0   # позволи бърза промяна около апекса
        else:
            if abs(curvature[i]) > 0.02 or abs(curvature[i_next]) > 0.02:
                max_dn = 2.0
            else:
                max_dn = 0.8

        opti.subject_to(dn >= -max_dn)
        opti.subject_to(dn <= max_dn)

    # d2n
    for i in range(N):
        i_prev = (i - 1) % N
        i_next = (i + 1) % N
        if i in single_apices or i_prev in single_apices or i_next in single_apices:
            max_d2n_local = 2.5
        else:
            max_d2n_local = 0.8
        d2n = n[i_next] - 2 * n[i] + n[i_prev]
        opti.subject_to(d2n >= -max_d2n_local)
        opti.subject_to(d2n <= max_d2n_local)



    # 9) Jerk limits
    max_jerk = 30.0
    for i in range(N):
        i_next = (i + 1) % N
        v_avg = 0.5 * (v[i] + v[i_next])
        dt = ds_array[i] / (v_avg + 1e-3)
        jerk = (a_lon[i_next] - a_lon[i]) / dt
        opti.subject_to(jerk >= -max_jerk)
        opti.subject_to(jerk <= max_jerk)

    return a_lat, corner_types, corner_phases


def create_objective_with_racing_line(
    opti, v, a_lon, slack_power, n, curvature, w_left, w_right, 
    corner_phases, ds_array, N, vehicle
):
    """
    Create objective function that balances lap time with proper racing line.
    """
    
    # 🆕 Класифицирай corner_types тук, за да ги подадеш
    corner_types = classify_corner_types(curvature, ds_array, N)
    
    # Primary objective: minimize lap time
    lap_time = 0
    for i in range(N):
        i_next = (i + 1) % N
        v_avg = 0.5 * (v[i] + v[i_next])
        dt = ds_array[i] / (v_avg + 1e-3)
        lap_time += dt
    
    # Penalty for using power slack
    power_slack_penalty = ca.sum1(slack_power ** 2)
    
    # 🔥🔥🔥 Racing line geometry cost with corner_types
    racing_line_cost = add_racing_line_geometry_cost(
        opti, n, curvature, w_left, w_right, corner_phases, ds_array, N, corner_types  # 🆕 подай corner_types
    )
    
    # Smoothness cost (penalize excessive lateral movement)
    smoothness_cost = 0
    for i in range(N):
        i_next = (i + 1) % N
        dn = n[i_next] - n[i]
        smoothness_cost += dn ** 2
    
    # Path length cost (shorter path is better, but not at expense of apex)
    path_length_cost = 0
    for i in range(N):
        if abs(curvature[i]) < 0.005:
            path_length_cost += n[i] ** 2
    
    # Combined objective
    total_cost = (
        lap_time * 1.0 +
        power_slack_penalty * 0.0001 +
        racing_line_cost * 0.0002 +
        smoothness_cost * 1.0 +
        path_length_cost * 0.0000005
    )
    
    opti.minimize(total_cost)
    
    return total_cost


def initialize_with_proper_racing_line(
    opti, n, v, a_lon, curvature, ds_array, w_left, w_right, vehicle, N
):
    """Initialize optimization variables with proper racing line: outside-apex-outside.
    
    🔥🔥🔥 MAXIMUM AGGRESSION: Initialize with extreme apex positions.
    """
    
    # Classify corners and phases
    corner_types = classify_corner_types(curvature, ds_array, N)
    corner_phases = find_corner_phases(curvature, ds_array, N)
    
    # Initialize lateral position with racing line logic
    n_init = np.zeros(N)
    # 🆕 Специална инициализация за шикани
    if 'chicane_sequences' in corner_types:
        for chicane in corner_types['chicane_sequences']:
            corner1 = chicane['corner1']
            corner2 = chicane['corner2']
            dir1 = chicane['direction1']
            dir2 = chicane['direction2']
            
            apex1_idx = max(corner1, key=lambda i: abs(curvature[i]))
            apex2_idx = max(corner2, key=lambda i: abs(curvature[i]))
            
            # Entry
            entry_idx = (min(corner1) - 3) % N
            n_init[entry_idx] = w_right[entry_idx] * 0.85 if dir1 == 'left' else -w_left[entry_idx] * 0.85
            
            # Apex1
            n_init[apex1_idx] = -w_left[apex1_idx] * 0.95 if dir1 == 'left' else w_right[apex1_idx] * 0.95
            
            # Transition
            if apex2_idx > apex1_idx:
                trans = list(range(apex1_idx + 1, apex2_idx))
            else:
                trans = list(range(apex1_idx + 1, N)) + list(range(0, apex2_idx))
            
            for idx, t_idx in enumerate(trans):
                t = (idx + 1) / (len(trans) + 1)
                apex2_target = -w_left[apex2_idx] * 0.95 if dir2 == 'left' else w_right[apex2_idx] * 0.95
                n_init[t_idx] = (1 - t) * n_init[apex1_idx] + t * apex2_target
            
            # Apex2
            n_init[apex2_idx] = -w_left[apex2_idx] * 0.95 if dir2 == 'left' else w_right[apex2_idx] * 0.95
            
            # Exit
            exit_idx = (max(corner2) + 3) % N
            n_init[exit_idx] = w_right[exit_idx] * 0.85 if dir2 == 'left' else -w_left[exit_idx] * 0.85
            
    for i in range(N):
        abs_curv = abs(curvature[i])
        
        if abs_curv < 0.005:  # Straight
            n_init[i] = 0.0
            continue
        
        is_left_turn = curvature[i] > 0
        
        # ENTRY: outside of corner
        if i in corner_phases['entry']:
            if is_left_turn:
                n_init[i] = w_right[i] * 0.7  # Right side for left turn
            else:
                n_init[i] = -w_left[i] * 0.7  # Left side for right turn
        
        # 🔥🔥🔥 APEX: EXTREME inside kerb position!
        elif i in corner_phases['apex']:
            if is_left_turn:
                n_init[i] = -w_left[i] * 0.95  # 🔥🔥 Was 0.90, now 0.95 (KISSING the kerb!)
            else:
                n_init[i] = w_right[i] * 0.95  # 🔥🔥 Was 0.90, now 0.95 (KISSING the kerb!)
        
        # EXIT: outside of corner
        elif i in corner_phases['exit']:
            if is_left_turn:
                n_init[i] = w_right[i] * 0.8  # Right side for left turn
            else:
                n_init[i] = -w_left[i] * 0.8  # Left side for right turn
        
        # Default: geometric racing line
        else:
            if is_left_turn:
                # Interpolate based on position in corner
                n_init[i] = w_right[i] * 0.5
            else:
                n_init[i] = -w_left[i] * 0.5
    
    # Извлечи единични апекси
    single_apices = extract_single_apex_indices(curvature, corner_phases)

    # Направи локален Gaussian pull към апекса (остър апекс)
    n_init_before_smooth = n_init.copy()
    for apex_idx in single_apices:
        if abs(curvature[apex_idx]) < 0.01:
            continue

        # целева позиция на апекса
        if curvature[apex_idx] > 0:  # left turn
            apex_target = w_left[apex_idx] * 0.95
        else:
            apex_target = -w_right[apex_idx] * 0.95

        # Параметри: small sigma => по-остър pull (тест: 2..4)
        sigma_idx = 1.0
        window = int(max(3, round(sigma_idx * 4)))  # обхват около апекса

        # Apply gaussian influence on indices around apex
        for offset in range(-window, window+1):
            i = (apex_idx + offset) % N
            influence = np.exp(-0.5 * (offset / sigma_idx)**2)
            # Миксваме текущата стойност с целта, силно локално
            n_init_before_smooth[i] = (1.0 - influence) * n_init_before_smooth[i] + influence * apex_target

    # По-слаб глобален smooth: sigma малък (или го махни)
    n_init = gaussian_filter1d(n_init_before_smooth, sigma=2, mode='wrap')

    
    # Initialize velocity
    v_init = np.full(N, vehicle.v_max)
    
    for i in range(N):
        abs_curv = abs(curvature[i])
        
        if abs_curv < 1e-6:
            continue
        
        # Calculate path curvature
        kappa_center = curvature[i]
        denom = max(1.0 - kappa_center * n_init[i], 1e-3)
        kappa_path = kappa_center / denom
        
        # Speed limit from lateral acceleration
        a_lat_max = vehicle.a_lat_max
        v_max_corner = np.sqrt(a_lat_max / (abs(kappa_path) + 1e-6))
        
        # Reduce for entry, normal for apex, increase for exit
        if i in corner_phases['entry']:
            v_init[i] = min(v_max_corner * 0.85, v_init[i])
        elif i in corner_phases['apex']:
            v_init[i] = min(v_max_corner * 0.90, v_init[i])
        elif i in corner_phases['exit']:
            v_init[i] = min(v_max_corner * 0.95, v_init[i])
        else:
            v_init[i] = min(v_max_corner, v_init[i])
    
    # Forward/backward pass for realistic velocity profile
    for iteration in range(5):
        # Forward pass
        for i in range(N):
            i_prev = (i - 1) % N
            dt = ds_array[i_prev] / (v_init[i_prev] + 1e-3)
            v_reachable = v_init[i_prev] + vehicle.a_accel_max * dt
            v_init[i] = min(v_init[i], v_reachable)
        
        # Backward pass
        for i in range(N-1, -1, -1):
            i_next = (i + 1) % N
            dt = ds_array[i] / (v_init[i_next] + 1e-3)
            v_reachable = v_init[i_next] + vehicle.a_brake_max * dt
            v_init[i] = min(v_init[i], v_reachable)
    
    v_init = np.clip(v_init, vehicle.v_min, vehicle.v_max)
    
    # Initialize acceleration
    a_init = np.zeros(N)
    for i in range(N):
        i_next = (i + 1) % N
        dv = v_init[i_next] - v_init[i]
        dt = ds_array[i] / (v_init[i] + 1e-3)
        a_init[i] = dv / dt
    
    a_init = np.clip(a_init, -vehicle.a_brake_max, vehicle.a_accel_max)
    
    # Set initial values
    opti.set_initial(n, n_init)
    opti.set_initial(v, v_init)
    opti.set_initial(a_lon, a_init)
    
    return corner_types, corner_phases