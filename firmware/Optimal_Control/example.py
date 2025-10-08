# example_fixed.py
# Full MGSA CasADi example (fixed to run)
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# --- Vehicle / model parameters (example values, replace with real car data) ---
m = 1450.0        # mass [kg]
Iz = 3000.0       # yaw inertia [kg*m^2]
a = 1.2           # distance CG to front axle [m]
b = 1.4           # distance CG to rear axle [m]
L = a + b         # wheelbase

# Pacejka 'magic formula' parameters (example)
Bf, Cf, Df = 10.0, 1.9, 8000.0   # front (tune these)
Br, Cr, Dr = 12.0, 1.9, 8000.0   # rear

# Collocation degree and nodes (Radau)
d = 3

# Discretization
N = 30                   # control intervals (reduced for testing)
s_total = 400.0

# track geometry (example)
s = np.linspace(0, s_total, N+1)
cx = 50*np.cos(2*np.pi*s/300)
cy = 50*np.sin(2*np.pi*s/300)
half_width = 6.0

# step in s (nominal)
ds = s[1] - s[0]

# weights
w_time = 1.0
w_delta_smooth = 500.0
w_a_smooth = 50.0
w_delta_mag = 1.0
w_slack = 1e6     # heavy penalty for leaving track

# Helper: collocation coefficients (Radau)
def collocation_coeffs(d):
    import numpy as np
    if d == 3:
        tau = np.array([0.0, 0.155051025721682, 0.644948974278318, 1.0])
    else:
        tau = np.linspace(0, 1, d+1)
    C = np.zeros((d+1,d+1))
    D = np.zeros(d+1)
    B = np.zeros(d+1)
    for j in range(d+1):
        coeffs = np.poly1d([1.])
        for r in range(d+1):
            if r != j:
                denom = (tau[j]-tau[r])
                coeffs = np.poly1d(np.convolve(coeffs, [1, -tau[r]])) / denom
        D[j] = np.poly1d(coeffs)(1.0)
        B[j] = np.poly1d(coeffs).integ()(1.0) - np.poly1d(coeffs).integ()(0.0)
        deriv = np.polyder(coeffs)
        for r in range(d+1):
            C[j, r] = deriv(tau[r])
    return tau, C, D, B

tau, C, D, B = collocation_coeffs(d)

# CasADi MX symbols via ca.MX
MX = ca.MX
vertcat = ca.vertcat
Function = ca.Function
nlpsol = ca.nlpsol

# CasADi symbolic states and controls
x = MX.sym('x'); y = MX.sym('y'); psi = MX.sym('psi')
vx = MX.sym('vx'); vy = MX.sym('vy'); r = MX.sym('r')
state = vertcat(x, y, psi, vx, vy, r)

delta = MX.sym('delta'); Fx = MX.sym('Fx')
control = vertcat(delta, Fx)

# Pacejka lateral force using ca ops (works for MX symbolic alpha)
def pacejka_lateral_force_mx(Bp, Cp, Dp, alpha):
    # Fy = D * sin(C * atan(B * alpha))
    return Dp * ca.sin(Cp * ca.atan(Bp * alpha))

# vehicle dynamics (symbolic)
alpha_f = ca.atan2((vy + a*r), vx) - delta
alpha_r = ca.atan2((vy - b*r), vx)

Fyf = pacejka_lateral_force_mx(Bf, Cf, Df, alpha_f)
Fyr = pacejka_lateral_force_mx(Br, Cr, Dr, alpha_r)

xdot = vx*ca.cos(psi) - vy*ca.sin(psi)
ydot = vx*ca.sin(psi) + vy*ca.cos(psi)
psidot = r

vx_dot = (Fx - Fyf*ca.sin(delta)) / m + vy*r
vy_dot = (Fyr + Fyf*ca.cos(delta)) / m - vx*r
r_dot = (a*Fyf*ca.cos(delta) - b*Fyr) / Iz

f = vertcat(xdot, ydot, psidot, vx_dot, vy_dot, r_dot)

# Prebuild CasADi functions to evaluate dynamics to avoid recreating them many times
f_fun = Function('f_fun', [state, control], [f])

# Build NLP with collocation
w = []      # CasADi vars
w0 = []     # initial guess (numbers)
lbw = []
ubw = []

g = []      # constraints
lbg = []
ubg = []

# Initial state (start on centerline)
x0_val = float(cx[0]); y0_val = float(cy[0]); psi0_val = 0.0
vx0 = 8.0; vy0 = 0.0; r0 = 0.0

# initial state variable
Xk = MX.sym('X0', 6)
w += [Xk]; w0 += [x0_val, y0_val, psi0_val, vx0, vy0, r0]
lbw += [-1e3, -1e3, -1e3, 0.1, -50.0, -10.0]
ubw += [1e3, 1e3, 1e3, 80.0, 50.0, 10.0]

J = 0

# loop over intervals
for k in range(N):
    # control
    Uk = MX.sym(f'U_{k}', 2)
    w += [Uk]; w0 += [0.0, 0.0]; lbw += [-0.6, -8000.0]; ubw += [0.6, 8000.0]

    # collocation states
    Xc = []
    for j in range(1, d+1):
        Xkj = MX.sym(f'X_{k}_{j}', 6)
        Xc.append(Xkj)
        w += [Xkj]; w0 += [x0_val, y0_val, psi0_val, vx0, vy0, r0]
        lbw += [-1e3, -1e3, -1e3, 0.1, -50.0, -10.0]
        ubw += [1e3, 1e3, 1e3, 80.0, 50.0, 10.0]

    # collocation equations
    X_end = D[0]*Xk
    for j in range(1, d+1):
        xp_j = C[0, j]*Xk
        for rj in range(1, d+1):
            xp_j = xp_j + C[rj, j]*Xc[rj-1]
        # evaluate dynamics at collocation point
        xcoll = Xc[j-1]
        f_eval = f_fun(xcoll, Uk)  # returns MX vector
        # time step approx h_k = ds / vx_nom
        vx_nom = Xk[3]
        h_k = ds / (vx_nom + 1e-3)
        # collocation constraint
        g += [h_k * f_eval - xp_j]
        lbg += [0]*6; ubg += [0]*6
        X_end = X_end + D[j]*Xc[j-1]

    # new state at end
    Xkp1 = MX.sym(f'X_{k+1}', 6)
    w += [Xkp1]; w0 += [x0_val, y0_val, psi0_val, vx0, vy0, r0]
    lbw += [-1e3, -1e3, -1e3, 0.1, -50.0, -10.0]
    ubw += [1e3, 1e3, 1e3, 80.0, 50.0, 10.0]

    # continuity
    g += [X_end - Xkp1]; lbg += [0]*6; ubg += [0]*6

    # objective: approximate time
    J += w_time * h_k

    # simple regularization on steering magnitude
    J += w_delta_mag * Uk[0]**2

    # track constraint at interval end: dist^2 <= half_width^2 + slack
    dx = Xkp1[0] - float(cx[k+1])
    dy = Xkp1[1] - float(cy[k+1])
    dist2 = dx*dx + dy*dy
    slack_k = MX.sym(f'slack_{k}', 1)
    w += [slack_k]; w0 += [0.0]; lbw += [0.0]; ubw += [1e3]
    # dist2 - slack_k <= half_width^2  -> dist2 - slack_k - half_width^2 <= 0
    g += [dist2 - slack_k]
    lbg += [-1e20]; ubg += [half_width**2]

    # lateral acceleration approx via Pacejka at Xkp1 (build expressions)
    vx_sym = Xkp1[3]; vy_sym = Xkp1[4]; r_sym = Xkp1[5]
    alpha_f_k = ca.atan2((vy_sym + a*r_sym), vx_sym) - Uk[0]
    alpha_r_k = ca.atan2((vy_sym - b*r_sym), vx_sym)
    Fyf_k = pacejka_lateral_force_mx(Bf, Cf, Df, alpha_f_k)
    Fyr_k = pacejka_lateral_force_mx(Br, Cr, Dr, alpha_r_k)
    ay_approx = (Fyf_k*ca.cos(Uk[0]) + Fyr_k) / m
    a_y_max = 9.0
    g += [ay_approx]; lbg += [-1e20]; ubg += [a_y_max]

    # advance
    Xk = Xkp1

# collect NLP
w_vec = vertcat(*w)
g_vec = vertcat(*g)

nlp = {'x': w_vec, 'f': J, 'g': g_vec}

# solver options
opts = {
    'ipopt.max_iter': 1500,
    'ipopt.tol': 1e-4,
    'ipopt.print_level': 5,
    'print_time': True
}

solver = nlpsol('solver', 'ipopt', nlp, opts)

# Solve (use initial guess w0)
print('Starting solve... this may take a while')
sol = solver(x0=np.array(w0), lbx=np.array(lbw), ubx=np.array(ubw), lbg=np.array(lbg), ubg=np.array(ubg))

w_opt = np.array(sol['x']).flatten()

# Extract states for plotting (attempt simple parse)
ptr = 0
X_opt = []
for k in range(N+1):
    Xk_vals = w_opt[ptr:ptr+6]
    X_opt.append(Xk_vals)
    ptr += 6
    if k < N:
        ptr += 2           # U_k
        ptr += d*6         # collocation states
        ptr += 6           # X_{k+1} (it will be consumed in next loop iteration)
        ptr += 1           # slack

# convert to array
X_opt = np.array(X_opt).T

plt.figure(figsize=(8,8))
plt.plot(cx, cy, '--', label='centerline')
plt.plot(X_opt[0,:], X_opt[1,:], '-r', label='optimal (collocation + Pacejka)')
plt.axis('equal')
plt.legend()
plt.title('MGSA - Radau collocation + Pacejka tyres (fixed script)')
plt.show()

print('Finished')
