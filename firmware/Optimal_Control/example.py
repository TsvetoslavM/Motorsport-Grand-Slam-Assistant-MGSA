# Requires: pip install casadi numpy matplotlib
import numpy as np
from casadi import SX, vertcat, Function, nlpsol
import matplotlib.pyplot as plt

# --- Parameters ---
L = 2.6            # wheelbase
N = 100            # number of discretization intervals (nodes = N+1)
s_total = 400.0

# discretization
ds = s_total / N

# state and control symbols
x = SX.sym('x')
y = SX.sym('y')
psi = SX.sym('psi')
v = SX.sym('v')
state = vertcat(x, y, psi, v)

delta = SX.sym('delta')
a = SX.sym('a')
control = vertcat(delta, a)

# kinematic bicycle dynamics
f = vertcat(v*SX.cos(psi), v*SX.sin(psi), v/L*SX.tan(delta), a)

# Decision variables (stacked)
X = SX.sym('X', 4, N+1)
U = SX.sym('U', 2, N)

# Cost weights (tune these)
w_time = 1.0
w_delta_smooth = 100.0    # penalize large steering rate
w_a_smooth = 10.0         # penalize large acceleration rate
w_delta_mag = 1.0         # penalize large steering magnitude
w_out_of_track = 1e5      # soft penalty weight for leaving track

# Bounds
v_min = 1.0
v_max = 60.0
delta_max = 0.6
a_max = 6.0

a_y_max = 9.0   # m/s^2 (approx) - can be set to mu*g

# Example centerline and track
s = np.linspace(0, s_total, N+1)
cx = 50*np.cos(2*np.pi*s/300)
cy = 50*np.sin(2*np.pi*s/300)
half_width = 6.0

# compute centerline derivatives to estimate curvature (numpy)
cx_p = np.gradient(cx, s)
cy_p = np.gradient(cy, s)
cx_pp = np.gradient(cx_p, s)
cy_pp = np.gradient(cy_p, s)
curv = (cx_p * cy_pp - cy_p * cx_pp) / (cx_p**2 + cy_p**2)**1.5
curv = np.nan_to_num(curv)

# Build objective and constraints
obj = 0
g = []

# dynamics constraints (Euler integration here for simplicity)
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    xk_next = X[:, k+1]
    fk = vertcat(xk[3]*SX.cos(xk[2]), xk[3]*SX.sin(xk[2]), xk[3]/L*SX.tan(uk[0]), uk[1])
    x_pred = xk + fk * ds
    g.append(x_pred - xk_next)
    # time objective (approx dt ~ ds / v)
    vk = xk[3]
    obj += w_time * (ds / vk)

# smoothness penalties on controls (finite differences)
for k in range(N-1):
    dk = U[0, k+1] - U[0, k]
    ak = U[1, k+1] - U[1, k]
    obj += w_delta_smooth * dk**2 + w_a_smooth * ak**2

# penalize steering magnitude (avoid extreme steering)
for k in range(N):
    obj += w_delta_mag * U[0, k]**2

# Track constraints (soft via penalty) and lateral accel limits
for k in range(N+1):
    dx = X[0, k] - cx[k]
    dy = X[1, k] - cy[k]
    # soft constraint: distance^2 <= half_width^2
    dist2 = dx*dx + dy*dy
    slack = SX.sym(f'slack_{k}')
    # instead of adding slack as extra var, we use penalty directly (soft)
    # if dist2 > half_width^2 we add a big penalty
    obj += w_out_of_track * SX.fmax(dist2 - half_width**2, 0)
    # lateral accel approximation: a_y = v^2 * curvature(s)
    ay_approx = X[3, k]**2 * curv[k]
    # absolute lateral accel: use abs but approximate with squared constraint
    g.append(ay_approx - a_y_max)

# pack variables into a single vector
vars_list = []
for k in range(N+1):
    vars_list += [X[:, k]]
for k in range(N):
    vars_list += [U[:, k]]
var_stack = vertcat(*vars_list)

# concatenate constraints
g_vec = vertcat(*g)

# NLP
nlp = {'x': var_stack, 'f': obj, 'g': g_vec}
opts = {"ipopt.print_level":0, "print_time":0}
solver = nlpsol('solver', 'ipopt', nlp, opts)

# initial guess (centerline + reasonable speed)
x0 = []
for k in range(N+1):
    x0 += [cx[k], cy[k], 0.0, 8.0]
for k in range(N):
    x0 += [0.0, 0.0]

# bounds for vars
lbx = []
ubx = []
for k in range(N+1):
    lbx += [-1e3, -1e3, -1e3, v_min]
    ubx += [ 1e3,  1e3,  1e3, v_max]
for k in range(N):
    lbx += [-delta_max, -a_max]
    ubx += [ delta_max,  a_max]

# bounds for constraints
# dynamics equality constraints: 4*N entries equal to zero
lbg = [0.0] * (4*N)
ubg = [0.0] * (4*N)
# lateral accel constraints: N+1 entries ay - a_y_max <= 0  -> lbg = -inf, ubg = 0
lbg += [-1e20] * (N+1)
ubg += [0.0] * (N+1)

# Solve
sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

w_opt = np.array(sol['x']).flatten()

# unpack
X_opt = w_opt[:4*(N+1)].reshape((N+1,4)).T
U_opt = w_opt[4*(N+1):].reshape((N,2)).T

# plot
plt.figure(figsize=(8,8))
plt.plot(cx, cy, '--', label='centerline')
plt.plot(X_opt[0,:], X_opt[1,:], '-r', label='optimal (smoothed)')
plt.axis('equal')
plt.legend()
plt.title('MGSA Optimal Control - smoothed objective & constraints')
plt.show()

print('Finished')
