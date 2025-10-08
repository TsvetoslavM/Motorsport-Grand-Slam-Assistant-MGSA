import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Параметри на колата ---
L = 2.5       # междуосие [m]
mu = 1.8      # коеф. сцепление
g = 9.81
v_min, v_max = 1.0, 80.0   # [m/s]

# --- Параметри на оптимизация ---
N = 100  # брой точки по пистата
ds = 1.0 # разстояние между точките [m]

# Променливи
x = ca.MX.sym('x', N)
y = ca.MX.sym('y', N)
v = ca.MX.sym('v', N)
psi = ca.MX.sym('psi', N)
delta = ca.MX.sym('delta', N)
a = ca.MX.sym('a', N)

# Функционал за минимизация: минимално време
J = 0
for i in range(N - 1):
    J += ds / v[i]

# --- Ограничения ---
g_dyn = []

for i in range(N - 1):
    # кинематика на колата (Euler integration)
    x_next = x[i] + ds * ca.cos(psi[i])
    y_next = y[i] + ds * ca.sin(psi[i])
    psi_next = psi[i] + ds / L * ca.tan(delta[i])
    v_next = v[i] + ds * a[i]

    # добавяме разликите като ограничения
    g_dyn += [x_next - x[i + 1],
              y_next - y[i + 1],
              psi_next - psi[i + 1],
              v_next - v[i + 1]]

# сцепление: (a_lat^2 + a_long^2 <= (μg)^2)
for i in range(N):
    a_lat = v[i]**2 / (L / ca.tan(delta[i]))
    a_long = a[i]
    g_dyn.append(a_lat**2 + a_long**2 - (mu * g)**2)

# --- Граници на променливите ---
lbx, ubx = [], []
for i in range(N):
    lbx += [-10, -10, v_min, -np.pi, -0.4, -10]
    ubx += [ 10,  10, v_max,  np.pi,  0.4,  10]

# --- Формулираме NLP ---
opt_vars = ca.vertcat(x, y, v, psi, delta, a)
nlp = {'x': opt_vars, 'f': J, 'g': ca.vertcat(*g_dyn)}

solver = ca.nlpsol('solver', 'ipopt', nlp,
                   {'ipopt.print_level': 0,
                    'print_time': 0})

# --- Стартиране на оптимизация ---
sol = solver(lbx=lbx, ubx=ubx, lbg=0, ubg=0)
sol_x = np.array(sol['x']).flatten()

# --- Извличане на резултата ---
x_opt = sol_x[0:N]
y_opt = sol_x[N:2*N]

plt.plot(x_opt, y_opt, label='Optimal line')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.legend()
plt.show()
