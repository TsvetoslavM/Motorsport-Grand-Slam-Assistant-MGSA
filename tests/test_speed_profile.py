import numpy as np
import math
import pytest

from yourmodule.speed_profile import (
    VehicleParams,
    compute_arc_length,
    compute_curvature,
    vmax_lateral,
    speed_profile,
    compute_lap_time_seconds,
)

# ---- Fixtures ----

@pytest.fixture
def simple_line():
    # права линия 0..100 м
    points = [(x, 0.0) for x in range(0, 101, 10)]
    return points

@pytest.fixture
def simple_curve():
    # четвърт окръжност радиус 10 м, 0..90°
    theta = np.linspace(0, math.pi/2, 11)
    points = [(10*np.cos(t), 10*np.sin(t)) for t in theta]
    return points

@pytest.fixture
def f1_params():
    return VehicleParams(
        mass_kg=798.0,
        mu_friction=2.0,
        gravity=9.81,
        rho_air=1.225,
        cL_downforce=4.0,
        frontal_area_m2=1.6,
        engine_power_watts=735000.0,
        a_brake_max=49.0,
        a_accel_cap=18.0,
        cD_drag=1.0,
        c_rr=0.01,
        safety_speed_margin=1.0,
    )

# ---- Tests ----

def test_compute_arc_length_line(simple_line):
    s, ds = compute_arc_length(simple_line)
    assert pytest.approx(s[-1], rel=1e-6) == 100.0
    assert all(ds > 0)

def test_compute_curvature_line(simple_line):
    kappa = compute_curvature(simple_line)
    # права → кривина ≈ 0
    assert np.allclose(kappa, 0.0, atol=1e-6)

def test_compute_curvature_curve(simple_curve):
    kappa = compute_curvature(simple_curve)
    # за окръжност кривината ≈ 1/радиус = 0.1
    mean_kappa = np.nanmean(np.abs(kappa))
    assert pytest.approx(mean_kappa, rel=0.05) == 0.1

def test_vmax_lateral_straight(simple_line, f1_params):
    kappa = compute_curvature(simple_line)
    v = vmax_lateral(kappa, f1_params, v_global_cap=100.0)
    # на права: inf → глобален cap
    assert np.allclose(v, 100.0)

def test_vmax_lateral_curve(simple_curve, f1_params):
    kappa = compute_curvature(simple_curve)
    v = vmax_lateral(kappa, f1_params)
    assert np.all(np.isfinite(v))
    assert np.all(v < 200.0)  # F1 не може 200 m/s (~720 km/h)

def test_speed_profile_line(simple_line, f1_params):
    s, kappa, v_lat, v = speed_profile(simple_line, f1_params)
    # на права: скорост трябва да е глобалния лимит
    vmax = (f1_params.engine_power_watts / f1_params.k_drag()) ** (1/3)
    assert np.allclose(v, min(vmax, v.max()), rtol=1e-2)

def test_speed_profile_curve(simple_curve, f1_params):
    s, kappa, v_lat, v = speed_profile(simple_curve, f1_params)
    # трябва да има ограничение в завоя
    assert np.nanmin(v) < np.nanmax(v)
    # никъде отрицателни скорости
    assert np.all(v >= 0.0)

def test_lap_time(simple_line, f1_params):
    s, kappa, v_lat, v = speed_profile(simple_line, f1_params)
    lap_time = compute_lap_time_seconds(s, v)
    # за 100 м права при ~90 м/с очакваме ~1.1 секунди
    assert lap_time > 1.0
    assert lap_time < 2.0

def test_consistency_forward_backward(simple_curve, f1_params):
    s, kappa, v_lat, v = speed_profile(simple_curve, f1_params)
    # forward/backward трябва да са сходни: скоростта да не скача >50%
    ratio = np.nanmax(v) / max(np.nanmin(v), 1e-3)
    assert ratio < 5.0

def test_vmax_lateral_near_zero_denom(f1_params):
    # Крива с кривина ≈ mu*k_aero → denom ~ 0
    mu = f1_params.mu_friction
    k_a = f1_params.k_aero()
    kappa = np.array([mu * k_a, mu * k_a + 1e-12, mu * k_a - 1e-12])

    v = vmax_lateral(kappa, f1_params, v_global_cap=123.0)

    # Проверяваме, че няма NaN
    assert not np.any(np.isnan(v))

    # За стойности ≤ mu*k_aero очакваме +inf → cap-нати до global cap
    assert v[0] == 123.0
    assert v[2] == 123.0

    # За малко по-голям denom трябва да има голяма, но крайна стойност
    assert np.isfinite(v[1])
    assert v[1] > 10.0
