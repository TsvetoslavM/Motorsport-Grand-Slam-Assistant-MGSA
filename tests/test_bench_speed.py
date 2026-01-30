import time
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
    
def test_speed_profile_benchmark(f1_params):
    # Synthetic trajectory: sinusoid from 0..1000 m
    x = np.linspace(0, 1000, 2001)   # 2000 points
    y = 10.0 * np.sin(0.01 * x)      # small corners
    points = list(zip(x, y))

    start = time.perf_counter()
    s, kappa, v_lat, v = speed_profile(points, f1_params)
    elapsed = time.perf_counter() - start

    # Checks
    assert s[-1] > 900.0
    assert np.all(v >= 0.0)
    assert elapsed < 0.1, f"Speed profile too slow: {elapsed:.3f}s"
