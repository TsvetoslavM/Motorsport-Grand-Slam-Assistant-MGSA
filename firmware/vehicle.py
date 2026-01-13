from dataclasses import dataclass

# Centralized vehicle defaults (sourced from Optimal_Control/CasADi_IPOPT.py)
VEHICLE_DEFAULTS = {
    "mass_kg": 798.0,
    "mu_friction": 1.8,
    "gravity": 9.81,
    "rho_air": 1.225,
    "cL_downforce": 3.0,
    "cD_drag": 1.2,
    "frontal_area_m2": 1.5,
    "engine_power_watts": 750000.0,
    "brake_power_watts": 2500000.0,
    "a_accel_max": 12.0,
    "a_brake_max": 45.0,
    "a_lat_max": 60.0,
    "c_rr": 0.02,
    "wheelbase_m": 3.6,
    "v_min": 15.0,
    "v_max": 200.0,
    "safety_speed_margin": 1.0,
}


@dataclass
class VehicleParams:
    """Shared vehicle parameters used across vmax and optimal control."""

    mass_kg: float = VEHICLE_DEFAULTS["mass_kg"]
    mu_friction: float = VEHICLE_DEFAULTS["mu_friction"]
    gravity: float = VEHICLE_DEFAULTS["gravity"]
    rho_air: float = VEHICLE_DEFAULTS["rho_air"]
    cL_downforce: float = VEHICLE_DEFAULTS["cL_downforce"]
    cD_drag: float = VEHICLE_DEFAULTS["cD_drag"]
    frontal_area_m2: float = VEHICLE_DEFAULTS["frontal_area_m2"]
    engine_power_watts: float = VEHICLE_DEFAULTS["engine_power_watts"]
    brake_power_watts: float = VEHICLE_DEFAULTS["brake_power_watts"]
    a_accel_max: float = VEHICLE_DEFAULTS["a_accel_max"]
    a_accel_cap: float = VEHICLE_DEFAULTS["a_accel_max"]  # alias for vmax_raceline
    a_brake_max: float = VEHICLE_DEFAULTS["a_brake_max"]
    a_lat_max: float = VEHICLE_DEFAULTS["a_lat_max"]
    c_rr: float = VEHICLE_DEFAULTS["c_rr"]
    wheelbase_m: float = VEHICLE_DEFAULTS["wheelbase_m"]
    v_min: float = VEHICLE_DEFAULTS["v_min"]
    v_max: float = VEHICLE_DEFAULTS["v_max"]
    safety_speed_margin: float = VEHICLE_DEFAULTS["safety_speed_margin"]

    def k_aero(self) -> float:
        return (0.5 * self.rho_air * self.cL_downforce * self.frontal_area_m2) / max(self.mass_kg, 1e-9)

    def k_drag(self) -> float:
        return 0.5 * self.rho_air * self.cD_drag * self.frontal_area_m2

