from __future__ import annotations

import math

import numpy as np

from .aero import GenericAerodynamics
from .guidance import GuidanceSettings, OrbitSegment, PointMassPlan
from .nine_dof import NineDofParameters, NineDofState
from .point_mass import PointMassConfig
from .sizing import SizingInputs, estimate_parafoil


def make_public_case() -> dict[str, object]:
    sizing_inputs = SizingInputs(
        payload_mass_kg=2028.0,
        target_lift_to_drag=4.0,
        trim_speed_mps=15.0,
        trim_lift_coefficient=0.55,
        air_density_kgpm3=1.225,
        aspect_ratio=2.7,
        canopy_areal_density_kgpm2=0.27,
    )
    sizing = estimate_parafoil(sizing_inputs)
    aero = GenericAerodynamics.from_aspect_ratio(
        aspect_ratio=sizing.geometry.aspect_ratio,
        target_lift_to_drag=sizing_inputs.target_lift_to_drag,
        payload_drag_area_m2=0.08,
        reference_span_m=sizing.geometry.span_m,
        reference_chord_m=sizing.geometry.chord_m,
    )

    plan = PointMassPlan(
        release_xy_m=np.array([0.0, -900.0], dtype=float),
        release_altitude_m=5000.0,
        loiter=OrbitSegment(
            center_xy_m=np.array([500.0, -800.0], dtype=float),
            radius_m=150.0,
            exit_altitude_m=3300.0,
        ),
        energy_management=OrbitSegment(
            center_xy_m=np.array([2100.0, -450.0], dtype=float),
            radius_m=150.0,
            exit_altitude_m=1200.0,
        ),
        touchdown_xy_m=np.array([3550.0, 0.0], dtype=float),
        runway_heading_rad=0.0,
    )
    guidance = GuidanceSettings(
        nominal_lift_to_drag=sizing_inputs.target_lift_to_drag,
        final_leg_m=500.0,
        turn_radius_m=120.0,
        max_turn_rate_radps=math.radians(6.0),
    )
    point_mass_config = PointMassConfig(
        airspeed_mps=sizing_inputs.trim_speed_mps,
        nominal_lift_to_drag=sizing_inputs.target_lift_to_drag,
        dt_s=0.5,
        max_time_s=2500.0,
        wind_ned_mps=np.array([2.0, 0.0, 0.0], dtype=float),
        initial_heading_rad=0.0,
        guidance=guidance,
    )

    canopy_inertia = np.diag([2.5e4, 4.0e4, 2.8e4]).astype(float)
    payload_inertia = np.diag([1.1e4, 1.6e4, 1.3e4]).astype(float)
    nine_dof_parameters = NineDofParameters(
        mass_canopy_kg=sizing.estimated_canopy_mass_kg,
        mass_payload_kg=sizing.payload_mass_kg,
        canopy_inertia_kgm2=canopy_inertia,
        payload_inertia_kgm2=payload_inertia,
        canopy_area_m2=sizing.geometry.surface_area_m2,
        payload_drag_area_m2=0.08,
        reference_span_m=sizing.geometry.span_m,
        reference_chord_m=sizing.geometry.chord_m,
        line_length_m=sizing.geometry.line_length_m,
        rigging_angle_rad=math.radians(3.0),
        aero=aero,
    )
    initial_nine_dof_state = NineDofState(
        body_velocity_mps=np.array([point_mass_config.airspeed_mps, 0.0, -3.6], dtype=float),
        canopy_rates_radps=np.zeros(3, dtype=float),
        payload_rates_radps=np.zeros(3, dtype=float),
        canopy_euler_rad=np.zeros(3, dtype=float),
        payload_euler_rad=np.array([0.0, math.radians(3.0), 0.0], dtype=float),
        position_m=np.array([0.0, -900.0, 5000.0], dtype=float),
    )
    return {
        "sizing_inputs": sizing_inputs,
        "sizing": sizing,
        "aero": aero,
        "plan": plan,
        "point_mass_config": point_mass_config,
        "nine_dof_parameters": nine_dof_parameters,
        "initial_nine_dof_state": initial_nine_dof_state,
    }
