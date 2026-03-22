from __future__ import annotations

import math

import numpy as np

from pimento import make_public_case, simulate_nine_dof, simulate_point_mass


def main() -> None:
    case = make_public_case()
    sizing = case["sizing"]
    point_mass = simulate_point_mass(case["plan"], case["point_mass_config"])
    landing_error = np.linalg.norm(
        np.array([point_mass["x_m"][-1], point_mass["y_m"][-1]])
        - case["plan"].touchdown_xy_m
    )
    print("Public paper-like sizing case")
    print(f"  surface area      : {sizing.geometry.surface_area_m2:8.1f} m^2")
    print(f"  span              : {sizing.geometry.span_m:8.2f} m")
    print(f"  chord             : {sizing.geometry.chord_m:8.2f} m")
    print(f"  canopy mass       : {sizing.estimated_canopy_mass_kg:8.1f} kg")
    print(f"  wing loading      : {sizing.wing_loading_npm2:8.1f} N/m^2")
    print(f"  sink rate         : {sizing.estimated_sink_rate_mps:8.2f} m/s")
    print(f"  trajectory time   : {point_mass['time_s'][-1]:8.1f} s")
    print(f"  landing error     : {landing_error:8.1f} m")
    print(f"  touchdown phase   : {point_mass['phase'][-1]}")
    print(f"  final path angle  : {math.degrees(point_mass['flight_path_angle_rad'][-1]):8.2f} deg")

    nine_dof = simulate_nine_dof(
        initial_state=case["initial_nine_dof_state"],
        parameters=case["nine_dof_parameters"],
        duration_s=2.0,
        dt_s=0.02,
    )
    final_speed = np.linalg.norm(nine_dof["body_velocity_mps"][-1])
    print("Nine-DOF starter sanity check")
    print(f"  propagated time   : {nine_dof['time_s'][-1]:8.2f} s")
    print(f"  final body speed  : {final_speed:8.2f} m/s")


if __name__ == "__main__":
    main()
