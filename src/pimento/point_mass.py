from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .guidance import GuidanceSettings, PointMassPlan, TrajectoryGuidance
from .integrators import rk4_step
from .utils import wrap_angle


@dataclass(frozen=True)
class PointMassConfig:
    airspeed_mps: float
    nominal_lift_to_drag: float
    dt_s: float = 0.5
    max_time_s: float = 2400.0
    wind_ned_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    initial_heading_rad: float = 0.0
    initial_flight_path_angle_rad: float | None = None
    guidance: GuidanceSettings | None = None


@dataclass(frozen=True)
class PointMassState:
    x_m: float
    y_m: float
    altitude_m: float
    heading_rad: float
    flight_path_angle_rad: float
    airspeed_mps: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.x_m,
                self.y_m,
                self.altitude_m,
                self.heading_rad,
                self.flight_path_angle_rad,
                self.airspeed_mps,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "PointMassState":
        return cls(
            x_m=float(vector[0]),
            y_m=float(vector[1]),
            altitude_m=float(vector[2]),
            heading_rad=float(vector[3]),
            flight_path_angle_rad=float(vector[4]),
            airspeed_mps=float(vector[5]),
        )


def simulate_point_mass(
    plan: PointMassPlan,
    config: PointMassConfig,
) -> dict[str, np.ndarray]:
    guidance_settings = config.guidance or GuidanceSettings(
        nominal_lift_to_drag=config.nominal_lift_to_drag
    )
    guidance = TrajectoryGuidance(plan, guidance_settings)
    initial_gamma = (
        config.initial_flight_path_angle_rad
        if config.initial_flight_path_angle_rad is not None
        else -math.atan(1.0 / config.nominal_lift_to_drag)
    )
    state = PointMassState(
        x_m=float(plan.release_xy_m[0]),
        y_m=float(plan.release_xy_m[1]),
        altitude_m=plan.release_altitude_m,
        heading_rad=config.initial_heading_rad,
        flight_path_angle_rad=initial_gamma,
        airspeed_mps=config.airspeed_mps,
    )

    history: list[np.ndarray] = []
    phases: list[str] = []
    times: list[float] = []
    time_s = 0.0
    wind = np.asarray(config.wind_ned_mps, dtype=float)

    while time_s <= config.max_time_s and state.altitude_m > 0.0:
        command = guidance.command(
            position_xy_m=np.array([state.x_m, state.y_m], dtype=float),
            altitude_m=state.altitude_m,
            heading_rad=state.heading_rad,
            airspeed_mps=state.airspeed_mps,
            wind_xy_mps=wind[:2],
            dt_s=config.dt_s,
        )
        history.append(state.as_vector())
        phases.append(command.phase)
        times.append(time_s)

        def derivative(_: float, vector: np.ndarray) -> np.ndarray:
            x_m, y_m, altitude_m, heading_rad, gamma_rad, airspeed_mps = vector
            gamma_dot = (
                command.target_flight_path_angle_rad - gamma_rad
            ) / guidance_settings.flight_path_time_constant_s
            x_dot = airspeed_mps * math.cos(gamma_rad) * math.cos(heading_rad) + wind[0]
            y_dot = airspeed_mps * math.cos(gamma_rad) * math.sin(heading_rad) + wind[1]
            altitude_dot = airspeed_mps * math.sin(gamma_rad) + wind[2]
            return np.array(
                [
                    x_dot,
                    y_dot,
                    altitude_dot,
                    command.heading_rate_radps,
                    gamma_dot,
                    0.0,
                ],
                dtype=float,
            )

        new_vector = rk4_step(derivative, time_s, state.as_vector(), config.dt_s)
        new_vector[3] = wrap_angle(new_vector[3])
        new_vector[2] = max(new_vector[2], 0.0)
        state = PointMassState.from_vector(new_vector)
        time_s += config.dt_s

    history.append(state.as_vector())
    phases.append("touchdown" if state.altitude_m <= 0.0 else "timeout")
    times.append(time_s)
    stacked = np.vstack(history)
    return {
        "time_s": np.asarray(times, dtype=float),
        "x_m": stacked[:, 0],
        "y_m": stacked[:, 1],
        "altitude_m": stacked[:, 2],
        "heading_rad": stacked[:, 3],
        "flight_path_angle_rad": stacked[:, 4],
        "airspeed_mps": stacked[:, 5],
        "phase": np.asarray(phases, dtype=object),
    }
