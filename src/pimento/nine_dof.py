from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aero import GenericAerodynamics, evaluate_aerodynamics
from .integrators import rk4_step
from .utils import euler_rates_matrix_zyx, rotation_matrix_zyx

GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class NineDofParameters:
    mass_canopy_kg: float
    mass_payload_kg: float
    canopy_inertia_kgm2: np.ndarray
    payload_inertia_kgm2: np.ndarray
    canopy_area_m2: float
    payload_drag_area_m2: float
    reference_span_m: float
    reference_chord_m: float
    line_length_m: float
    rigging_angle_rad: float
    joint_stiffness_nmprad: float = 1200.0
    joint_damping_nmsprad: float = 350.0
    angular_damping_nms: float = 50.0
    aero: GenericAerodynamics | None = None

    def with_default_aero(self, aspect_ratio: float, target_lift_to_drag: float) -> "NineDofParameters":
        aero = self.aero or GenericAerodynamics.from_aspect_ratio(
            aspect_ratio=aspect_ratio,
            target_lift_to_drag=target_lift_to_drag,
            payload_drag_area_m2=self.payload_drag_area_m2,
            reference_span_m=self.reference_span_m,
            reference_chord_m=self.reference_chord_m,
        )
        return NineDofParameters(
            mass_canopy_kg=self.mass_canopy_kg,
            mass_payload_kg=self.mass_payload_kg,
            canopy_inertia_kgm2=self.canopy_inertia_kgm2,
            payload_inertia_kgm2=self.payload_inertia_kgm2,
            canopy_area_m2=self.canopy_area_m2,
            payload_drag_area_m2=self.payload_drag_area_m2,
            reference_span_m=self.reference_span_m,
            reference_chord_m=self.reference_chord_m,
            line_length_m=self.line_length_m,
            rigging_angle_rad=self.rigging_angle_rad,
            joint_stiffness_nmprad=self.joint_stiffness_nmprad,
            joint_damping_nmsprad=self.joint_damping_nmsprad,
            angular_damping_nms=self.angular_damping_nms,
            aero=aero,
        )


@dataclass(frozen=True)
class NineDofState:
    body_velocity_mps: np.ndarray
    canopy_rates_radps: np.ndarray
    payload_rates_radps: np.ndarray
    canopy_euler_rad: np.ndarray
    payload_euler_rad: np.ndarray
    position_m: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.body_velocity_mps,
                self.canopy_rates_radps,
                self.payload_rates_radps,
                self.canopy_euler_rad,
                self.payload_euler_rad,
                self.position_m,
            ]
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "NineDofState":
        return cls(
            body_velocity_mps=np.asarray(vector[0:3], dtype=float),
            canopy_rates_radps=np.asarray(vector[3:6], dtype=float),
            payload_rates_radps=np.asarray(vector[6:9], dtype=float),
            canopy_euler_rad=np.asarray(vector[9:12], dtype=float),
            payload_euler_rad=np.asarray(vector[12:15], dtype=float),
            position_m=np.asarray(vector[15:18], dtype=float),
        )


def _payload_drag_force(
    payload_velocity_body_mps: np.ndarray,
    density_kgpm3: float,
    drag_area_m2: float,
) -> np.ndarray:
    speed = float(np.linalg.norm(payload_velocity_body_mps))
    if speed < 1.0e-9:
        return np.zeros(3, dtype=float)
    return -0.5 * density_kgpm3 * drag_area_m2 * speed * payload_velocity_body_mps


def simulate_nine_dof(
    initial_state: NineDofState,
    parameters: NineDofParameters,
    density_kgpm3: float = 1.225,
    wind_inertial_mps: np.ndarray | None = None,
    dt_s: float = 0.02,
    duration_s: float = 10.0,
    symmetric_deflection_rad: float = 0.0,
    asymmetric_deflection_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    if parameters.aero is None:
        raise ValueError("NineDofParameters.aero must be provided or created with with_default_aero().")
    wind = np.zeros(3, dtype=float) if wind_inertial_mps is None else np.asarray(wind_inertial_mps, dtype=float)
    total_mass = parameters.mass_canopy_kg + parameters.mass_payload_kg

    def derivative(_: float, vector: np.ndarray) -> np.ndarray:
        state = NineDofState.from_vector(vector)
        roll_c, pitch_c, yaw_c = state.canopy_euler_rad
        roll_p, pitch_p, yaw_p = state.payload_euler_rad
        r_ib_canopy = rotation_matrix_zyx(roll_c, pitch_c, yaw_c)
        r_ib_payload = rotation_matrix_zyx(roll_p, pitch_p, yaw_p)
        wind_body = r_ib_canopy.T @ wind
        airspeed_canopy_body = state.body_velocity_mps - wind_body
        payload_velocity_payload = r_ib_payload.T @ (r_ib_canopy @ state.body_velocity_mps - wind)

        canopy_aero = evaluate_aerodynamics(
            air_velocity_body_mps=airspeed_canopy_body,
            density_kgpm3=density_kgpm3,
            surface_area_m2=parameters.canopy_area_m2,
            aero=parameters.aero,
            symmetric_deflection_rad=symmetric_deflection_rad,
            asymmetric_deflection_rad=asymmetric_deflection_rad,
            body_rates_radps=state.canopy_rates_radps,
        )
        payload_drag_payload = _payload_drag_force(
            payload_velocity_body_mps=payload_velocity_payload,
            density_kgpm3=density_kgpm3,
            drag_area_m2=parameters.payload_drag_area_m2,
        )
        payload_drag_canopy = r_ib_canopy.T @ (r_ib_payload @ payload_drag_payload)
        gravity_canopy = r_ib_canopy.T @ np.array([0.0, 0.0, -total_mass * GRAVITY_MPS2], dtype=float)

        relative_angles = state.payload_euler_rad - state.canopy_euler_rad - np.array(
            [0.0, parameters.rigging_angle_rad, 0.0],
            dtype=float,
        )
        relative_rates = state.payload_rates_radps - state.canopy_rates_radps
        joint_moment = (
            parameters.joint_stiffness_nmprad * relative_angles
            + parameters.joint_damping_nmsprad * relative_rates
        )

        translational_force = canopy_aero.force_body_n + payload_drag_canopy + gravity_canopy
        translational_accel = translational_force / total_mass - np.cross(
            state.canopy_rates_radps, state.body_velocity_mps
        )

        canopy_moment = canopy_aero.moment_body_nm + joint_moment - parameters.angular_damping_nms * state.canopy_rates_radps
        payload_moment = -joint_moment - parameters.angular_damping_nms * state.payload_rates_radps

        canopy_rates_dot = np.linalg.solve(
            parameters.canopy_inertia_kgm2,
            canopy_moment - np.cross(state.canopy_rates_radps, parameters.canopy_inertia_kgm2 @ state.canopy_rates_radps),
        )
        payload_rates_dot = np.linalg.solve(
            parameters.payload_inertia_kgm2,
            payload_moment - np.cross(state.payload_rates_radps, parameters.payload_inertia_kgm2 @ state.payload_rates_radps),
        )

        canopy_euler_dot = euler_rates_matrix_zyx(roll_c, pitch_c) @ state.canopy_rates_radps
        payload_euler_dot = euler_rates_matrix_zyx(roll_p, pitch_p) @ state.payload_rates_radps
        position_dot = r_ib_canopy @ state.body_velocity_mps

        return np.concatenate(
            [
                translational_accel,
                canopy_rates_dot,
                payload_rates_dot,
                canopy_euler_dot,
                payload_euler_dot,
                position_dot,
            ]
        )

    history: list[np.ndarray] = []
    times: list[float] = []
    time_s = 0.0
    state_vector = initial_state.as_vector()
    while time_s <= duration_s:
        history.append(state_vector.copy())
        times.append(time_s)
        state_vector = rk4_step(derivative, time_s, state_vector, dt_s)
        time_s += dt_s

    stacked = np.vstack(history)
    return {
        "time_s": np.asarray(times, dtype=float),
        "body_velocity_mps": stacked[:, 0:3],
        "canopy_rates_radps": stacked[:, 3:6],
        "payload_rates_radps": stacked[:, 6:9],
        "canopy_euler_rad": stacked[:, 9:12],
        "payload_euler_rad": stacked[:, 12:15],
        "position_m": stacked[:, 15:18],
    }
