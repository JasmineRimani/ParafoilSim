from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .utils import unit


@dataclass(frozen=True)
class GenericAerodynamics:
    cl_0: float
    cl_alpha: float
    cd_0: float
    cd_alpha: float
    cd_alpha2: float
    cy_beta: float
    cm_0: float
    cm_alpha: float
    cm_q: float
    cl_beta: float
    cl_p: float
    cl_r: float
    cl_da: float
    cn_beta: float
    cn_p: float
    cn_r: float
    cn_da: float
    cl_ds: float
    cd_ds: float
    payload_drag_area_m2: float
    max_deflection_rad: float
    trim_alpha_rad: float
    target_lift_to_drag: float
    reference_span_m: float
    reference_chord_m: float

    @classmethod
    def from_aspect_ratio(
        cls,
        aspect_ratio: float,
        target_lift_to_drag: float = 4.0,
        trim_alpha_deg: float = 5.0,
        payload_drag_area_m2: float = 0.05,
        reference_span_m: float = 28.0,
        reference_chord_m: float = 10.0,
    ) -> "GenericAerodynamics":
        scale = aspect_ratio / 2.7
        trim_alpha_rad = math.radians(trim_alpha_deg)
        cl_alpha = 1.9 + 0.22 * aspect_ratio
        cl_trim = 0.30 + 0.085 * aspect_ratio
        cl_0 = cl_trim - cl_alpha * trim_alpha_rad
        cd_alpha = 0.09 + 0.015 * scale
        cd_alpha2 = 1.2 + 0.18 * aspect_ratio
        cd_trim = cl_trim / target_lift_to_drag
        cd_0 = max(
            0.02,
            cd_trim - cd_alpha * abs(trim_alpha_rad) - cd_alpha2 * trim_alpha_rad**2,
        )
        return cls(
            cl_0=cl_0,
            cl_alpha=cl_alpha,
            cd_0=cd_0,
            cd_alpha=cd_alpha,
            cd_alpha2=cd_alpha2,
            cy_beta=-0.55 * scale,
            cm_0=0.02,
            cm_alpha=-0.45 * scale,
            cm_q=-0.75,
            cl_beta=-0.22 * scale,
            cl_p=-0.35 * scale,
            cl_r=-0.08 * scale,
            cl_da=-0.18 * scale,
            cn_beta=0.10 * scale,
            cn_p=0.05 * scale,
            cn_r=-0.16 * scale,
            cn_da=0.11 * scale,
            cl_ds=-0.05,
            cd_ds=0.18,
            payload_drag_area_m2=payload_drag_area_m2,
            max_deflection_rad=math.radians(30.0),
            trim_alpha_rad=trim_alpha_rad,
            target_lift_to_drag=target_lift_to_drag,
            reference_span_m=reference_span_m,
            reference_chord_m=reference_chord_m,
        )


@dataclass(frozen=True)
class EvaluatedAerodynamics:
    alpha_rad: float
    beta_rad: float
    cl: float
    cd: float
    cy: float
    cm: float
    cl_roll: float
    cn: float
    force_body_n: np.ndarray
    moment_body_nm: np.ndarray


def _alpha_beta_from_velocity(air_velocity_body_mps: np.ndarray) -> tuple[float, float]:
    vx, vy, vz = air_velocity_body_mps
    alpha = math.atan2(vz, max(vx, 1.0e-9))
    beta = math.atan2(vy, math.sqrt(vx * vx + vz * vz) + 1.0e-9)
    return alpha, beta


def evaluate_aerodynamics(
    air_velocity_body_mps: np.ndarray,
    density_kgpm3: float,
    surface_area_m2: float,
    aero: GenericAerodynamics,
    symmetric_deflection_rad: float = 0.0,
    asymmetric_deflection_rad: float = 0.0,
    body_rates_radps: np.ndarray | None = None,
) -> EvaluatedAerodynamics:
    velocity = np.asarray(air_velocity_body_mps, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed < 1.0e-9:
        zero = np.zeros(3, dtype=float)
        return EvaluatedAerodynamics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero, zero)

    p_rate, q_rate, r_rate = (0.0, 0.0, 0.0) if body_rates_radps is None else body_rates_radps
    alpha, beta = _alpha_beta_from_velocity(velocity)
    qbar = 0.5 * density_kgpm3 * speed**2
    b_ref = aero.reference_span_m
    c_ref = aero.reference_chord_m
    p_hat = b_ref * p_rate / (2.0 * speed)
    q_hat = c_ref * q_rate / (2.0 * speed)
    r_hat = b_ref * r_rate / (2.0 * speed)

    cl = aero.cl_0 + aero.cl_alpha * alpha + aero.cl_ds * symmetric_deflection_rad
    cd = (
        aero.cd_0
        + aero.cd_alpha * abs(alpha)
        + aero.cd_alpha2 * alpha * alpha
        + abs(aero.cd_ds * symmetric_deflection_rad)
    )
    cy = aero.cy_beta * beta
    cm = aero.cm_0 + aero.cm_alpha * alpha + aero.cm_q * q_hat
    cl_roll = (
        aero.cl_beta * beta
        + aero.cl_p * p_hat
        + aero.cl_r * r_hat
        + aero.cl_da * asymmetric_deflection_rad
    )
    cn = (
        aero.cn_beta * beta
        + aero.cn_p * p_hat
        + aero.cn_r * r_hat
        + aero.cn_da * asymmetric_deflection_rad
    )

    v_hat = unit(velocity)
    up_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    side_dir = np.cross(up_ref, v_hat)
    if np.linalg.norm(side_dir) < 1.0e-9:
        side_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    side_dir = unit(side_dir)
    lift_dir = unit(np.cross(v_hat, side_dir))
    force_body_n = qbar * surface_area_m2 * (-cd * v_hat + cy * side_dir + cl * lift_dir)
    moment_body_nm = qbar * surface_area_m2 * np.array(
        [aero.reference_span_m * cl_roll, aero.reference_chord_m * cm, aero.reference_span_m * cn],
        dtype=float,
    )
    return EvaluatedAerodynamics(
        alpha_rad=alpha,
        beta_rad=beta,
        cl=cl,
        cd=cd,
        cy=cy,
        cm=cm,
        cl_roll=cl_roll,
        cn=cn,
        force_body_n=force_body_n,
        moment_body_nm=moment_body_nm,
    )
