from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-inertial rotation for ZYX Euler angles."""
    cphi = math.cos(roll)
    sphi = math.sin(roll)
    ctheta = math.cos(pitch)
    stheta = math.sin(pitch)
    cpsi = math.cos(yaw)
    spsi = math.sin(yaw)
    return np.array(
        [
            [
                cpsi * ctheta,
                cpsi * stheta * sphi - spsi * cphi,
                cpsi * stheta * cphi + spsi * sphi,
            ],
            [
                spsi * ctheta,
                spsi * stheta * sphi + cpsi * cphi,
                spsi * stheta * cphi - cpsi * sphi,
            ],
            [-stheta, ctheta * sphi, ctheta * cphi],
        ],
        dtype=float,
    )


def euler_rates_matrix_zyx(roll: float, pitch: float) -> np.ndarray:
    cphi = math.cos(roll)
    sphi = math.sin(roll)
    ctheta = math.cos(pitch)
    ttheta = math.tan(pitch)
    safe_ctheta = ctheta if abs(ctheta) > 1.0e-6 else math.copysign(1.0e-6, ctheta or 1.0)
    return np.array(
        [
            [1.0, sphi * ttheta, cphi * ttheta],
            [0.0, cphi, -sphi],
            [0.0, sphi / safe_ctheta, cphi / safe_ctheta],
        ],
        dtype=float,
    )


def unit(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(value)
    if norm < 1.0e-9:
        return np.zeros_like(value)
    return value / norm


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
