from __future__ import annotations

from collections.abc import Callable

import numpy as np


def rk4_step(
    derivative: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = derivative(t, y)
    k2 = derivative(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = derivative(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = derivative(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
