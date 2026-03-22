from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .utils import clamp, cross2d, unit, wrap_angle


@dataclass(frozen=True)
class OrbitSegment:
    center_xy_m: np.ndarray
    radius_m: float
    exit_altitude_m: float


@dataclass(frozen=True)
class PointMassPlan:
    release_xy_m: np.ndarray
    release_altitude_m: float
    loiter: OrbitSegment
    energy_management: OrbitSegment
    touchdown_xy_m: np.ndarray
    runway_heading_rad: float = 0.0


@dataclass(frozen=True)
class GuidanceSettings:
    nominal_lift_to_drag: float = 4.0
    max_turn_rate_radps: float = math.radians(6.0)
    turn_radius_m: float = 110.0
    final_leg_m: float = 450.0
    transition_tolerance_m: float = 20.0
    heading_preview_time_s: float = 2.0
    flight_path_time_constant_s: float = 1.0
    flare_altitude_m: float = 8.0
    flare_flight_path_angle_rad: float = math.radians(-5.0)


@dataclass(frozen=True)
class GuidanceCommand:
    heading_rate_radps: float
    target_flight_path_angle_rad: float
    phase: str


class Phase(str, Enum):
    TO_LOITER = "to_loiter"
    ENTER_LOITER = "enter_loiter"
    LOITER = "loiter"
    TO_ENERGY = "to_energy"
    ENTER_ENERGY = "enter_energy"
    ENERGY = "energy"
    TO_FINAL_ENTRY = "to_final_entry"
    FINAL_TURN = "final_turn"
    FINAL_LEG = "final_leg"
    DONE = "done"


class TrajectoryGuidance:
    def __init__(self, plan: PointMassPlan, settings: GuidanceSettings):
        self.plan = plan
        self.settings = settings
        self.phase = Phase.TO_LOITER
        self._loiter_sign = 1.0
        self._energy_sign = 1.0
        self._final_sign = 1.0
        self._final_entry_xy = self.plan.touchdown_xy_m - self.settings.final_leg_m * self._runway_unit
        self._final_center_xy = self._final_entry_xy.copy()
        self._pre_turn_xy = self._final_entry_xy.copy()

    @property
    def nominal_flight_path_angle(self) -> float:
        return -math.atan(1.0 / self.settings.nominal_lift_to_drag)

    @property
    def _runway_unit(self) -> np.ndarray:
        return np.array(
            [math.cos(self.plan.runway_heading_rad), math.sin(self.plan.runway_heading_rad)],
            dtype=float,
        )

    @property
    def _runway_normal(self) -> np.ndarray:
        direction = self._runway_unit
        return np.array([-direction[1], direction[0]], dtype=float)

    def command(
        self,
        position_xy_m: np.ndarray,
        altitude_m: float,
        heading_rad: float,
        airspeed_mps: float,
        wind_xy_mps: np.ndarray,
        dt_s: float,
    ) -> GuidanceCommand:
        gamma = self.nominal_flight_path_angle
        if altitude_m <= self.settings.flare_altitude_m:
            gamma = self.settings.flare_flight_path_angle_rad

        if self.phase == Phase.TO_LOITER:
            if self._within_entry_gate(position_xy_m, self.plan.loiter.center_xy_m, self.plan.loiter.radius_m):
                self.phase = Phase.ENTER_LOITER
                self._loiter_sign = self._entry_sign(
                    position_xy_m, self.plan.loiter.center_xy_m, heading_rad
                )
            return GuidanceCommand(
                heading_rate_radps=self._line_heading_rate(
                    position_xy_m,
                    self.plan.loiter.center_xy_m,
                    heading_rad,
                    wind_xy_mps,
                    dt_s,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.ENTER_LOITER:
            if self._on_orbit(position_xy_m, self.plan.loiter.center_xy_m, self.plan.loiter.radius_m):
                self.phase = Phase.LOITER
            return GuidanceCommand(
                heading_rate_radps=self._orbit_rate(
                    airspeed_mps,
                    gamma,
                    self.settings.turn_radius_m,
                    self._loiter_sign,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.LOITER:
            if altitude_m <= self.plan.loiter.exit_altitude_m:
                self.phase = Phase.TO_ENERGY
            return GuidanceCommand(
                heading_rate_radps=self._orbit_rate(
                    airspeed_mps,
                    gamma,
                    self.plan.loiter.radius_m,
                    self._loiter_sign,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.TO_ENERGY:
            if self._within_entry_gate(
                position_xy_m,
                self.plan.energy_management.center_xy_m,
                self.plan.energy_management.radius_m,
            ):
                self.phase = Phase.ENTER_ENERGY
                self._energy_sign = self._entry_sign(
                    position_xy_m,
                    self.plan.energy_management.center_xy_m,
                    heading_rad,
                )
            return GuidanceCommand(
                heading_rate_radps=self._line_heading_rate(
                    position_xy_m,
                    self.plan.energy_management.center_xy_m,
                    heading_rad,
                    wind_xy_mps,
                    dt_s,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.ENTER_ENERGY:
            if self._on_orbit(
                position_xy_m,
                self.plan.energy_management.center_xy_m,
                self.plan.energy_management.radius_m,
            ):
                self.phase = Phase.ENERGY
            return GuidanceCommand(
                heading_rate_radps=self._orbit_rate(
                    airspeed_mps,
                    gamma,
                    self.settings.turn_radius_m,
                    self._energy_sign,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.ENERGY:
            if altitude_m <= self.plan.energy_management.exit_altitude_m:
                self.phase = Phase.TO_FINAL_ENTRY
                self._prepare_final_geometry(position_xy_m)
            return GuidanceCommand(
                heading_rate_radps=self._orbit_rate(
                    airspeed_mps,
                    gamma,
                    self.plan.energy_management.radius_m,
                    self._energy_sign,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.TO_FINAL_ENTRY:
            if np.linalg.norm(position_xy_m - self._pre_turn_xy) <= self.settings.transition_tolerance_m:
                self.phase = Phase.FINAL_TURN
            return GuidanceCommand(
                heading_rate_radps=self._line_heading_rate(
                    position_xy_m,
                    self._pre_turn_xy,
                    heading_rad,
                    wind_xy_mps,
                    dt_s,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.FINAL_TURN:
            if np.linalg.norm(position_xy_m - self._final_entry_xy) <= self.settings.transition_tolerance_m:
                self.phase = Phase.FINAL_LEG
            return GuidanceCommand(
                heading_rate_radps=self._orbit_rate(
                    airspeed_mps,
                    gamma,
                    self.settings.turn_radius_m,
                    self._final_sign,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        if self.phase == Phase.FINAL_LEG:
            if np.linalg.norm(position_xy_m - self.plan.touchdown_xy_m) <= self.settings.transition_tolerance_m:
                self.phase = Phase.DONE
            return GuidanceCommand(
                heading_rate_radps=self._line_heading_rate(
                    position_xy_m,
                    self.plan.touchdown_xy_m,
                    heading_rad,
                    wind_xy_mps,
                    dt_s,
                ),
                target_flight_path_angle_rad=gamma,
                phase=self.phase.value,
            )

        return GuidanceCommand(0.0, gamma, Phase.DONE.value)

    def _prepare_final_geometry(self, position_xy_m: np.ndarray) -> None:
        runway_normal = self._runway_normal
        runway_unit = self._runway_unit
        offset = position_xy_m - self._final_entry_xy
        self._final_sign = -1.0 if cross2d(runway_unit, offset) > 0.0 else 1.0
        self._final_center_xy = (
            self._final_entry_xy + self._final_sign * self.settings.turn_radius_m * runway_normal
        )
        self._pre_turn_xy = self._final_center_xy - self.settings.turn_radius_m * runway_unit

    def _line_heading_rate(
        self,
        position_xy_m: np.ndarray,
        target_xy_m: np.ndarray,
        heading_rad: float,
        wind_xy_mps: np.ndarray,
        dt_s: float,
    ) -> float:
        preview_target = target_xy_m + wind_xy_mps * self.settings.heading_preview_time_s
        delta = preview_target - position_xy_m
        desired_heading = math.atan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - heading_rad)
        requested_rate = heading_error / max(dt_s, 1.0e-3)
        return clamp(
            requested_rate,
            -self.settings.max_turn_rate_radps,
            self.settings.max_turn_rate_radps,
        )

    def _orbit_rate(self, airspeed_mps: float, gamma_rad: float, radius_m: float, sign: float) -> float:
        rate = sign * airspeed_mps * math.cos(gamma_rad) / max(radius_m, 1.0)
        return clamp(
            rate,
            -self.settings.max_turn_rate_radps,
            self.settings.max_turn_rate_radps,
        )

    def _entry_sign(self, position_xy_m: np.ndarray, center_xy_m: np.ndarray, heading_rad: float) -> float:
        target_heading = math.atan2(center_xy_m[1] - position_xy_m[1], center_xy_m[0] - position_xy_m[0])
        return 1.0 if wrap_angle(target_heading - heading_rad) >= 0.0 else -1.0

    def _within_entry_gate(self, position_xy_m: np.ndarray, center_xy_m: np.ndarray, radius_m: float) -> bool:
        entry_distance = self._entry_distance(radius_m)
        return np.linalg.norm(position_xy_m - center_xy_m) <= entry_distance

    def _on_orbit(self, position_xy_m: np.ndarray, center_xy_m: np.ndarray, radius_m: float) -> bool:
        distance = np.linalg.norm(position_xy_m - center_xy_m)
        return distance <= radius_m + self.settings.transition_tolerance_m

    def _entry_distance(self, orbit_radius_m: float) -> float:
        radius_sum = orbit_radius_m + self.settings.turn_radius_m
        alpha = math.asin(min(0.999, self.settings.turn_radius_m / max(radius_sum, 1.0)))
        return radius_sum * math.cos(alpha)
