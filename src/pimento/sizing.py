from __future__ import annotations

from dataclasses import dataclass
import math

GRAVITY_MPS2 = 9.80665
LINE_AREA_PER_SUSPENSION_LINE_M2 = 1.11
DEFAULT_LINE_DIAMETER_M = 2.5e-3


@dataclass(frozen=True)
class SizingInputs:
    payload_mass_kg: float
    target_lift_to_drag: float = 4.0
    trim_speed_mps: float = 15.0
    trim_lift_coefficient: float = 0.55
    air_density_kgpm3: float = 1.225
    aspect_ratio: float = 2.7
    canopy_areal_density_kgpm2: float = 0.27
    line_length_to_span_ratio: float = 0.58
    harness_length_to_line_ratio: float = 0.50
    canopy_thickness_to_chord_ratio: float = 0.15
    payload_drag_area_m2: float = 0.0
    desired_surface_area_m2: float | None = None


@dataclass(frozen=True)
class GeometryEstimate:
    surface_area_m2: float
    aspect_ratio: float
    span_m: float
    chord_m: float
    canopy_thickness_m: float
    canopy_height_m: float
    line_length_m: float
    harness_length_m: float
    anhedral_angle_rad: float
    suspension_line_count: int
    suspension_line_diameter_m: float


@dataclass(frozen=True)
class SizingResult:
    payload_mass_kg: float
    estimated_canopy_mass_kg: float
    total_system_mass_kg: float
    trim_glide_angle_rad: float
    estimated_sink_rate_mps: float
    wing_loading_npm2: float
    dynamic_pressure_pa: float
    geometry: GeometryEstimate


def geometry_from_surface_area(
    surface_area_m2: float,
    aspect_ratio: float,
    line_length_to_span_ratio: float = 0.58,
    harness_length_to_line_ratio: float = 0.50,
    canopy_thickness_to_chord_ratio: float = 0.15,
) -> GeometryEstimate:
    span_m = math.sqrt(surface_area_m2 * aspect_ratio)
    chord_m = math.sqrt(surface_area_m2 / aspect_ratio)
    canopy_thickness_m = canopy_thickness_to_chord_ratio * chord_m
    canopy_height_m = span_m * canopy_thickness_to_chord_ratio
    line_length_m = line_length_to_span_ratio * span_m
    harness_length_m = harness_length_to_line_ratio * line_length_m
    anhedral_angle_rad = math.atan2(span_m / 2.0, max(line_length_m, 1.0e-6))
    suspension_line_count = max(1, math.ceil(surface_area_m2 / LINE_AREA_PER_SUSPENSION_LINE_M2))
    return GeometryEstimate(
        surface_area_m2=surface_area_m2,
        aspect_ratio=aspect_ratio,
        span_m=span_m,
        chord_m=chord_m,
        canopy_thickness_m=canopy_thickness_m,
        canopy_height_m=canopy_height_m,
        line_length_m=line_length_m,
        harness_length_m=harness_length_m,
        anhedral_angle_rad=anhedral_angle_rad,
        suspension_line_count=suspension_line_count,
        suspension_line_diameter_m=DEFAULT_LINE_DIAMETER_M,
    )


def estimate_surface_area(inputs: SizingInputs) -> float:
    if inputs.desired_surface_area_m2 is not None:
        return inputs.desired_surface_area_m2

    glide_angle = math.atan(1.0 / inputs.target_lift_to_drag)
    dynamic_pressure = 0.5 * inputs.air_density_kgpm3 * inputs.trim_speed_mps**2
    numerator = inputs.payload_mass_kg * GRAVITY_MPS2 * math.cos(glide_angle)
    denominator = (
        dynamic_pressure * inputs.trim_lift_coefficient
        - inputs.canopy_areal_density_kgpm2 * GRAVITY_MPS2 * math.cos(glide_angle)
    )
    if denominator <= 0.0:
        raise ValueError("Chosen trim condition is too weak to sustain the canopy.")
    return numerator / denominator


def estimate_parafoil(inputs: SizingInputs) -> SizingResult:
    surface_area_m2 = estimate_surface_area(inputs)
    geometry = geometry_from_surface_area(
        surface_area_m2=surface_area_m2,
        aspect_ratio=inputs.aspect_ratio,
        line_length_to_span_ratio=inputs.line_length_to_span_ratio,
        harness_length_to_line_ratio=inputs.harness_length_to_line_ratio,
        canopy_thickness_to_chord_ratio=inputs.canopy_thickness_to_chord_ratio,
    )
    canopy_mass_kg = geometry.surface_area_m2 * inputs.canopy_areal_density_kgpm2
    total_mass_kg = inputs.payload_mass_kg + canopy_mass_kg
    trim_glide_angle_rad = math.atan(1.0 / inputs.target_lift_to_drag)
    estimated_sink_rate_mps = inputs.trim_speed_mps * math.sin(trim_glide_angle_rad)
    wing_loading_npm2 = total_mass_kg * GRAVITY_MPS2 / geometry.surface_area_m2
    dynamic_pressure_pa = 0.5 * inputs.air_density_kgpm3 * inputs.trim_speed_mps**2
    return SizingResult(
        payload_mass_kg=inputs.payload_mass_kg,
        estimated_canopy_mass_kg=canopy_mass_kg,
        total_system_mass_kg=total_mass_kg,
        trim_glide_angle_rad=trim_glide_angle_rad,
        estimated_sink_rate_mps=estimated_sink_rate_mps,
        wing_loading_npm2=wing_loading_npm2,
        dynamic_pressure_pa=dynamic_pressure_pa,
        geometry=geometry,
    )
