from __future__ import annotations

import math

SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_TEMPERATURE_K = 288.15
TEMPERATURE_LAPSE_K_PER_M = 0.0065
GAS_CONSTANT_AIR = 287.05
GRAVITY_MPS2 = 9.80665


def isa_density(altitude_m: float) -> float:
    """Return ISA density for the troposphere."""
    altitude = max(0.0, altitude_m)
    temperature = SEA_LEVEL_TEMPERATURE_K - TEMPERATURE_LAPSE_K_PER_M * altitude
    temperature = max(150.0, temperature)
    exponent = GRAVITY_MPS2 / (GAS_CONSTANT_AIR * TEMPERATURE_LAPSE_K_PER_M)
    pressure = SEA_LEVEL_PRESSURE_PA * (
        temperature / SEA_LEVEL_TEMPERATURE_K
    ) ** exponent
    return pressure / (GAS_CONSTANT_AIR * temperature)
