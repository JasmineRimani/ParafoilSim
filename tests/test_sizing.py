from __future__ import annotations

import math
import unittest

from pimento.sizing import SizingInputs, estimate_parafoil


class SizingTests(unittest.TestCase):
    def test_paper_like_case_is_in_expected_band(self) -> None:
        result = estimate_parafoil(
            SizingInputs(
                payload_mass_kg=2028.0,
                target_lift_to_drag=4.0,
                trim_speed_mps=15.0,
                trim_lift_coefficient=0.55,
                aspect_ratio=2.7,
            )
        )
        self.assertGreater(result.geometry.surface_area_m2, 200.0)
        self.assertLess(result.geometry.surface_area_m2, 320.0)
        self.assertAlmostEqual(
            result.geometry.span_m * result.geometry.chord_m,
            result.geometry.surface_area_m2,
            places=6,
        )
        self.assertAlmostEqual(
            result.geometry.span_m / result.geometry.chord_m,
            result.geometry.aspect_ratio,
            places=6,
        )
        self.assertAlmostEqual(
            result.estimated_sink_rate_mps,
            result.geometry.surface_area_m2 * 0.0 + 15.0 * math.sin(math.atan(1.0 / 4.0)),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
