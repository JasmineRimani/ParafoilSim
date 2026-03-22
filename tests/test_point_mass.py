from __future__ import annotations

import unittest

from pimento.point_mass import simulate_point_mass
from pimento.public_case import make_public_case


class PointMassTests(unittest.TestCase):
    def test_public_case_descends_to_ground(self) -> None:
        case = make_public_case()
        result = simulate_point_mass(case["plan"], case["point_mass_config"])
        self.assertGreater(result["time_s"][-1], 100.0)
        self.assertAlmostEqual(result["altitude_m"][-1], 0.0, places=6)
        self.assertIn(result["phase"][-1], {"touchdown", "timeout"})
        self.assertLess(result["altitude_m"][-1], result["altitude_m"][0])


if __name__ == "__main__":
    unittest.main()
