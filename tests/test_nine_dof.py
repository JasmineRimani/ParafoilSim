from __future__ import annotations

import unittest

import numpy as np

from pimento.nine_dof import simulate_nine_dof
from pimento.public_case import make_public_case


class NineDofTests(unittest.TestCase):
    def test_starter_model_stays_finite_for_short_rollout(self) -> None:
        case = make_public_case()
        result = simulate_nine_dof(
            initial_state=case["initial_nine_dof_state"],
            parameters=case["nine_dof_parameters"],
            duration_s=0.5,
            dt_s=0.02,
        )
        self.assertEqual(result["body_velocity_mps"].shape[1], 3)
        self.assertTrue(np.isfinite(result["body_velocity_mps"]).all())
        self.assertTrue(np.isfinite(result["canopy_euler_rad"]).all())


if __name__ == "__main__":
    unittest.main()
