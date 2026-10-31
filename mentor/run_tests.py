"""
Run all unit tests in tests/ directory using built-in unittest framework.
"""

import unittest
import sys
import os

# Ensure package path is added
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests.test_bbv import test_bbv_expectation_variance, test_bbv_ldp_ratio, test_bbv_epsilon_limits
from tests.test_glmf import test_glmf_e_step_responsibilities_sum_to_one, test_glmf_m_step_updates
from tests.test_metrics import test_rmse_computation, test_f_measure_computation


class TestBBV(unittest.TestCase):
    def test_expectation_variance(self):
        test_bbv_expectation_variance()

    def test_ldp_ratio(self):
        test_bbv_ldp_ratio()

    def test_epsilon_limits(self):
        test_bbv_epsilon_limits()


class TestGLMF(unittest.TestCase):
    def test_responsibilities(self):
        test_glmf_e_step_responsibilities_sum_to_one()

    def test_m_step(self):
        test_glmf_m_step_updates()


class TestMetrics(unittest.TestCase):
    def test_rmse(self):
        test_rmse_computation()

    def test_f_measure(self):
        test_f_measure_computation()


if __name__ == "__main__":
    print("=" * 60)
    print("        RUNNING MENTOR UNIT TEST SUITE")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
