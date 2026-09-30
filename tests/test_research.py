from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from curve_risk.experiment import prices
from curve_risk.research import (
    hessian_exposure, hac_mean_test, holm_adjust, exception_diagnostics,
    shrink_covariance, factor_covariance, exact_shock_pnl,
)


class ResearchTests(unittest.TestCase):
    def test_price_hessian_against_central_differences(self):
        curve = np.full(30, .03)
        face, hedge_faces = .9, np.array([-.1, .2, -.3, .05, .01])
        analytic = hessian_exposure(curve, face, hedge_faces)
        def book(y):
            target, hedges = prices(y)
            return face*target+hedges @ hedge_faces
        epsilon = 1e-5
        for index in [1, 4, 9, 14, 19, 29]:
            direction = np.eye(30)[index]*epsilon
            numerical = (book(curve+direction)-2*book(curve)+book(curve-direction))/epsilon**2
            self.assertAlmostEqual(numerical, analytic[index], delta=5e-5)

    def test_zero_shock_and_face_normalization(self):
        calibration = np.full(30, .02)
        curves = np.tile(calibration, (5, 1))
        np.testing.assert_array_equal(exact_shock_pnl(curves, calibration, np.arange(5)*.1), np.zeros(4))

    def test_hac_lag_zero_and_bartlett_covariances(self):
        differences = np.array([1., 4., -2., 7., 0., -3.])
        residual = differences-differences.mean()
        zero = hac_mean_test(differences, 0)
        self.assertAlmostEqual(zero['hac_standard_error'], np.std(differences, ddof=0)/np.sqrt(6))
        cov = (residual @ residual+2*(2/3)*(residual[1:] @ residual[:-1])+2*(1/3)*(residual[2:] @ residual[:-2]))/6
        self.assertAlmostEqual(hac_mean_test(differences, 2)['hac_standard_error'], np.sqrt(cov/6))

    def test_holm_adjustment_is_order_independent(self):
        np.testing.assert_allclose(holm_adjust([.04, .001, .03]), [.06, .003, .06])

    def test_exception_likelihood_handles_zero_counts(self):
        result = exception_diagnostics(np.zeros(100, dtype=int))
        self.assertEqual(result['transition_counts'], [[99, 0], [0, 0]])
        self.assertAlmostEqual(result['exact_binomial_p'], stats.binomtest(0, 100, .01).pvalue)
        self.assertEqual(result['independence_lr'], 0.)
        self.assertTrue(result['independence_sparse'])
        with self.assertRaises(ValueError):
            exception_diagnostics([0, 2])

    def test_covariance_endpoints_and_factor_rank(self):
        shocks = np.random.default_rng(14).normal(size=(100, 30))
        empirical = np.cov(shocks, rowvar=False)
        np.testing.assert_allclose(shrink_covariance(shocks, 0), empirical)
        np.testing.assert_allclose(shrink_covariance(shocks, 1), np.diag(np.diag(empirical)))
        self.assertEqual(np.linalg.matrix_rank(factor_covariance(shocks, 3), tol=1e-10), 3)


if __name__ == '__main__':
    unittest.main()
