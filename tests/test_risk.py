from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from curve_risk.experiment import prices, sensitivities, solve_hedge, risk_metrics, MATURITIES, HEDGE_TENORS


class RiskTests(unittest.TestCase):
    def test_cashflow_prices_on_zero_curve(self):
        target, hedge = prices(np.zeros(30))
        self.assertAlmostEqual(target, 1+.04*15)
        np.testing.assert_allclose(hedge, np.ones(5))

    def test_sensitivities_match_repricing(self):
        y = np.full(30, .035)
        target, instruments = sensitivities(y)
        p0, h0 = prices(y)
        epsilon = 1e-7
        for index in [1, 4, 9, 14, 29]:
            shifted = y.copy(); shifted[index] += epsilon
            p1, h1 = prices(shifted)
            self.assertAlmostEqual((p1-p0)/p0/epsilon, target[index], places=4)
            np.testing.assert_allclose((h1-h0)/h0/epsilon, instruments[index], atol=5e-5)

    def test_neutrality_leverage_and_worst_regime_objective(self):
        target, instruments = sensitivities(np.full(30, .03))
        t = MATURITIES/30
        covs = [np.diag(np.full(30, .1))+np.outer(v, v) for v in [np.ones(30), t, t*t]]
        weights, diagnostics = solve_hedge(target, instruments, covs, robust=True)
        exposure = target+instruments@weights
        self.assertLess(abs(exposure.sum()), 1e-7)
        self.assertLessEqual(np.abs(weights).sum(), 1.5+1e-7)
        simple = np.zeros(5); simple[2] = -target.sum()/instruments[:, 2].sum()
        simple_exposure = target+instruments@simple
        simple_risk = max(simple_exposure@cov@simple_exposure for cov in covs)
        self.assertLessEqual(diagnostics['worst_regime_linear_variance_bp2'], simple_risk+1e-7)

    def test_expected_shortfall_uses_exact_tail_count(self):
        values = -np.arange(1500, dtype=float)
        metrics = risk_metrics(values)
        self.assertEqual(metrics['es_tail_observations'], 15)
        self.assertEqual(metrics['es99_bp'], np.arange(1485, 1500).mean())


if __name__ == '__main__':
    unittest.main()
