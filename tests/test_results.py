from pathlib import Path
import hashlib
import json
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/research'


class ResultTests(unittest.TestCase):
    def test_frozen_artifact_checksums(self):
        metrics = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        for name, digest in metrics['csv_sha256'].items():
            self.assertEqual(hashlib.sha256((OUT/name).read_bytes()).hexdigest(), digest, name)
        for name, digest in metrics['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), digest, name)

    def test_chronology_and_risk_summary_reconstructed(self):
        daily = pd.read_csv(OUT/'daily_risk.csv')
        self.assertTrue((pd.to_datetime(daily.last_fit_date) < pd.to_datetime(daily.date)).all())
        selections = pd.read_csv(OUT/'model_selection.csv')
        self.assertTrue((pd.to_datetime(selections.fit_last_date) < pd.to_datetime(selections.validation_first_date)).all())
        for row in pd.read_csv(OUT/'risk_summary.csv').itertuples():
            subset = daily[daily.strategy == row.strategy]
            self.assertEqual(len(subset), row.observations)
            self.assertAlmostEqual(np.sqrt(np.mean(subset.shock_pnl_bp**2)), row.rms_bp, places=10)
            self.assertAlmostEqual(subset.implementation_cost_bp.sum(), row.total_implementation_cost_bp, places=10)
            np.testing.assert_allclose(subset.factor3_linear_pnl_bp+subset.orthogonal_linear_pnl_bp,
                                       subset.linear_pnl_bp, atol=1e-10)
