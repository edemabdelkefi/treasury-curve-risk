"""Frozen-training comparison of duration, PCA and regime-robust hedges."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results'
DATA = ROOT / 'data/gsw_nominal_vintage.csv'
MATURITIES = np.arange(1, 31, dtype=float)
HEDGE_TENORS = np.array([2, 5, 10, 20, 30], dtype=int)
TARGET_TENOR = 15
PROTOCOL = {
    'training_start': '1990-01-01', 'training_end': '2019-12-31',
    'test_start': '2020-01-01', 'test_end': '2025-12-31',
    'regimes': [['1990-01-01', '2006-12-31'], ['2007-01-01', '2012-12-31'], ['2013-01-01', '2019-12-31']],
    'target': {'maturity_years': 15, 'annual_coupon': .04, 'market_value_dollars': 1_000_000},
    'hedge_zero_coupon_maturities': HEDGE_TENORS.tolist(),
    'gross_hedge_market_value_cap': 1.5, 'diagonal_covariance_shrinkage': .05,
    'pca_factors': 3, 'bootstrap_seed': 20260930, 'bootstrap_draws': 2000,
    'bootstrap_block_days': 10, 'expected_shortfall_quantile': .99,
    'scope': 'constant-maturity yield-shock repricing; fixed faces calibrated at training end; carry, roll-down, funding and execution costs excluded',
}


def load_curves(path=DATA):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('Date,'))
    frame = pd.read_csv(path, skiprows=header, na_values=['NA', 'N/A'])
    names = [f'SVENY{i:02d}' for i in range(1, 31)]
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame[names] = frame[names].apply(pd.to_numeric, errors='coerce')
    selected = frame.loc[frame.Date.between(PROTOCOL['training_start'], PROTOCOL['test_end']), ['Date']+names]
    missing = int(selected[names].isna().any(axis=1).sum())
    selected = selected.dropna().sort_values('Date').reset_index(drop=True)
    assert not selected.Date.duplicated().any()
    yields = selected[names].to_numpy(dtype=float) / 100  # percent to decimal
    assert np.isfinite(yields).all()
    return selected.Date.to_numpy(), yields, missing


def prices(yields):
    discount = np.exp(-yields * MATURITIES)
    cashflows = np.full(TARGET_TENOR, PROTOCOL['target']['annual_coupon'])
    cashflows[-1] += 1
    target = discount[..., :TARGET_TENOR] @ cashflows
    hedges = discount[..., HEDGE_TENORS-1]
    return target, hedges


def sensitivities(calibration_yields):
    target_price, _ = prices(calibration_yields)
    cashflows = np.full(TARGET_TENOR, PROTOCOL['target']['annual_coupon'])
    cashflows[-1] += 1
    pv_weights = cashflows * np.exp(-calibration_yields[:TARGET_TENOR] * MATURITIES[:TARGET_TENOR]) / target_price
    target = np.zeros(30)
    target[:TARGET_TENOR] = -pv_weights * MATURITIES[:TARGET_TENOR]
    instruments = np.zeros((30, len(HEDGE_TENORS)))
    instruments[HEDGE_TENORS-1, np.arange(len(HEDGE_TENORS))] = -HEDGE_TENORS
    # Exposures convert a one-bp yield shock into bp of book-value P&L.
    return target, instruments


def covariance(shocks, shrink=True):
    empirical = np.cov(shocks, rowvar=False, ddof=1)
    if not shrink:
        return empirical
    alpha = PROTOCOL['diagonal_covariance_shrinkage']
    return (1-alpha) * empirical + alpha*np.diag(np.diag(empirical))


def solve_hedge(target, instruments, covariances, robust=False):
    """Convex quadratic hedging with exact parallel DV01 neutrality and L1 cap.

    Split positive/negative weights keep all constraints differentiable.
    The minimax formulation uses an epigraph for worst-regime variance.
    """
    count = instruments.shape[1]
    scale = max(float(target @ cov @ target) for cov in covariances)
    covs = np.array(covariances)/scale
    initial_weights = np.zeros(count)
    initial_weights[2] = -target.sum()/instruments[:, 2].sum()
    initial = np.r_[np.maximum(initial_weights, 0), np.maximum(-initial_weights, 0), 1.]
    neutral = np.r_[instruments.sum(axis=0), -instruments.sum(axis=0), 0][None, :]
    gross = np.r_[np.ones(2*count), 0][None, :]
    constraints = [LinearConstraint(neutral, -target.sum(), -target.sum()), LinearConstraint(gross, 0, PROTOCOL['gross_hedge_market_value_cap'])]

    def exposure(x):
        return target + instruments @ (x[:count]-x[count:2*count])

    def risk(x, cov):
        e = exposure(x)
        return float(e @ cov @ e)

    def risk_grad(x, cov):
        g = 2 * instruments.T @ cov @ exposure(x)
        return np.r_[g, -g, 0]

    if robust:
        def slack(x):
            return np.array([x[-1]-risk(x, cov) for cov in covs])
        def slack_grad(x):
            jac = np.array([-risk_grad(x, cov) for cov in covs])
            jac[:, -1] = 1
            return jac
        constraints.append(NonlinearConstraint(slack, 0, np.inf, jac=slack_grad))
        objective = lambda x: x[-1]
        gradient = lambda x: np.r_[np.zeros(2*count), 1.]
    else:
        objective = lambda x: risk(x, covs[0]) + 1e-10*np.sum(x[:2*count]**2)
        gradient = lambda x: risk_grad(x, covs[0]) + np.r_[2e-10*x[:2*count], 0]
    result = minimize(objective, initial, jac=gradient, method='SLSQP', bounds=Bounds(0, np.inf), constraints=constraints, options={'ftol': 1e-12, 'maxiter': 1500})
    if not result.success:
        raise RuntimeError(result.message)
    weights = result.x[:count]-result.x[count:2*count]
    e = target+instruments@weights
    assert abs(e.sum()) < 1e-7
    assert np.abs(weights).sum() <= PROTOCOL['gross_hedge_market_value_cap']+1e-7
    if robust:
        assert np.min(slack(result.x)) >= -1e-8
    return weights, {'iterations': result.nit, 'parallel_exposure_residual': float(e.sum()), 'gross_hedge_ratio': float(np.abs(weights).sum()), 'worst_regime_linear_variance_bp2': float(max(e@cov@e for cov in covariances)), 'normalized_epigraph_slack': float(np.min(slack(result.x))) if robust else None}


def risk_metrics(pnl):
    losses = -np.asarray(pnl)
    tail = max(1, len(losses)-int(np.floor(len(losses)*PROTOCOL['expected_shortfall_quantile'])))
    return {'observations': len(pnl), 'rms_bp': float(np.sqrt(np.mean(np.asarray(pnl)**2))), 'std_bp': float(np.std(pnl, ddof=1)), 'es99_bp': float(np.sort(losses)[-tail:].mean()), 'worst_loss_bp': float(losses.max()), 'mean_bp': float(np.mean(pnl)), 'es_tail_observations': tail}


def paired_bootstrap(baseline, candidate):
    n = len(baseline)
    block = PROTOCOL['bootstrap_block_days']
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    reductions = []
    for _ in range(PROTOCOL['bootstrap_draws']):
        starts = rng.integers(0, n, size=(n+block-1)//block)
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        ratio = np.sqrt(np.mean(candidate[indices]**2)/np.mean(baseline[indices]**2))
        reductions.append(1-ratio)
    return np.quantile(reductions, [.025, .975]).tolist()


def stress_scenarios():
    t = MATURITIES
    return {
        'parallel_up_100bp': np.full(30, 100.), 'parallel_down_100bp': np.full(30, -100.),
        'steepener_100bp': 100*(t-15)/15, 'flattener_100bp': -100*(t-15)/15,
        'belly_shock_100bp': 100*np.exp(-((t-10)/5)**2),
        'front_end_hike_200bp': 200*np.exp(-t/5),
        'bear_twist': 50 + 150*np.exp(-t/5),
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL, indent=2), encoding='utf-8')
    dates, yields, missing = load_curves()
    training = dates <= np.datetime64(PROTOCOL['training_end'])
    calibration_index = np.where(training)[0][-1]
    calibration = yields[calibration_index]
    shock_dates = dates[1:]
    shocks = np.diff(yields, axis=0)*10000
    train_mask = shock_dates <= np.datetime64(PROTOCOL['training_end'])
    test_mask = shock_dates >= np.datetime64(PROTOCOL['test_start'])
    assert not np.any(train_mask & test_mask)
    target, instruments = sensitivities(calibration)
    global_cov = covariance(shocks[train_mask])
    regime_covs = []
    regime_counts = []
    for start, end in PROTOCOL['regimes']:
        mask = (shock_dates >= np.datetime64(start)) & (shock_dates <= np.datetime64(end))
        assert np.all(shock_dates[mask] <= np.datetime64(PROTOCOL['training_end']))
        regime_covs.append(covariance(shocks[mask]))
        regime_counts.append(int(mask.sum()))
    empirical = covariance(shocks[train_mask], shrink=False)
    eigenvalues, eigenvectors = np.linalg.eigh(empirical)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    count = PROTOCOL['pca_factors']
    pca_cov = (eigenvectors[:, :count]*eigenvalues[:count]) @ eigenvectors[:, :count].T
    strategies = {'unhedged': np.zeros(5)}
    dv01 = np.zeros(5)
    dv01[2] = -target.sum()/instruments[:, 2].sum()
    strategies['dv01'] = dv01
    diagnostics = {}
    for name, covs, robust in [('pca3', [pca_cov], False), ('covariance', [global_cov], False), ('minimax', regime_covs, True)]:
        strategies[name], diagnostics[name] = solve_hedge(target, instruments, covs, robust)
    target_prices, hedge_prices = prices(yields)
    target_face = 1/target_prices[calibration_index]
    calibrated_hedge_prices = hedge_prices[calibration_index]
    pnl, rows, weights_rows = {}, [], []
    for name, weights in strategies.items():
        hedge_faces = weights/calibrated_hedge_prices
        book_prices = target_face*target_prices + hedge_prices @ hedge_faces
        change = np.diff(book_prices)*10000
        pnl[name] = change[test_mask]
        for d, previous, value in zip(shock_dates[test_mask], dates[:-1][test_mask], pnl[name]):
            rows.append({'date': str(d)[:10], 'previous_curve_date': str(previous)[:10], 'strategy': name, 'shock_pnl_bp': value})
        for t, w, face in zip(HEDGE_TENORS, weights, hedge_faces):
            weights_rows.append({'strategy': name, 'hedge_tenor': int(t), 'calibration_market_value_ratio': float(w), 'face_per_target_market_value': float(face)})
    pd.DataFrame(rows).to_csv(OUT/'holdout_pnl.csv', index=False)
    pd.DataFrame(weights_rows).to_csv(OUT/'hedge_weights.csv', index=False)
    risk = {name: risk_metrics(values) for name, values in pnl.items()}
    pd.DataFrame([{'strategy': name, **values} for name, values in risk.items()]).to_csv(OUT/'risk_metrics.csv', index=False)
    pd.DataFrame({'date': dates, **{f'zero_yield_{i:02d}': yields[:, i-1] for i in range(1, 31)}}).to_csv(OUT/'clean_curves.csv', index=False)
    stress_rows = []
    p0, h0 = prices(calibration)
    for scenario, shock in stress_scenarios().items():
        p1, h1 = prices(calibration+shock/10000)
        for name, weights in strategies.items():
            exact = ((p1-p0)/p0 + ((h1-h0)/h0) @ weights)*10000
            linear = (target+instruments@weights) @ shock
            stress_rows.append({'scenario': scenario, 'strategy': name, 'repriced_pnl_bp': float(exact), 'linear_pnl_bp': float(linear), 'nonlinear_residual_bp': float(exact-linear)})
    pd.DataFrame(stress_rows).to_csv(OUT/'stress.csv', index=False)
    reduction = 1-risk['minimax']['rms_bp']/risk['dv01']['rms_bp']
    comparison_cis = {name: paired_bootstrap(pnl['dv01'], pnl[name]) for name in ['pca3', 'covariance', 'minimax']}
    ci = comparison_cis['minimax']
    results = {
        'protocol': PROTOCOL, 'data': {'source': 'https://www.federalreserve.gov/data/nominal-yield-curve.htm', 'snapshot_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(), 'curve_rows': len(dates), 'training_shocks': int(train_mask.sum()), 'test_shocks': int(test_mask.sum()), 'maturities': 30, 'rows_excluded_for_missing_yields': missing, 'first_curve': str(dates[0])[:10], 'last_curve': str(dates[-1])[:10], 'calibration_date': str(dates[calibration_index])[:10], 'regime_shock_counts': regime_counts},
        'pca_three_factor_variance_share': float(eigenvalues[:3].sum()/eigenvalues.sum()),
        'risk': risk, 'optimizer': diagnostics,
        'minimax_rms_reduction_vs_dv01': float(reduction), 'reduction_ci95': ci,
        'minimax_es_reduction_vs_dv01': float(1-risk['minimax']['es99_bp']/risk['dv01']['es99_bp']),
        'minimax_rms_change_vs_pca3': float(risk['minimax']['rms_bp']/risk['pca3']['rms_bp']-1),
        'rms_reduction_vs_dv01': {name: float(1-risk[name]['rms_bp']/risk['dv01']['rms_bp']) for name in ['pca3', 'covariance', 'minimax']},
        'rms_reduction_ci95': comparison_cis,
        'stress_scenarios': len(stress_scenarios()),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__},
    }
    results['csv_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))}
    (OUT/'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    test_dates = shock_dates[test_mask]
    for name in ['dv01', 'pca3', 'minimax']:
        rolling = pd.Series(pnl[name], index=test_dates).rolling(63).std()
        ax[0, 0].plot(rolling.index, rolling, label=name, linewidth=1)
    ax[0, 0].set(title='Holdout: 63-session shock volatility', ylabel='bp of target calibration market value')
    ax[0, 0].legend()
    names = ['dv01', 'pca3', 'covariance', 'minimax']
    ax[0, 1].bar(np.arange(4)-.18, [risk[n]['rms_bp'] for n in names], .36, label='RMS')
    ax[0, 1].bar(np.arange(4)+.18, [risk[n]['es99_bp'] for n in names], .36, label='99% expected shortfall')
    ax[0, 1].set_xticks(np.arange(4), names)
    ax[0, 1].set(title='2020-2025 risk comparison', ylabel='bp'); ax[0, 1].legend()
    for name in ['dv01', 'pca3', 'minimax']:
        by_year = pd.Series(pnl[name], index=pd.to_datetime(test_dates)).groupby(lambda d: d.year).apply(lambda a: np.sqrt(np.mean(a*a)))
        ax[1, 0].plot(by_year.index, by_year, 'o-', label=name)
    ax[1, 0].set(title='Risk by calendar year', ylabel='RMS shock P&L, bp'); ax[1, 0].legend()
    stress_frame = pd.DataFrame(stress_rows)
    for name in ['dv01', 'pca3', 'minimax']:
        rows = stress_frame[stress_frame.strategy == name]
        ax[1, 1].plot(np.arange(len(rows)), rows.repriced_pnl_bp, 'o-', label=name)
    ax[1, 1].set_xticks(np.arange(7), ['+100', '-100', 'steep', 'flat', 'belly', 'front', 'twist'], rotation=30)
    ax[1, 1].set(title='Full nonlinear repricing under seven shocks', ylabel='Shock P&L, bp'); ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(OUT/'hedging_risk.png', dpi=180)
    fig.savefig(OUT/'hedging_risk.pdf')
    plt.close(fig)
    print(json.dumps({k: results[k] for k in ['data', 'risk', 'minimax_rms_reduction_vs_dv01', 'reduction_ci95', 'minimax_es_reduction_vs_dv01', 'minimax_rms_change_vs_pca3']}, indent=2))
    return results


if __name__ == '__main__':
    run()
