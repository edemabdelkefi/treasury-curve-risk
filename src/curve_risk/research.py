"""Chronological hedge selection, conditional tail risk and loss inference."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import xlogy

from .experiment import (
    ROOT, DATA, HEDGE_TENORS, MATURITIES, load_curves, prices,
    sensitivities, solve_hedge, risk_metrics,
)

PROTOCOL = {
    'seed': 20261001, 'test_years': list(range(2020, 2026)),
    'lookback_sessions': 1260, 'validation_sessions': 252,
    'shrinkage_grid': [0., .01, .05, .15, .35, .65, 1.],
    'factor_grid': [1, 2, 3, 4, 5], 'regime_blocks': 3,
    'ewma_decay': .97, 'student_degrees_of_freedom': 5,
    'tail_probability': .01, 'hac_lag': 10,
    'bootstrap_draws': 2000, 'bootstrap_blocks': [5, 10, 20],
    'one_way_cost_bp': 2., 'gross_hedge_cap': 1.5,
    'scope': 'annual calibration, fixed faces within each year, constant-maturity shock repricing',
    'selection': 'one chronological validation block inside each trailing training window; exact repricing MSE',
    'forecast': 'pre-shock EWMA covariance; Gaussian and variance-matched Student t innovations',
    'cost_scope': 'hedge implementation only; annual target rescaling, carry, roll-down and funding excluded',
}


def shrink_covariance(shocks, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('Shrinkage must lie in [0, 1].')
    empirical = np.cov(shocks, rowvar=False, ddof=1)
    return (1-alpha)*empirical + alpha*np.diag(np.diag(empirical))


def factor_covariance(shocks, factors):
    empirical = np.cov(shocks, rowvar=False, ddof=1)
    values, vectors = np.linalg.eigh(empirical)
    vectors, values = vectors[:, -factors:], values[-factors:]
    return (vectors*values) @ vectors.T


def exact_shock_pnl(curves, calibration, weights):
    target, hedges = prices(curves)
    p0, h0 = prices(calibration)
    return np.diff(target/p0 + hedges @ (weights/h0))*10000


def select_models(history_curves, history_shocks, validation_sessions=252):
    """Validation calibration predates every validation shock and price."""
    split = len(history_shocks)-validation_sessions
    if split < 60:
        raise ValueError('At least 60 fit shocks are required before validation.')
    calibration = history_curves[split]
    target, instruments = sensitivities(calibration)
    fit = history_shocks[:split]
    validation = history_curves[split:]
    records, chosen = [], {}
    for family, grid in [('shrinkage', PROTOCOL['shrinkage_grid']), ('pca', PROTOCOL['factor_grid'])]:
        candidates = []
        for parameter in grid:
            cov = shrink_covariance(fit, parameter) if family == 'shrinkage' else factor_covariance(fit, parameter)
            weights, _ = solve_hedge(target, instruments, [cov])
            error = exact_shock_pnl(validation, calibration, weights)
            score = float(np.mean(error**2))
            records.append({'family': family, 'parameter': parameter, 'validation_mse_bp2': score})
            candidates.append((score, parameter))
        chosen[family] = min(candidates)[1]
    return chosen, records, split


def hessian_exposure(curve, target_face, hedge_faces):
    """Diagonal price Hessian for independently quoted annual zero rates."""
    cashflows = np.full(15, .04)
    cashflows[-1] += 1
    diagonal = np.zeros(30)
    diagonal[:15] = target_face*cashflows*np.exp(-curve[:15]*MATURITIES[:15])*MATURITIES[:15]**2
    diagonal[HEDGE_TENORS-1] += hedge_faces*np.exp(-curve[HEDGE_TENORS-1]*HEDGE_TENORS)*HEDGE_TENORS**2
    return diagonal


def hac_mean_test(differences, lag=10):
    """Two-sided Newey-West test of zero mean paired loss difference."""
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n < 3 or lag < 0 or lag >= n or not np.isfinite(differences).all():
        raise ValueError('Invalid loss sample or HAC lag.')
    residuals = differences-differences.mean()
    long_run = float(residuals @ residuals/n)
    for k in range(1, lag+1):
        long_run += 2*(1-k/(lag+1))*float(residuals[k:] @ residuals[:-k]/n)
    se = np.sqrt(max(0., long_run)/n)
    mean = float(differences.mean())
    statistic = mean/se if se > 0 else (0. if mean == 0 else np.copysign(np.inf, mean))
    return {'mean_difference_bp2': mean, 'hac_standard_error': float(se),
            'z_statistic': float(statistic), 'two_sided_p': float(2*stats.norm.sf(abs(statistic)))}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate((len(pvalues)-np.arange(len(pvalues)))*pvalues[order]))
    return adjusted


def exception_diagnostics(exceptions, tail_probability=.01):
    exceptions = np.asarray(exceptions, dtype=int)
    if exceptions.ndim != 1 or len(exceptions) < 2 or not np.isin(exceptions, [0, 1]).all():
        raise ValueError('Exceptions must be a binary vector of length at least two.')
    transitions = np.bincount(2*exceptions[:-1]+exceptions[1:], minlength=4).reshape(2, 2)
    total = transitions.sum(axis=0)
    pooled = total[1]/total.sum()
    unrestricted = 0.
    for row in transitions:
        probability = row[1]/row.sum() if row.sum() else 0.
        unrestricted += float(xlogy(row[1], probability)+xlogy(row[0], 1-probability))
    restricted = float(xlogy(total[1], pooled)+xlogy(total[0], 1-pooled))
    lr = max(0., 2*(unrestricted-restricted))
    return {'exceptions': int(exceptions.sum()), 'observations': len(exceptions),
            'exception_rate': float(exceptions.mean()),
            'exact_binomial_p': float(stats.binomtest(int(exceptions.sum()), len(exceptions), tail_probability).pvalue),
            'independence_lr': lr, 'independence_asymptotic_p': float(stats.chi2.sf(lr, 1)),
            'transition_counts': transitions.tolist(),
            'independence_sparse': bool(np.any(transitions < 5))}


def block_risk_intervals(baseline, candidate, rng, block, draws):
    n = len(baseline)
    reductions = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n, size=(n+block-1)//block)
        indices = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        reductions[i] = 1-np.sqrt(np.mean(candidate[indices]**2)/np.mean(baseline[indices]**2))
    return np.quantile(reductions, [.025, .975]).tolist()


def run(output=None, quick=False):
    out = Path(output) if output else ROOT/'results/research'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    if quick:
        protocol.update(test_years=[2020], bootstrap_draws=100)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    dates, curves, missing = load_curves()
    shock_dates = dates[1:]
    shocks = np.diff(curves, axis=0)*10000
    rows, selection_rows, weight_rows, diagnostics = [], [], [], []
    previous_faces = {}
    last_exposure = {}
    for year in protocol['test_years']:
        start = int(np.searchsorted(shock_dates, np.datetime64(f'{year}-01-01')))
        end = int(np.searchsorted(shock_dates, np.datetime64(f'{year+1}-01-01')))
        left = max(0, start-protocol['lookback_sessions'])
        training = shocks[left:start]
        chosen, scores, split = select_models(curves[left:start+1], training, protocol['validation_sessions'])
        for score in scores:
            selection_rows.append({'year': year, 'fit_last_date': str(shock_dates[left+split-1])[:10],
                                   'validation_first_date': str(shock_dates[left+split])[:10],
                                   'validation_last_date': str(shock_dates[start-1])[:10], **score})
        calibration = curves[start]
        target, instruments = sensitivities(calibration)
        weights = {'dv01': np.zeros(5)}
        weights['dv01'][2] = -target.sum()/instruments[:, 2].sum()
        cov = shrink_covariance(training, chosen['shrinkage'])
        regime_covs = [shrink_covariance(block, chosen['shrinkage']) for block in np.array_split(training, 3)]
        for name, covs, robust in [
            ('pca_cv', [factor_covariance(training, chosen['pca'])], False),
            ('shrinkage_cv', [cov], False), ('minimax_cv', regime_covs, True),
        ]:
            weights[name], diag = solve_hedge(target, instruments, covs, robust)
            diagnostics.append({'year': year, 'strategy': name, **diag})
        p0, h0 = prices(calibration)
        face = 1/p0
        faces = {name: value/h0 for name, value in weights.items()}
        costs = {name: protocol['one_way_cost_bp']*float(np.abs(value-previous_faces.get(name, np.zeros(5))) @ h0)
                 for name, value in faces.items()}
        for name, value in weights.items():
            for tenor, w, f in zip(HEDGE_TENORS, value, faces[name]):
                weight_rows.append({'year': year, 'calibration_date': str(dates[start])[:10],
                                    'strategy': name, 'tenor': int(tenor), 'weight': w, 'face': f,
                                    'selected_alpha': chosen['shrinkage'], 'selected_factors': chosen['pca']})
        ewma = np.cov(training, rowvar=False)
        for shock in training:
            ewma = protocol['ewma_decay']*ewma+(1-protocol['ewma_decay'])*np.outer(shock, shock)
        for index in range(start, end):
            before, after = curves[index], curves[index+1]
            pb, hb = prices(before)
            pa, ha = prices(after)
            dynamic_target, dynamic_instruments = sensitivities(before)
            shock = shocks[index]
            eigenvalues, eigenvectors = np.linalg.eigh(ewma)
            factors = eigenvectors[:, -3:]
            factor_shock = factors @ (factors.T @ shock)
            for name, hedge_faces in faces.items():
                exposure = dynamic_target*(pb*face)+dynamic_instruments @ (hedge_faces*hb)
                linear = float(exposure @ shock)
                second = float(.5*hessian_exposure(before, face, hedge_faces) @ (shock**2)/10000)
                pnl = float(((pa-pb)*face+(ha-hb) @ hedge_faces)*10000)
                sigma = np.sqrt(max(float(exposure @ ewma @ exposure), 1e-14))
                gaussian_var = stats.norm.isf(protocol['tail_probability'])*sigma
                nu = protocol['student_degrees_of_freedom']
                student_var = stats.t.isf(protocol['tail_probability'], nu)*np.sqrt((nu-2)/nu)*sigma
                cost = costs[name] if index == start else 0.
                rows.append({'date': str(shock_dates[index])[:10], 'previous_date': str(dates[index])[:10],
                             'year': year, 'strategy': name, 'shock_pnl_bp': pnl,
                             'implementation_cost_bp': cost, 'pnl_after_implementation_bp': pnl-cost,
                             'linear_pnl_bp': linear, 'quadratic_pnl_bp': linear+second,
                             'factor3_linear_pnl_bp': float(exposure @ factor_shock),
                             'orthogonal_linear_pnl_bp': float(exposure @ (shock-factor_shock)),
                             'pre_shock_sigma_bp': sigma, 'gaussian_var99_bp': gaussian_var,
                             'student_var99_bp': student_var,
                             'gaussian_exception': int(-pnl > gaussian_var),
                             'student_exception': int(-pnl > student_var),
                             'exposure_change_l2': float(np.linalg.norm(exposure-last_exposure.get(name, exposure))),
                             'last_fit_date': str(dates[start])[:10]})
                last_exposure[name] = exposure
            # The current shock enters only tomorrow's forecast.
            ewma = protocol['ewma_decay']*ewma+(1-protocol['ewma_decay'])*np.outer(shock, shock)
        previous_faces = faces
        print(f'Walk-forward {year}: alpha={chosen["shrinkage"]}, factors={chosen["pca"]}, shocks={end-start}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'daily_risk.csv', index=False, lineterminator='\n')
    pd.DataFrame(selection_rows).to_csv(out/'model_selection.csv', index=False, lineterminator='\n')
    pd.DataFrame(weight_rows).to_csv(out/'hedge_weights.csv', index=False, lineterminator='\n')
    pivot = frame.pivot(index='date', columns='strategy', values='shock_pnl_bp')
    baseline = pivot.dv01.to_numpy()
    summaries, tests, tail = [], [], {}
    streams = iter(np.random.SeedSequence(protocol['seed']).spawn(9))
    intervals = []
    for name in pivot.columns:
        subset = frame[frame.strategy == name]
        values = pivot[name].to_numpy()
        summary = {'strategy': name, **risk_metrics(values),
                   'annualized_shock_vol_bp': float(np.std(values, ddof=1)*np.sqrt(252)),
                   'total_implementation_cost_bp': float(subset.implementation_cost_bp.sum()),
                   'linear_rmse_bp': float(np.sqrt(np.mean((subset.shock_pnl_bp-subset.linear_pnl_bp)**2))),
                   'quadratic_rmse_bp': float(np.sqrt(np.mean((subset.shock_pnl_bp-subset.quadratic_pnl_bp)**2)))}
        summaries.append(summary)
        tail[name] = {kind: exception_diagnostics(subset[f'{kind}_exception']) for kind in ['gaussian', 'student']}
        if name != 'dv01':
            tests.append({'strategy': name, **hac_mean_test(values**2-baseline**2, protocol['hac_lag'])})
            for block in protocol['bootstrap_blocks']:
                interval = block_risk_intervals(baseline, values, np.random.default_rng(next(streams)), block, protocol['bootstrap_draws'])
                intervals.append({'strategy': name, 'block_sessions': block,
                                  'rms_reduction': float(1-np.sqrt(np.mean(values**2)/np.mean(baseline**2))),
                                  'ci95_low': interval[0], 'ci95_high': interval[1]})
    for test, adjusted in zip(tests, holm_adjust([test['two_sided_p'] for test in tests])):
        test['holm_p'] = float(adjusted)
    pd.DataFrame(summaries).to_csv(out/'risk_summary.csv', index=False, lineterminator='\n')
    pd.DataFrame(tests).to_csv(out/'paired_loss_tests.csv', index=False, lineterminator='\n')
    pd.DataFrame(intervals).to_csv(out/'bootstrap_intervals.csv', index=False, lineterminator='\n')
    annual = []
    for (year, name), subset in frame.groupby(['year', 'strategy']):
        annual.append({'year': int(year), 'strategy': name, **risk_metrics(subset.shock_pnl_bp)})
    pd.DataFrame(annual).to_csv(out/'annual_risk.csv', index=False, lineterminator='\n')
    metrics = {'protocol': protocol, 'source_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
               'excluded_source_rows': missing, 'risk': summaries, 'paired_tests': tests,
               'tail_backtests': tail, 'optimization': diagnostics,
               'last_fit_precedes_shock': bool((pd.to_datetime(frame.last_fit_date) < pd.to_datetime(frame.date)).all()),
               'code_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((ROOT/'src/curve_risk').glob('*.py'))},
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__},
               'csv_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name in pivot:
        dates_plot = pd.to_datetime(pivot.index)
        axes[0, 0].plot(dates_plot, pd.Series(pivot[name].to_numpy()).rolling(63).std(), label=name, lw=1)
        subset = pd.DataFrame(annual).query('strategy == @name')
        axes[0, 1].plot(subset.year, subset.rms_bp, 'o-', label=name)
    axes[0, 0].set(title='Chronological holdout: 63-session volatility', ylabel='Shock P&L, bp')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Annual recalibration', ylabel='RMS shock P&L, bp')
    choices = pd.DataFrame(weight_rows).groupby('year').first()
    axes[1, 0].plot(choices.index, choices.selected_alpha, 'o-', label='Diagonal shrinkage')
    axes[1, 0].set(title='Validation-selected regularization', ylabel='Shrinkage intensity', ylim=(-.05, 1.05))
    names = list(pivot.columns)
    axes[1, 1].bar(np.arange(len(names))-.18, [tail[n]['gaussian']['exception_rate'] for n in names], .36, label='Gaussian')
    axes[1, 1].bar(np.arange(len(names))+.18, [tail[n]['student']['exception_rate'] for n in names], .36, label='Student t, df=5')
    axes[1, 1].axhline(.01, color='black', ls='--', lw=1)
    axes[1, 1].set_xticks(np.arange(len(names)), names, rotation=15)
    axes[1, 1].set(title='Pre-shock 99% VaR exception rates', ylabel='Exception fraction')
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'walk_forward.png', dpi=180)
    fig.savefig(out/'walk_forward.pdf')
    plt.close(fig)
    print(json.dumps({'risk': summaries, 'chronology_verified': metrics['last_fit_precedes_shock']}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run(args.output, args.quick)


if __name__ == '__main__':
    main()
