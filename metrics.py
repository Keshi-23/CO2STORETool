"""Accuracy metrics, interval metrics and post-hoc uncertainty calibration."""
import numpy as np
from scipy import stats

TARGETS = ['STTI', 'RTI', 'STI', 'MTI']


def accuracy(y, p):
    out = {}
    for i, n in enumerate(TARGETS):
        yt, yp = y[:, i], p[:, i]
        out[f'R2_{n}'] = float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        out[f'RMSE_{n}'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        out[f'MAE_{n}'] = float(np.mean(np.abs(yt - yp)))
    out['R2_mean'] = float(np.mean([out[f'R2_{n}'] for n in TARGETS]))
    out['RMSE_mean'] = float(np.mean([out[f'RMSE_{n}'] for n in TARGETS]))
    out['MAE_mean'] = float(np.mean(np.abs(y - p)))
    s = p.sum(1)
    out['closure_MAE'] = float(np.mean(np.abs(s - 1.0)))
    out['closure_viol_pct'] = float(np.mean(np.abs(s - 1.0) > 0.01) * 100)
    return out


def intervals(samples, level=0.90):
    """Empirical predictive interval from samples (S,N,4)."""
    a = (1 - level) / 2
    lo = np.quantile(samples, a, axis=0)
    hi = np.quantile(samples, 1 - a, axis=0)
    return lo, hi


def picp_mpiw(y, samples, level=0.90):
    lo, hi = intervals(samples, level)
    out = {}
    rng = y.max(0) - y.min(0)
    for i, n in enumerate(TARGETS):
        cov = float(np.mean((y[:, i] >= lo[:, i]) & (y[:, i] <= hi[:, i])))
        w = float(np.mean(hi[:, i] - lo[:, i]))
        out[f'PICP_{n}'] = cov
        out[f'MPIW_{n}'] = w
        out[f'NMPIW_{n}'] = w / max(rng[i], 1e-9)      # width normalised by range
    out['PICP_mean'] = float(np.mean([out[f'PICP_{n}'] for n in TARGETS]))
    out['MPIW_mean'] = float(np.mean([out[f'MPIW_{n}'] for n in TARGETS]))
    out['NMPIW_mean'] = float(np.mean([out[f'NMPIW_{n}'] for n in TARGETS]))
    out['level'] = level
    return out


def crps_ensemble(y, samples, max_s=60, max_n=6000, seed=0):
    """CRPS by the energy form, averaged over indices. Lower is better.
    Sub-sampled in both the ensemble and row dimension to bound memory."""
    rng = np.random.RandomState(seed)
    si = np.linspace(0, len(samples) - 1, min(max_s, len(samples))).astype(int)
    ni = (rng.choice(len(y), max_n, replace=False) if len(y) > max_n
          else np.arange(len(y)))
    S = samples[np.ix_(si, ni)]
    yy = y[ni]
    t1 = np.abs(S - yy[None]).mean(0)
    t2 = np.zeros_like(t1)
    for a in range(len(S)):                      # chunked pairwise term
        t2 += np.abs(S[a][None] - S).mean(0)
    t2 *= 0.5 / len(S)
    v = (t1 - t2).mean(0)
    return {**{f'CRPS_{n}': float(v[i]) for i, n in enumerate(TARGETS)},
            'CRPS_mean': float(v.mean())}


def calibration_curve(y, samples, n_levels=11):
    """Observed vs nominal coverage across central intervals."""
    nom = np.linspace(0.05, 0.95, n_levels)
    obs = np.zeros((n_levels, len(TARGETS)))
    for j, L in enumerate(nom):
        lo, hi = intervals(samples, L)
        for i in range(len(TARGETS)):
            obs[j, i] = np.mean((y[:, i] >= lo[:, i]) & (y[:, i] <= hi[:, i]))
    ece = np.abs(obs - nom[:, None]).mean(0)
    return dict(nominal=nom, observed=obs,
                ECE={n: float(ece[i]) for i, n in enumerate(TARGETS)},
                ECE_mean=float(ece.mean()))


def fit_variance_scaling(y_val, samples_val, grid=None):
    """Post-hoc calibration: inflate/deflate spread about the predictive mean by a
    single factor s per index, chosen on validation to minimise coverage error
    across nominal levels. Cheap, monotone, and does not change the mean."""
    grid = np.geomspace(0.25, 6.0, 40) if grid is None else grid
    mu = samples_val.mean(0)
    dev = samples_val - mu[None]
    best = np.ones(len(TARGETS))
    nom = np.linspace(0.05, 0.95, 11)
    for i in range(len(TARGETS)):
        err = []
        for s in grid:
            sm = mu[:, i][None] + dev[:, :, i] * s
            e = 0.0
            for L in nom:
                a = (1 - L) / 2
                lo = np.quantile(sm, a, axis=0); hi = np.quantile(sm, 1 - a, axis=0)
                e += abs(np.mean((y_val[:, i] >= lo) & (y_val[:, i] <= hi)) - L)
            err.append(e)
        best[i] = grid[int(np.argmin(err))]
    return best


def apply_variance_scaling(samples, s, renormalise=False):
    mu = samples.mean(0)
    out = mu[None] + (samples - mu[None]) * np.asarray(s)[None, None, :]
    if renormalise:                      # keep samples on the simplex
        out = np.clip(out, 1e-9, None)
        out = out / out.sum(2, keepdims=True)
    return out


def summarise(y, samples, level=0.90, prefix=''):
    p = samples.mean(0)
    out = {**accuracy(y, p), **picp_mpiw(y, samples, level), **crps_ensemble(y, samples)}
    cc = calibration_curve(y, samples)
    out['ECE_mean'] = cc['ECE_mean']
    out['sd_mean'] = float(samples.std(0).mean())
    return ({prefix + k: v for k, v in out.items()} if prefix else out), cc
