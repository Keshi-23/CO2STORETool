"""Publication-quality figures for protocol comparison and uncertainty quantification."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .data import TARGETS, NICE
from .metrics import calibration_curve, intervals, accuracy

STYLE = {'font.size': 9, 'axes.linewidth': 0.8, 'figure.dpi': 160, 'axes.grid': True,
         'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'savefig.bbox': 'tight',
         'axes.spines.top': False, 'axes.spines.right': False}
C_RAND, C_GRP, C_TRUE = '#C2452D', '#1F5673', '#222222'
IDXC = {'STTI': '#1F5673', 'RTI': '#C2452D', 'STI': '#3F7D20', 'MTI': '#8A5FA8'}
UQC = {'deterministic': '#777777', 'mc_dropout': '#E08214',
       'deep_ensemble': '#1F5673', 'bayesian': '#8A5FA8'}


def use_style():
    plt.rcParams.update(STYLE)


# --------------------------------------------------------------------------- #
def fig_protocol_bars(res, path, title='Protocol comparison'):
    """res: DataFrame with columns protocol, and R2_<index> / R2_mean."""
    use_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    protos = list(res['protocol'])
    x = np.arange(len(TARGETS)); w = 0.8 / len(protos)
    for k, pr in enumerate(protos):
        r = res[res.protocol == pr].iloc[0]
        ax[0].bar(x + (k - (len(protos) - 1) / 2) * w,
                  [r[f'R2_{t}'] for t in TARGETS], w,
                  color=C_RAND if pr == 'random' else C_GRP,
                  label='Protocol 1 (random rows)' if pr == 'random'
                        else 'Protocol 2 (unseen cases)')
    ax[0].set_xticks(x); ax[0].set_xticklabels(TARGETS)
    ax[0].set_ylabel('$R^2$'); ax[0].set_ylim(min(0, res[[f'R2_{t}' for t in TARGETS]].values.min()) - 0.05, 1.02)
    ax[0].axhline(0, color='0.4', lw=0.8)
    ax[0].legend(fontsize=7.5); ax[0].set_title('Accuracy per index', fontsize=9.5)

    for k, pr in enumerate(protos):
        r = res[res.protocol == pr].iloc[0]
        ax[1].bar(x + (k - (len(protos) - 1) / 2) * w,
                  [r[f'RMSE_{t}'] for t in TARGETS], w,
                  color=C_RAND if pr == 'random' else C_GRP)
    ax[1].set_xticks(x); ax[1].set_xticklabels(TARGETS)
    ax[1].set_ylabel('RMSE'); ax[1].set_yscale('log')
    ax[1].set_title('Error per index', fontsize=9.5)
    fig.suptitle(title, y=1.03, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_parity(y_by_proto, p_by_proto, path, title='Parity'):
    use_style()
    keys = list(y_by_proto)
    fig, ax = plt.subplots(len(keys), 4, figsize=(11, 2.8 * len(keys)),
                           squeeze=False)
    rng = np.random.RandomState(0)
    for r, k in enumerate(keys):
        y, p = y_by_proto[k], p_by_proto[k]
        s = rng.choice(len(y), min(6000, len(y)), replace=False)
        m = accuracy(y, p)
        col = C_RAND if 'random' in k else C_GRP
        for c, t in enumerate(TARGETS):
            a = ax[r, c]
            lo = min(y[:, c].min(), p[:, c].min()); hi = max(y[:, c].max(), p[:, c].max())
            pad = 0.04 * (hi - lo)
            a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='0.45', lw=0.9)
            a.scatter(y[s, c], p[s, c], s=2.5, alpha=0.18, color=col, lw=0)
            a.text(0.04, 0.96, f'$R^2$={m[f"R2_{t}"]:.3f}\nRMSE={m[f"RMSE_{t}"]:.4f}',
                   transform=a.transAxes, va='top', fontsize=7.5,
                   bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.6))
            a.set_xlim(lo - pad, hi + pad); a.set_ylim(lo - pad, hi + pad)
            if r == 0:
                a.set_title(NICE[t], fontsize=9.5)
            if r == len(keys) - 1:
                a.set_xlabel('Simulated')
            if c == 0:
                a.set_ylabel(f'{k}\n\nPredicted', fontsize=8.5)
    fig.suptitle(title, y=1.0, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_calibration(cc_by_method, path, title='Reliability diagrams'):
    """cc_by_method: {method_name: calibration_curve dict}."""
    use_style()
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.1), sharey=True)
    for c, t in enumerate(TARGETS):
        a = ax[c]
        a.plot([0, 1], [0, 1], color='0.45', lw=1.0, ls='--', label='ideal')
        for name, cc in cc_by_method.items():
            a.plot(cc['nominal'], cc['observed'][:, c], 'o-', ms=3, lw=1.3,
                   color=UQC.get(name, None),
                   label=f"{name} (ECE {cc['ECE'][t]:.3f})")
        a.set_title(NICE[t], fontsize=9.5)
        a.set_xlabel('Nominal coverage')
        if c == 0:
            a.set_ylabel('Observed coverage')
        a.legend(fontsize=6.4, loc='upper left')
    fig.suptitle(title, y=1.04, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_picp_mpiw(table, path, title='Coverage vs. interval width'):
    """table: DataFrame with method, protocol, PICP_mean, NMPIW_mean, level."""
    use_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for pr, mk in [('random', 'o'), ('group', 's')]:
        sub = table[table.protocol == pr]
        for _, r in sub.iterrows():
            ax[0].scatter(r.NMPIW_mean, r.PICP_mean, s=55, marker=mk,
                          color=UQC.get(r.method, '#444'), zorder=3,
                          label=f'{r.method} / {pr}')
    lvl = float(table.level.iloc[0])
    ax[0].axhline(lvl, color='0.4', ls='--', lw=1)
    ax[0].text(ax[0].get_xlim()[1], lvl, f' nominal {lvl:.0%}', fontsize=7.5,
               va='bottom', ha='right', color='0.4')
    ax[0].set_xlabel('Normalised mean interval width (NMPIW)')
    ax[0].set_ylabel('PICP')
    ax[0].set_title('Sharpness vs. calibration', fontsize=9.5)
    ax[0].legend(fontsize=6.2, ncol=2)

    piv = table.pivot(index='method', columns='protocol', values='CRPS_mean')
    piv.plot.bar(ax=ax[1], color=[C_RAND, C_GRP], width=0.72, legend=True)
    ax[1].set_ylabel('CRPS (lower is better)'); ax[1].set_xlabel('')
    ax[1].tick_params(axis='x', rotation=15)
    ax[1].legend(['random rows', 'unseen cases'], fontsize=7.5)
    ax[1].set_title('Probabilistic accuracy', fontsize=9.5)
    fig.suptitle(title, y=1.03, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_uncertainty_bands(y, samples, t, g, cases, path, level=0.90,
                          title='Predictive intervals on unseen simulations'):
    use_style()
    lo, hi = intervals(samples, level)
    mu = samples.mean(0)
    n = len(cases)
    fig, ax = plt.subplots(n, 4, figsize=(11.5, 2.05 * n), sharex=True, squeeze=False)
    for i, c in enumerate(cases):
        m = g == c; o = np.argsort(t[m])
        tt = t[m][o]
        for j, k in enumerate(TARGETS):
            a = ax[i, j]
            a.fill_between(tt, lo[m][o, j], hi[m][o, j], color=IDXC[k], alpha=0.25,
                           lw=0, label=f'{level:.0%} predictive interval')
            a.plot(tt, mu[m][o, j], '--', color=IDXC[k], lw=1.4, label='Predictive mean')
            a.plot(tt, y[m][o, j], '-', color=C_TRUE, lw=1.5, label='Simulation (truth)')
            a.set_xscale('log')
            if i == 0:
                a.set_title(NICE[k], fontsize=9.5)
            if j == 0:
                a.set_ylabel(f'case {c}\nindex (-)', fontsize=8)
            if i == n - 1:
                a.set_xlabel('Time (years)')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.015),
               fontsize=8.5, frameon=False)
    fig.suptitle(title, y=1.005, fontsize=10.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(path); plt.close(fig)


def fig_uncertainty_vs_time(y, samples, t, path, decomposition=None,
                            title='Uncertainty and coverage vs. horizon'):
    use_style()
    bins = [0, 1, 5, 15, 30, 50, 100, 150, 200.1]
    lab = ['<1', '1-5', '5-15', '15-30', '30-50', '50-100', '100-150', '150-200']
    ctr = [0.5, 3, 10, 22, 40, 75, 125, 175]
    b = pd.cut(t, bins, labels=lab)
    lo, hi = intervals(samples, 0.90)
    sd = samples.std(0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, k in enumerate(TARGETS):
        cov, wid = [], []
        for L in lab:
            m = np.asarray(b == L)
            if m.sum() < 30:
                cov.append(np.nan); wid.append(np.nan); continue
            cov.append(np.mean((y[m, i] >= lo[m, i]) & (y[m, i] <= hi[m, i])))
            wid.append(np.mean(sd[m, i]))
        ax[0].plot(ctr, cov, 'o-', ms=3.5, lw=1.3, color=IDXC[k], label=NICE[k])
        ax[1].plot(ctr, wid, 'o-', ms=3.5, lw=1.3, color=IDXC[k])
    ax[0].axhline(0.90, color='0.4', ls='--', lw=1)
    ax[0].set_ylabel('Observed 90% coverage'); ax[0].set_ylim(0, 1.02)
    ax[0].set_title('Coverage vs. horizon', fontsize=9.5); ax[0].legend(fontsize=7)
    ax[1].set_ylabel('Predictive SD'); ax[1].set_title('Spread vs. horizon', fontsize=9.5)
    for a in ax[:2]:
        a.set_xscale('log'); a.set_xlabel('Time (years)')

    if decomposition is not None:
        al = decomposition['aleatoric'].mean(1); ep = decomposition['epistemic'].mean(1)
        A, E = [], []
        for L in lab:
            m = np.asarray(b == L)
            A.append(np.sqrt(al[m].mean()) if m.sum() > 30 else np.nan)
            E.append(np.sqrt(ep[m].mean()) if m.sum() > 30 else np.nan)
        ax[2].plot(ctr, A, 'o-', ms=3.5, lw=1.4, color='#E08214', label='Aleatoric')
        ax[2].plot(ctr, E, 's-', ms=3.5, lw=1.4, color='#1F5673', label='Epistemic')
        ax[2].legend(fontsize=7.5)
    ax[2].set_xscale('log'); ax[2].set_xlabel('Time (years)')
    ax[2].set_ylabel('SD contribution')
    ax[2].set_title('Variance decomposition', fontsize=9.5)
    fig.suptitle(title, y=1.03, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_risk(y, samples, path, thresholds=(0.05, 0.10, 0.20),
             index='RTI', title='Risk-informed screening'):
    """Exceedance probability curves: P(index > threshold) at the final time,
    the quantity a storage-permitting decision actually needs."""
    use_style()
    i = TARGETS.index(index)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
    grid = np.linspace(0, max(float(np.quantile(y[:, i], 0.99)), 1e-3), 60)
    prob = np.array([(samples[:, :, i] > th).mean(0) for th in grid])   # (G,N)
    ax[0].plot(grid, prob.mean(1), lw=1.8, color=IDXC[index])
    ax[0].fill_between(grid, np.quantile(prob, 0.1, axis=1),
                       np.quantile(prob, 0.9, axis=1), color=IDXC[index], alpha=0.2, lw=0)
    ax[0].set_xlabel(f'{index} threshold'); ax[0].set_ylabel(f'P({index} > threshold)')
    ax[0].set_title('Exceedance probability', fontsize=9.5)

    for th in thresholds:
        p_hat = (samples[:, :, i] > th).mean(0)
        obs = (y[:, i] > th).astype(float)
        edges = np.linspace(0, 1, 11)
        bi = np.clip(np.digitize(p_hat, edges) - 1, 0, 9)
        xs, ys = [], []
        for k in range(10):
            m = bi == k
            if m.sum() > 20:
                xs.append(p_hat[m].mean()); ys.append(obs[m].mean())
        ax[1].plot(xs, ys, 'o-', ms=4, lw=1.3, label=f'{index} > {th}')
    ax[1].plot([0, 1], [0, 1], '--', color='0.45', lw=1)
    ax[1].set_xlabel('Predicted probability'); ax[1].set_ylabel('Observed frequency')
    ax[1].set_title('Decision reliability', fontsize=9.5); ax[1].legend(fontsize=7.5)
    fig.suptitle(title, y=1.03, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_trajectories(splits, preds, path, n=5, seed=7,
                     title='Randomly drawn simulations'):
    """Ground truth vs prediction over the full horizon, both protocols overlaid.
    splits/preds are dicts keyed by protocol name."""
    use_style()
    g = splits['group']
    yg, pg, tg, gg = g['y_te'], preds['group'], g['t_te'], g['g_te']
    r = splits.get('random')
    rng = np.random.RandomState(seed)
    cases = rng.choice(np.unique(gg), n, replace=False)
    fig, ax = plt.subplots(n, 4, figsize=(11.5, 2.05 * n), sharex=True, squeeze=False)
    for i, c in enumerate(cases):
        m = gg == c; o = np.argsort(tg[m])
        for j, k in enumerate(TARGETS):
            a = ax[i, j]
            a.plot(tg[m][o], yg[m][o, j], '-', color=C_TRUE, lw=1.6,
                   label='Simulation (truth)')
            a.plot(tg[m][o], pg[m][o, j], '--', color=C_GRP, lw=1.5,
                   label='Protocol 2 (case unseen)')
            if r is not None:
                mr = r['g_te'] == c
                if mr.sum():
                    a.scatter(r['t_te'][mr], preds['random'][mr][:, j], s=11,
                              color=C_RAND, lw=0, alpha=0.85,
                              label='Protocol 1 (random rows)')
            a.set_xscale('log')
            if i == 0:
                a.set_title(NICE[k], fontsize=9.5)
            if j == 0:
                a.set_ylabel(f'case {c}\nindex (-)', fontsize=8)
            if i == n - 1:
                a.set_xlabel('Time (years)')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.015),
               fontsize=8.5, frameon=False)
    fig.suptitle(title, y=1.005, fontsize=10.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(path); plt.close(fig)
