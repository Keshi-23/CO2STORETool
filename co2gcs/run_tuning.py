"""Resumable Optuna hyperparameter search.

The study is stored in SQLite, so the command can be re-run to add trials.

Examples
--------
python run_tuning.py --arch closure  --protocol group --n-trials 15
python run_tuning.py --arch baseline --protocol group --n-trials 10
python run_tuning.py --arch closure  --protocol group --report      # no new trials
"""
import argparse, json, os
import numpy as np, pandas as pd
import co2gcs as C
from co2gcs import tune as T


def tv_check(model, d, arch):
    """Total variation of the predicted trajectories vs. the simulations - the
    smoothness guard. Pointwise R2 is nearly blind to temporal oscillation."""
    from co2gcs.train import predict_mean
    P = predict_mean(model, d['Xb_te'], d['Xt_te'])
    y, g, t = d['y_te'], d['groups_te'], d['t_te']
    pv, yv = [], []
    for c in np.unique(g):
        m = g == c
        o = np.argsort(t[m])
        pv.append(np.abs(np.diff(P[m][o], axis=0)).sum(0))
        yv.append(np.abs(np.diff(y[m][o], axis=0)).sum(0))
    return np.median(pv, 0), np.median(yv, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='closure', choices=['closure', 'baseline'])
    ap.add_argument('--protocol', default='group', choices=['group', 'random'])
    ap.add_argument('--mode', default='fixed', choices=['fixed', 'orig'])
    ap.add_argument('--n-trials', type=int, default=10)
    ap.add_argument('--search-epochs', type=int, default=45,
                    help='short budget during search; the best config is '
                         'retrained at the full budget afterwards')
    ap.add_argument('--timeout', type=float, default=None)
    ap.add_argument('--space', default='basic', choices=['basic', 'extended'],
                    help="'basic' = the 7 core params; 'extended' adds dropout, "
                         "KL weight, n_freq, epochs, patience, variance weighting "
                         "and the physics-loss weight")
    ap.add_argument('--report', action='store_true',
                    help='summarise the existing study without running trials')
    ap.add_argument('--no-final', action='store_true')
    ap.add_argument('--csv', default='All_dataset.csv')
    ap.add_argument('--outdir', default='tuning')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    study, d = T.run_study(a.arch, a.protocol, mode=a.mode,
                           n_trials=0 if a.report else a.n_trials,
                           search_epochs=a.search_epochs, timeout=a.timeout,
                           csv=a.csv, space=a.space)

    done = [t for t in study.trials if t.value is not None]
    pruned = [t for t in study.trials
              if str(t.state) == 'TrialState.PRUNED']
    print(f'\nstudy {a.arch}/{a.protocol}/{a.mode}: {len(study.trials)} trials '
          f'({len(done)} complete, {len(pruned)} pruned)')
    print(f'space={a.space} -> {T.FIXED_NOTE[a.space][a.arch]}')
    if a.space == 'extended':
        print(f'note: {T.EXTENDED_NOTE}')
    if not done:
        print('no completed trials yet'); return

    print(f'\nbest validation R2_mean = {study.best_value:.4f}')
    print('best hyperparameters:')
    for k, v in study.best_params.items():
        print(f'  {k:12s} {v}')

    imp = T.importance(study)
    if imp:
        print('\nparameter importance (fANOVA over completed trials):')
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            print(f'  {k:12s} {v:.3f}')

    hist = pd.DataFrame([{**t.params, 'value': t.value, 'state': str(t.state)}
                         for t in study.trials])
    hist.to_csv(f'{a.outdir}/trials_{a.arch}_{a.protocol}_{a.mode}_{a.space}.csv', index=False)

    if a.no_final:
        return

    print('\nretraining at full epoch budget and scoring on the untouched test '
          'cases...')
    rows = {}
    for tag, hp in [('default', T.DEFAULTS[a.arch]),
                    ('tuned', {**T.DEFAULTS[a.arch], **study.best_params})]:
        m, met, _ = T.final_fit(a.arch, hp, d)
        pv, yv = tv_check(m, d, a.arch)
        from co2gcs.physics import violation_rate
        vr = violation_rate(m, d['Xb_te'], d['Xt_te'], d['Xt_plus_te'],
                            d['phys_sign_te'])
        rows[tag] = dict(**{f'phys_viol_{k}': v for k, v in vr.items()},
                         **{k: met[k] for k in
                            ['R2_mean'] + [f'R2_{t}' for t in C.TARGETS] +
                            ['RMSE_mean', 'closure_MAE']},
                         TV_ratio=float(np.mean(pv / np.maximum(yv, 1e-9))))
        print(f'  {tag:8s} R2_mean={met["R2_mean"]:.4f}  ' +
              '  '.join(f'{t}={met[f"R2_{t}"]:.3f}' for t in C.TARGETS) +
              f'  closure={met["closure_MAE"]:.4f}  TVratio={rows[tag]["TV_ratio"]:.2f}')
        print(f'           physics violation rate: ' +
              '  '.join(f'{k}={v:.3f}' for k, v in vr.items()))

    out = pd.DataFrame(rows).T
    out.to_csv(f'{a.outdir}/compare_{a.arch}_{a.protocol}_{a.mode}_{a.space}.csv')
    json.dump({'best_params': study.best_params,
               'best_val_r2': study.best_value,
               'importance': {k: float(v) for k, v in imp.items()},
               'test': rows},
              open(f'{a.outdir}/best_{a.arch}_{a.protocol}_{a.mode}_{a.space}.json', 'w'),
              indent=2)
    d_ = out.loc['tuned', 'R2_mean'] - out.loc['default', 'R2_mean']
    print(f'\ntuned - default on held-out cases: {d_:+.4f} R2_mean')
    print('TVratio near 1.0 means the predicted trajectories are as smooth as '
          'the simulations; >>1 means the model oscillates between timesteps.')


if __name__ == '__main__':
    main()
