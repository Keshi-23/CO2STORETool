"""Run a one-at-a-time sensitivity / trend analysis.

Examples
--------
python run_sensitivity.py --param salinity
python run_sensitivity.py --param temperature --protocol both --overlay nearest
python run_sensitivity.py --param injection --arch baseline --n-levels 15
python run_sensitivity.py --param vdp --min 0.2 --max 0.9 --logx
python run_sensitivity.py --all                # sweep every listed parameter
"""
import argparse, os
import numpy as np, pandas as pd
import co2gcs as C
from co2gcs import sensitivity as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='salinity',
                    help=f'one of {list(S.PARAMS)} (or the raw column name)')
    ap.add_argument('--all', action='store_true', help='sweep every parameter')
    ap.add_argument('--protocol', default='both',
                    choices=['random', 'group', 'both'])
    ap.add_argument('--arch', default='closure', choices=['closure', 'baseline'])
    ap.add_argument('--n-levels', type=int, default=12)
    ap.add_argument('--min', type=float, default=None)
    ap.add_argument('--max', type=float, default=None)
    ap.add_argument('--overlay', default='none', choices=['none', 'nearest'])
    ap.add_argument('--logx', action='store_true')
    ap.add_argument('--csv', default='All_dataset.csv')
    ap.add_argument('--outdir', default='figs_sens')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df = C.load(a.csv, fix_injection=True)
    Xb, Xt, y, _, _ = C.build_features(df, mode='fixed')

    protos = ['random', 'group'] if a.protocol == 'both' else [a.protocol]
    models = {}
    for p in protos:
        models[p] = S.get_model(a.arch, p, df, Xb, Xt, y)

    base = S.base_case(df)
    print('\nBase case (dataset case-level mean):')
    for k, v in base.items():
        print(f'  {k:32s} {v:10.3f}')

    params = list(S.PARAMS) if a.all else [a.param]
    for prm in params:
        col, label = S.resolve(prm)
        levels = None
        if a.min is not None and a.max is not None and not a.all:
            levels = np.linspace(a.min, a.max, a.n_levels)
        res = {}
        for p in protos:
            m, sp = models[p]
            res[p] = S.sweep(m, sp, prm, df, levels=levels, n_levels=a.n_levels)

        path = f'{a.outdir}/sens_{a.arch}_{prm}.png'
        S.fig_sensitivity(res, path, df=df, overlay=a.overlay, logx=a.logx,
                          title=f'{a.arch} DeepONet - one-at-a-time sensitivity to '
                                f'{label} (all other inputs at dataset mean)')
        print(f'\n=== {label} -> {path}')

        for p in protos:
            chk = S.physics_check(res[p], df=df)
            mp = chk.pivot(index='index', columns='t', values='model_spearman')
            dp = chk.pivot(index='index', columns='t', values='data_spearman')
            comb = pd.concat({'model': mp.reindex(C.TARGETS),
                              'simulation data': dp.reindex(C.TARGETS)},
                             axis=1).round(3)
            print(f'  [{p}] trend vs {label} (Spearman; data = partial corr. '
                  f'over the 761 real cases):')
            print('   ' + comb.to_string().replace('\n', '\n   '))
            md = chk.dropna(subset=['matches_data'])
            if len(md):
                print(f'   model reproduces the ensemble trend in '
                      f'{int(md.matches_data.sum())}/{len(md)} checks')
            tb = chk.dropna(subset=['matches_textbook'])
            if len(tb):
                print(f'   matches textbook prior in '
                      f'{int(tb.matches_textbook.sum())}/{len(tb)} checks')
            chk.to_csv(f'{a.outdir}/check_{a.arch}_{prm}_{p}.csv', index=False)

        if len(protos) == 2:
            d = np.abs(res['group']['pred'] - res['random']['pred']).mean((0, 1))
            print('  protocol disagreement (mean |group - random|): ' +
                  ', '.join(f'{t}={v:.4f}' for t, v in zip(C.TARGETS, d)))


if __name__ == '__main__':
    main()
