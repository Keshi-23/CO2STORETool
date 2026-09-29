"""Data loading, physics-based feature engineering, and splitting protocols."""
import numpy as np
import pandas as pd

STATIC9 = ['Dykstra Parson Coefficient', 'Average Porosity', 'Biots Coefficient',
           "Young's Modulus (MPa)", "Poisson's Ratio", 'Perforation Thickness (m)',
           'Reservoir Thickness (m)', 'Acquifer Salinity (PPM)',
           'Reservoir Temperature (oC)']
TARGETS = ['STTI', 'RTI', 'STI', 'MTI']
NICE = {'STTI': 'Structural (STTI)', 'RTI': 'Residual (RTI)',
        'STI': 'Solubility (STI)', 'MTI': 'Mineral (MTI)'}
T_SHUT = 30.0
BASE_COLS = ['Dykstra Parson Coefficient', 'Average Porosity', 'Biots Coefficient',
             "Young's Modulus (MPa)", "Poisson's Ratio", 'Perforation Thickness (m)',
             'Reservoir Thickness (m)', 'Acquifer Salinity (PPM)',
             'Reservoir Temperature (oC)', 'q_inj']


def load(path='All_dataset.csv', fix_injection=True):
    """Load the GCS ensemble.

    fix_injection=True restores each case's true injection rate over the whole
    trajectory. In the raw file the column is hard-set to 0 for every t > 30 yr,
    which (a) leaks the post-injection regime label and (b) deletes the single
    strongest control on long-term trapping across 85% of the time horizon.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])
    df = df.rename(columns={"Structural Trapping Index": "STTI",
                            "Residual Trapping Index": "RTI",
                            "Solubility Trapping Index": "STI",
                            "Mineral Trapping Index": "MTI"})
    df['case'] = df.groupby(STATIC9, sort=False).ngroup()
    df = df.sort_values(['case', 'Time (years)']).reset_index(drop=True)
    df['q_inj'] = (df.groupby('case')['Injection rate (m3/day)'].transform('max')
                   if fix_injection else df['Injection rate (m3/day)'])
    return df


def build_features(df, mode='fixed'):
    """Branch (case) features and trunk (time) features."""
    t = df['Time (years)'].values
    if mode == 'orig':
        cols = ['Dykstra Parson Coefficient', 'Average Porosity', 'Injection rate (m3/day)',
                'Biots Coefficient', "Young's Modulus (MPa)", "Poisson's Ratio",
                'Perforation Thickness (m)', 'Reservoir Thickness (m)',
                'Acquifer Salinity (PPM)', 'Reservoir Temperature (oC)']
        Xb, names_b = df[cols].values.astype(np.float32), cols
        Xt, names_t = t.reshape(-1, 1).astype(np.float32), ['t']
    else:
        q = df['q_inj'].values
        phi = df['Average Porosity'].values
        h_res = df['Reservoir Thickness (m)'].values
        h_perf = df['Perforation Thickness (m)'].values
        S = df['Acquifer Salinity (PPM)'].values
        T = df['Reservoir Temperature (oC)'].values
        M = q * T_SHUT * 365.0
        feats = {
            'Vdp': df['Dykstra Parson Coefficient'].values, 'phi': phi, 'q_inj': q,
            'biot': df['Biots Coefficient'].values, 'E': df["Young's Modulus (MPa)"].values,
            'nu': df["Poisson's Ratio"].values, 'h_perf': h_perf, 'h_res': h_res,
            'salinity': S, 'temp': T,
            'log_M_inj': np.log10(M + 1.0),
            'PV_proxy': np.log10(phi * h_res + 1e-9),
            'perf_frac': h_perf / h_res,
            'M_over_PV': np.log10(M / (phi * h_res) + 1.0),
            'log_salinity': np.log10(S),
            'sol_proxy': (T - 60.0) / 50.0 - (S / 1e5),
        }
        Xb = np.column_stack(list(feats.values())).astype(np.float32)
        names_b = list(feats.keys())
        t_post = np.maximum(t - T_SHUT, 0.0)
        Xt = np.column_stack([
            np.log10(t + 1e-2), t / 200.0, np.minimum(t, T_SHUT) / T_SHUT,
            np.log10(t_post + 1.0) / np.log10(171.0), (t > T_SHUT).astype(float),
        ]).astype(np.float32)
        names_t = ['log_t', 't_lin', 't_inj', 'log_t_post', 'post_flag']

    y = (df[TARGETS].values.astype(np.float32)
         if all(c in df.columns for c in TARGETS) else None)
    return Xb, Xt, y, names_b, names_t


# --------------------------------------------------------------------------- #
#  Splitting protocols
# --------------------------------------------------------------------------- #
def make_split(protocol, n_rows, groups, seed=42, test_frac=0.15, val_frac=0.15,
               n_folds=5, fold=0):
    """Return boolean masks (train, val, test).

    protocol='random' : rows assigned independently. Timesteps from the SAME
        simulation appear in train and test, so the test task is interpolation
        along an already-seen trajectory.
    protocol='group'  : whole simulations assigned to one partition. The test
        task is prediction for an unseen reservoir realisation, which is the
        deployment question for site screening.
    """
    rng = np.random.RandomState(seed)
    if protocol == 'random':
        idx = rng.permutation(n_rows)
        n_te = int(test_frac * n_rows); n_va = int(val_frac * n_rows)
        te, va = idx[:n_te], idx[n_te:n_te + n_va]
        m_te = np.zeros(n_rows, bool); m_te[te] = True
        m_va = np.zeros(n_rows, bool); m_va[va] = True
        return ~(m_te | m_va), m_va, m_te

    cases = np.unique(groups)
    perm = rng.permutation(cases)
    if n_folds and n_folds > 1:                      # GroupKFold-style fold
        folds = np.array_split(perm, n_folds)
        te_c = set(folds[fold])
        rest = np.concatenate([f for i, f in enumerate(folds) if i != fold])
    else:
        n_te = int(test_frac * len(cases))
        te_c, rest = set(perm[:n_te]), perm[n_te:]
    va_c = set(rest[:max(1, int(val_frac * len(cases)))])
    m_te = np.isin(groups, list(te_c))
    m_va = np.isin(groups, list(va_c))
    return ~(m_te | m_va), m_va, m_te


PROTOCOL_NOTES = {
    'random': dict(
        name='Protocol 1 - random row split',
        pros=['Maximises training rows for a fixed ensemble size.',
              'Appropriate if the deployment task is gap-filling or temporal '
              'interpolation within simulations that are already available.',
              'Low variance between repeats, so results look stable.'],
        cons=['Neighbouring timesteps of the same trajectory land in both train and '
              'test, so the test set is not independent of training.',
              'Reported R2 measures interpolation along a seen curve, not '
              'generalisation to a new reservoir.',
              'Smooth, slowly varying indices (MTI in particular) are inflated most, '
              'because the neighbouring years almost determine the answer.',
              'Hides loss-weighting and feature defects that only appear when the '
              'model must extrapolate to an unseen case.']),
    'group': dict(
        name='Protocol 2 - GroupKFold by simulation case',
        pros=['Test cases are fully unseen, matching the site-screening use case '
              'where a new reservoir realisation must be scored.',
              'Exposes genuine parameter-to-response learning rather than curve '
              'interpolation.',
              'Fold-to-fold spread is a meaningful estimate of deployment variance.',
              'Reveals per-index failures (e.g. MTI collapse under uniform loss '
              'weighting) that random splitting conceals.'],
        cons=['Pessimistic if the real task truly is interpolation within a fixed '
              'ensemble.',
              'Fewer effective independent samples (761 cases, not 47k rows), so '
              'estimates are noisier and more folds are needed.',
              'Requires the grouping key to be correct; here it must be derived from '
              'the static parameters, not from row order.',
              'Does not test temporal extrapolation on its own - a separate '
              'train-early / predict-late protocol is still needed.']),
}
