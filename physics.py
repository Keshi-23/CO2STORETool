"""Regime-aware monotonicity (physics) loss over all four trapping indices.

The notebook's original term,

    L_phys = ReLU(-d_t RTI)^2 + ReLU(-d_t STI)^2,

forces both indices to increase at every time. Measured on the 761 simulations
that is only true after shut-in, and it is false for the other two mechanisms,
which is why the term was disabled rather than extended: applied globally it
contradicts the data it is meant to regularise.

Fraction of timesteps satisfying "non-decreasing", measured on the ensemble:

    regime            STTI     RTI      STI      MTI
    injection t<=30   0.899    0.437    0.090    0.628
    post shut-in      0.000    0.986    0.996    0.973

So the defensible constraint set is regime-dependent:

    t <= 30 :  d_t STTI >= 0 ,  d_t STI <= 0
    t >  30 :  d_t STTI <= 0 ,  d_t RTI >= 0 ,  d_t STI >= 0 ,  d_t MTI >= 0

RTI and MTI are left free during injection (0.44 and 0.63 - no usable prior).
This covers all four indices, as requested, while remaining consistent with the
simulator.

The derivative is evaluated by forward difference on the model itself: the trunk
features are rebuilt at t(1+delta) and the sign of the predicted change is
penalised. That avoids the chain-rule bookkeeping needed to differentiate through
five correlated time encodings, and it costs one extra forward pass.
"""
import numpy as np
import torch

from .data import build_features, TARGETS, T_SHUT

# +1 -> should not decrease,  -1 -> should not increase,  0 -> unconstrained
PHYS_SPEC = {
    'injection': dict(STTI=+1, RTI=0, STI=-1, MTI=0),
    'post':      dict(STTI=-1, RTI=+1, STI=+1, MTI=+1),
}


def build_shift(df, mode='fixed', delta=0.05, spec=PHYS_SPEC):
    """Trunk features at t(1+delta) plus the per-row expected-sign matrix."""
    t = df['Time (years)'].values.astype(float)
    t2 = np.minimum(t * (1.0 + delta) + 1e-3, 200.0)
    d2 = df.copy()
    d2['Time (years)'] = t2
    if 'Injection rate (m3/day)' in d2:
        q = df['q_inj'].values if 'q_inj' in df else df['Injection rate (m3/day)'].values
        d2['Injection rate (m3/day)'] = np.where(t2 <= T_SHUT, q, 0.0)
    _, Xt_plus, _, _, _ = build_features(d2, mode=mode)

    sign = np.zeros((len(t), 4), dtype=np.float32)
    post = t > T_SHUT
    for k, name in enumerate(TARGETS):
        sign[~post, k] = spec['injection'][name]
        sign[post, k] = spec['post'][name]
    # do not constrain across the shut-in discontinuity itself
    sign[(t <= T_SHUT) & (t2 > T_SHUT)] = 0.0
    return Xt_plus.astype(np.float32), sign


def physics_penalty(model, xb, xt, xt_plus, sign, weights=None):
    """ReLU hinge on the forward difference, squared, over all four indices."""
    from .train import _pred_mean
    y0 = _pred_mean(model(xb, xt))
    y1 = _pred_mean(model(xb, xt_plus))
    d = y1 - y0
    viol = torch.relu(-sign * d)          # >0 only when the sign is wrong
    if weights is not None:
        viol = viol * weights
    return (viol ** 2).mean()


def violation_rate(model, Xb, Xt, Xt_plus, sign, bs=8192):
    """Fraction of constrained (row, index) pairs whose predicted derivative has
    the wrong sign. Reported per index."""
    from .train import _pred_mean
    model.eval()
    bad = np.zeros(4); tot = np.zeros(4)
    with torch.no_grad():
        for i in range(0, len(Xb), bs):
            sl = slice(i, i + bs)
            tt = lambda a: torch.as_tensor(a[sl], dtype=torch.float32)
            d = (_pred_mean(model(tt(Xb), tt(Xt_plus)))
                 - _pred_mean(model(tt(Xb), tt(Xt)))).numpy()
            s = sign[sl]
            m = s != 0
            bad += ((s * d < 0) & m).sum(0)
            tot += m.sum(0)
    return {t: float(bad[k] / max(tot[k], 1)) for k, t in enumerate(TARGETS)}
