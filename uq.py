"""Three interchangeable UQ engines. Each returns predictive samples of shape
(S, N, 4) from which mean, sd and quantiles follow.

Choosing a method
-----------------
deep_ensemble  - strongest epistemic estimates and best calibration in practice;
                 cost is M training runs. Recommended default for the paper.
mc_dropout     - one training run, S cheap forward passes. Good value, but the
                 dropout rate is an unprincipled knob and it systematically
                 under-disperses on unseen simulation cases.
bayesian       - variational posterior on the final projection. One training
                 run, principled, and especially natural for a DeepONet because
                 the output is linear in the learned basis, so the last layer
                 carries most of the predictive uncertainty. Full VI over all
                 weights was not used: it costs far more and generally does not
                 beat ensembles on calibration.

Aleatoric noise comes from the head (heteroscedastic Gaussian for the baseline
net, logit-normal for the closure net); epistemic spread comes from the engine.
Every engine can be combined with either head.
"""
import numpy as np
import torch

from .train import train_model, seed_all, _pred_mean


def _enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def _sample_head(out, head, rng, sample_aleatoric=True):
    """Draw one observation-level sample from the predictive head."""
    if head == 'hetero':
        mu, logvar = out
        mu = mu.numpy()
        if not sample_aleatoric:
            return mu
        sd = np.exp(0.5 * logvar.numpy())
        return np.clip(mu + rng.standard_normal(mu.shape) * sd, 0.0, None)
    if head == 'logitnormal':
        mean, logsig = out
        if not sample_aleatoric:
            return mean.numpy()
        m = np.clip(mean.numpy().astype(np.float64), 1e-9, None)
        alr = np.log(m[:, 1:]) - np.log(m[:, :1])
        alr = alr + np.exp(logsig.numpy()) * rng.standard_normal(alr.shape)
        z = np.concatenate([np.zeros((len(alr), 1)), alr], 1)
        z = np.exp(z - z.max(1, keepdims=True))
        return (z / z.sum(1, keepdims=True)).astype(np.float32)
    return _pred_mean(out).numpy()


@torch.no_grad()
def _forward_all(model, Xb, Xt, head, rng, bs=8192, sample_aleatoric=True,
                 bayes_sample=True):
    o = []
    for i in range(0, len(Xb), bs):
        xb = torch.as_tensor(Xb[i:i + bs], dtype=torch.float32)
        xt = torch.as_tensor(Xt[i:i + bs], dtype=torch.float32)
        try:
            out = model(xb, xt, sample=bayes_sample)
        except TypeError:
            out = model(xb, xt)
        o.append(_sample_head(out, head, rng, sample_aleatoric))
    return np.vstack(o)


# --------------------------------------------------------------------------- #
def run_uq(method, model_fn, data, head='point', n_samples=200, n_members=5,
           seed=42, train_kw=None, verbose=True, pretrained=None):
    """method in {'deterministic','mc_dropout','deep_ensemble','bayesian'}.

    model_fn(seed) must return a fresh model. data is a dict with keys
    Xb_tr, Xt_tr, y_tr, Xb_va, Xt_va, y_va, Xb_te, Xt_te.
    Returns dict with 'samples' (S,N,4), 'models', 'hist'.
    """
    train_kw = dict(train_kw or {})
    train_kw.setdefault('head', head)
    rng = np.random.default_rng(seed)
    d = data
    models, hists = [], []

    _pre = list(pretrained) if pretrained else []

    def fit(s):
        if _pre:                       # reuse a cached model, skip training
            return _pre.pop(0), None
        seed_all(s)
        m = model_fn(s)
        m, h = train_model(m, d['Xb_tr'], d['Xt_tr'], d['y_tr'],
                           d['Xb_va'], d['Xt_va'], d['y_va'], **train_kw)
        return m, h

    if method in ('deterministic', 'mc_dropout', 'bayesian'):
        m, h = fit(seed)
        models, hists = [m], [h]
        if method == 'deterministic':
            S = [_forward_all(m, d['Xb_te'], d['Xt_te'], head, rng,
                              sample_aleatoric=False, bayes_sample=False)]
        elif method == 'mc_dropout':
            _enable_dropout(m)
            S = [_forward_all(m, d['Xb_te'], d['Xt_te'], head, rng)
                 for _ in range(n_samples)]
            m.eval()
        else:
            S = [_forward_all(m, d['Xb_te'], d['Xt_te'], head, rng, bayes_sample=True)
                 for _ in range(n_samples)]
    elif method == 'deep_ensemble':
        per = max(1, n_samples // n_members)
        S = []
        for k in range(n_members):
            m, h = fit(seed + 101 * k)
            models.append(m); hists.append(h)
            S += [_forward_all(m, d['Xb_te'], d['Xt_te'], head, rng)
                  for _ in range(per)]
            if verbose:
                print(f'    ensemble member {k + 1}/{n_members} done', flush=True)
    else:
        raise ValueError(method)

    return dict(samples=np.stack(S), models=models, hist=hists)


def decompose(samples_by_member):
    """Total = aleatoric + epistemic, via the law of total variance.
    samples_by_member: list over members of (S_k, N, 4) arrays."""
    means = np.stack([s.mean(0) for s in samples_by_member])          # (M,N,4)
    within = np.stack([s.var(0) for s in samples_by_member]).mean(0)  # aleatoric
    between = means.var(0)                                            # epistemic
    return dict(aleatoric=within, epistemic=between, total=within + between,
                mean=means.mean(0))
