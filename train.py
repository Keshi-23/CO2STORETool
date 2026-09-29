"""
Training loop and loss functions for all DeepONet head types.

Supports
--------
- point prediction
- heteroscedastic Gaussian uncertainty
- logit-normal uncertainty in ALR space
- closure penalty
- physics-informed monotonicity penalty
- Bayesian last-layer KL regularisation
- Optuna epoch callbacks / pruning
"""

import copy
import numpy as np
import torch
import torch.nn as nn


torch.set_num_threads(1)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_all(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def make_loss(
    head,
    y_train,
    var_weight=True,
    eps=1e-3
):
    """
    Create the loss function for the selected prediction head.

    var_weight=True:
        inverse-variance weighting is used so that small-scale indices such
        as MTI are not dominated by STTI.
    """

    yt = torch.as_tensor(
        np.asarray(y_train),
        dtype=torch.float32
    )

    # -------------------------------------------------------------------------
    # Inverse variance weighting
    # -------------------------------------------------------------------------

    if var_weight:

        w = 1.0 / (
            yt.var(0)
            + 1e-8
        )

        w = (
            w
            / w.sum()
            * yt.shape[1]
        )

    else:

        w = torch.ones(
            yt.shape[1],
            dtype=torch.float32
        )

    # -------------------------------------------------------------------------
    # Point loss
    # -------------------------------------------------------------------------

    def point(out, y):

        return (
            ((out - y) ** 2) * w
        ).mean()

    # -------------------------------------------------------------------------
    # Heteroscedastic Gaussian
    # -------------------------------------------------------------------------

    def hetero(out, y):

        mu, logvar = out

        mse = (
            ((mu - y) ** 2) * w
        ).mean()

        nll = 0.5 * (
            ((y - mu.detach()) ** 2)
            / logvar.exp()
            + logvar
        )

        return mse + 0.05 * nll.mean()

    # -------------------------------------------------------------------------
    # Logit-normal uncertainty
    # -------------------------------------------------------------------------

    def logitnormal(out, y):

        mean, logsig = out

        mse = (
            ((mean - y) ** 2) * w
        ).mean()

        # Normalize target composition
        ys = y / y.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1e-6)

        ys = (
            (1.0 - 4.0 * eps)
            * ys
            + eps
        )

        # ALR target
        alr_t = (
            torch.log(ys[:, 1:])
            - torch.log(ys[:, :1])
        )

        # ALR prediction
        md = mean.detach().clamp_min(
            1e-8
        )

        alr_p = (
            torch.log(md[:, 1:])
            - torch.log(md[:, :1])
        )

        # Gaussian NLL in ALR space
        nll = 0.5 * (
            (
                (alr_t - alr_p) ** 2
                / (2.0 * logsig).exp()
            )
            + 2.0 * logsig
        )

        return mse + 0.05 * nll.mean()

    losses = {
        'point': point,
        'hetero': hetero,
        'logitnormal': logitnormal
    }

    if head not in losses:

        raise ValueError(
            f"Unknown head='{head}'. "
            f"Expected one of {list(losses.keys())}."
        )

    return losses[head]


# =============================================================================
# OUTPUT HELPER
# =============================================================================

def _pred_mean(out):

    if isinstance(out, tuple):
        return out[0]

    return out


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model,
    Xb_tr,
    Xt_tr,
    y_tr,

    Xb_va,
    Xt_va,
    y_va,

    head='point',

    epochs=150,
    bs=1024,

    lr=1.5e-3,
    wd=1e-5,

    patience=35,

    var_weight=True,

    closure_pen=0.0,

    kl_weight=1.0,

    phys_weight=0.0,
    Xt_plus_tr=None,
    phys_sign_tr=None,

    verbose=False,
    epoch_callback=None
):
    """
    Train a DeepONet using validation-based early stopping.

    Physics loss
    ------------
    If phys_weight > 0, both Xt_plus_tr and phys_sign_tr must be supplied.

    Total loss:

        L =
            L_data
            + closure_pen * L_closure
            + phys_weight * L_physics
            + kl_weight * KL / N
    """

    # =========================================================================
    # Convert to tensors
    # =========================================================================

    def tt(a):

        return torch.as_tensor(
            np.asarray(a),
            dtype=torch.float32
        )

    Xb_tr = tt(Xb_tr)
    Xt_tr = tt(Xt_tr)
    y_tr_t = tt(y_tr)

    Xb_va = tt(Xb_va)
    Xt_va = tt(Xt_va)
    y_va_t = tt(y_va)

    # =========================================================================
    # Data loss
    # =========================================================================

    loss_fn = make_loss(
        head,
        y_tr,
        var_weight=var_weight
    )

    # =========================================================================
    # Physics loss
    # =========================================================================

    use_phys = (
        phys_weight > 0.0
        and Xt_plus_tr is not None
        and phys_sign_tr is not None
    )

    if phys_weight > 0.0 and not use_phys:

        raise ValueError(
            "phys_weight > 0 requires both "
            "Xt_plus_tr and phys_sign_tr."
        )

    if use_phys:

        Xtp = tt(Xt_plus_tr)
        Psg = tt(phys_sign_tr)

    # =========================================================================
    # Optimizer
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    n = len(y_tr_t)

    steps = max(
        1,
        int(np.ceil(n / bs))
    )

    # =========================================================================
    # One-cycle LR scheduler
    # =========================================================================

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * steps,
        pct_start=0.15
    )

    # =========================================================================
    # Early stopping
    # =========================================================================

    best_val = np.inf
    best_state = None
    bad_epochs = 0

    history = {
        'train': [],
        'val': [],
        'data_loss': [],
        'closure_loss': [],
        'physics_loss': [],
        'kl_loss': []
    }

    # =========================================================================
    # Epoch loop
    # =========================================================================

    for ep in range(epochs):

        model.train()

        permutation = torch.randperm(n)

        run_total = 0.0
        run_data = 0.0
        run_closure = 0.0
        run_physics = 0.0
        run_kl = 0.0

        # ---------------------------------------------------------------------
        # Mini-batches
        # ---------------------------------------------------------------------

        for i in range(0, n, bs):

            idx = permutation[i:i + bs]

            optimizer.zero_grad()

            # -------------------------------------------------------------
            # Prediction
            # -------------------------------------------------------------

            out = model(
                Xb_tr[idx],
                Xt_tr[idx]
            )

            # -------------------------------------------------------------
            # Data loss
            # -------------------------------------------------------------

            data_loss = loss_fn(
                out,
                y_tr_t[idx]
            )

            loss = data_loss

            # -------------------------------------------------------------
            # Closure penalty
            # -------------------------------------------------------------

            closure_loss = torch.zeros(
                (),
                dtype=torch.float32
            )

            if closure_pen > 0.0:

                pred = _pred_mean(out)

                closure_loss = (
                    (
                        pred.sum(dim=1)
                        - 1.0
                    ) ** 2
                ).mean()

                loss = (
                    loss
                    + closure_pen * closure_loss
                )

            # -------------------------------------------------------------
            # Physics-informed penalty
            # -------------------------------------------------------------

            physics_loss = torch.zeros(
                (),
                dtype=torch.float32
            )

            if use_phys:

                from .physics import physics_penalty

                physics_loss = physics_penalty(
                    model,
                    Xb_tr[idx],
                    Xt_tr[idx],
                    Xtp[idx],
                    Psg[idx]
                )

                loss = (
                    loss
                    + phys_weight * physics_loss
                )

            # -------------------------------------------------------------
            # Bayesian KL
            # -------------------------------------------------------------

            kl_loss = model.kl()

            if float(
                kl_loss.detach()
            ) != 0.0:

                scaled_kl = (
                    kl_weight
                    * kl_loss
                    / n
                )

                loss = (
                    loss
                    + scaled_kl
                )

            else:

                scaled_kl = torch.zeros(
                    (),
                    dtype=torch.float32
                )

            # -------------------------------------------------------------
            # Backpropagation
            # -------------------------------------------------------------

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()
            scheduler.step()

            # -------------------------------------------------------------
            # Accumulate
            # -------------------------------------------------------------

            nj = len(idx)

            run_total += (
                float(loss.detach())
                * nj
            )

            run_data += (
                float(data_loss.detach())
                * nj
            )

            run_closure += (
                float(closure_loss.detach())
                * nj
            )

            run_physics += (
                float(physics_loss.detach())
                * nj
            )

            run_kl += (
                float(scaled_kl.detach())
                * nj
            )

        # =====================================================================
        # Validation
        # =====================================================================

        model.eval()

        with torch.no_grad():

            val_out = model(
                Xb_va,
                Xt_va
            )

            val_loss = float(
                loss_fn(
                    val_out,
                    y_va_t
                )
            )

        # =====================================================================
        # History
        # =====================================================================

        history['train'].append(
            run_total / n
        )

        history['val'].append(
            val_loss
        )

        history['data_loss'].append(
            run_data / n
        )

        history['closure_loss'].append(
            run_closure / n
        )

        history['physics_loss'].append(
            run_physics / n
        )

        history['kl_loss'].append(
            run_kl / n
        )

        # =====================================================================
        # Early stopping
        # =====================================================================

        if val_loss < best_val - 1e-9:

            best_val = val_loss

            bad_epochs = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:

            bad_epochs += 1

            if bad_epochs >= patience:

                break

        # =====================================================================
        # Optuna callback
        # =====================================================================

        if epoch_callback is not None:

            epoch_callback(
                ep,
                val_loss
            )

        # =====================================================================
        # Progress
        # =====================================================================

        if verbose and ep % 25 == 0:

            print(
                f"  ep {ep:3d} "
                f"train={history['train'][-1]:.4e} "
                f"val={history['val'][-1]:.4e} "
                f"data={history['data_loss'][-1]:.4e} "
                f"phys={history['physics_loss'][-1]:.4e}",
                flush=True
            )

    # =========================================================================
    # Restore best validation model
    # =========================================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, history


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict_mean(
    model,
    Xb,
    Xt,
    bs=8192
):
    """
    Return predictive mean for any supported output head.
    """

    model.eval()

    outputs = []

    for i in range(
        0,
        len(Xb),
        bs
    ):

        xb = torch.as_tensor(
            Xb[i:i + bs],
            dtype=torch.float32
        )

        xt = torch.as_tensor(
            Xt[i:i + bs],
            dtype=torch.float32
        )

        out = model(
            xb,
            xt
        )

        outputs.append(
            _pred_mean(out)
            .cpu()
            .numpy()
        )

    return np.vstack(outputs)