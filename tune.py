"""
Optuna hyperparameter optimisation for BaselineDeepONet and ClosureDeepONet.

Both architectures independently tune:
    n_branch_layers
    n_trunk_layers

The HPO search ranges can be supplied from the notebook through `search_space`.
If no custom search space is supplied, DEFAULT_SEARCH_SPACE is used.

Physics-aware HPO:
    phys_weight is passed directly to train_model(), together with
    Xt_plus_tr and phys_sign_tr.
"""

import os
import numpy as np
import optuna

from sklearn.preprocessing import StandardScaler

from .data import load, build_features, make_split
from .models import BaselineDeepONet, ClosureDeepONet
from .train import train_model, predict_mean, seed_all
from .metrics import accuracy
from .physics import build_shift


optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# DEFAULT SEARCH SPACES
# =============================================================================

DEFAULT_SEARCH_SPACE = {

    "closure": {

        "lr": {
            "type": "float",
            "low": 1e-4,
            "high": 5e-3,
            "log": True
        },

        "wd": {
            "type": "float",
            "low": 1e-7,
            "high": 1e-3,
            "log": True
        },

        "bs": {
            "type": "categorical",
            "choices": [512, 1024, 2048]
        },

        "width": {
            "type": "categorical",
            "choices": [128, 192, 256]
        },

        "n_branch_layers": {
            "type": "int",
            "low": 2,
            "high": 6
        },

        "n_trunk_layers": {
            "type": "int",
            "low": 2,
            "high": 8
        },

        "p": {
            "type": "categorical",
            "choices": [64, 128, 192, 256]
        },

        "sigma": {
            "type": "float",
            "low": 0.1,
            "high": 2.0,
            "log": True
        },

        # Extended parameters
        "p_drop": {
            "type": "float",
            "low": 0.0,
            "high": 0.30
        },

        "kl_weight": {
            "type": "float",
            "low": 1e-3,
            "high": 1.0,
            "log": True
        },

        "epochs": {
            "type": "int",
            "low": 60,
            "high": 220,
            "step": 20
        },

        "patience": {
            "type": "int",
            "low": 15,
            "high": 50,
            "step": 5
        },

        "var_weight": {
            "type": "categorical",
            "choices": [True, False]
        },

        "phys_weight": {
            "type": "float",
            "low": 1e-5,
            "high": 1.0,
            "log": True
        },

        "n_freq": {
            "type": "categorical",
            "choices": [16, 32, 64]
        },
    },

    "baseline": {

        "lr": {
            "type": "float",
            "low": 1e-4,
            "high": 5e-3,
            "log": True
        },

        "wd": {
            "type": "float",
            "low": 1e-7,
            "high": 1e-3,
            "log": True
        },

        "bs": {
            "type": "categorical",
            "choices": [1024, 2048, 4096]
        },

        "width": {
            "type": "categorical",
            "choices": [96, 128, 192]
        },

        "n_branch_layers": {
            "type": "int",
            "low": 1,
            "high": 4
        },

        "n_trunk_layers": {
            "type": "int",
            "low": 4,
            "high": 10
        },

        "closure_pen": {
            "type": "float",
            "low": 1e-2,
            "high": 1e1,
            "log": True
        },

        # Extended parameters
        "p_drop": {
            "type": "float",
            "low": 0.0,
            "high": 0.30
        },

        "kl_weight": {
            "type": "float",
            "low": 1e-3,
            "high": 1.0,
            "log": True
        },

        "epochs": {
            "type": "int",
            "low": 60,
            "high": 220,
            "step": 20
        },

        "patience": {
            "type": "int",
            "low": 15,
            "high": 50,
            "step": 5
        },

        "var_weight": {
            "type": "categorical",
            "choices": [True, False]
        },

        "phys_weight": {
            "type": "float",
            "low": 1e-5,
            "high": 1.0,
            "log": True
        },
    }
}


# =============================================================================
# FIXED SETTINGS
# =============================================================================

FIXED_NOTE = {

    "basic": {

        "closure": (
            "Basic HPO: lr, wd, batch size, width, "
            "branch depth, trunk depth, p and sigma."
        ),

        "baseline": (
            "Basic HPO: lr, wd, batch size, width, "
            "branch depth, trunk depth and closure penalty."
        ),
    },

    "extended": {

        "closure": (
            "Extended HPO: basic parameters plus dropout, KL weight, "
            "epochs, patience, variance weighting, physics weight and n_freq."
        ),

        "baseline": (
            "Extended HPO: basic parameters plus dropout, KL weight, "
            "epochs, patience, variance weighting and physics weight."
        ),
    }
}


# =============================================================================
# OPTUNA PARAMETER SUGGESTION
# =============================================================================

def _suggest_parameter(trial, name, spec):
    """Create one Optuna parameter from a search-space specification."""

    ptype = spec["type"]

    if ptype == "float":

        kwargs = {
            "name": name,
            "low": spec["low"],
            "high": spec["high"],
        }

        if "log" in spec:
            kwargs["log"] = spec["log"]

        if "step" in spec:
            kwargs["step"] = spec["step"]

        return trial.suggest_float(**kwargs)

    if ptype == "int":

        kwargs = {
            "name": name,
            "low": spec["low"],
            "high": spec["high"],
        }

        if "step" in spec:
            kwargs["step"] = spec["step"]

        return trial.suggest_int(**kwargs)

    if ptype == "categorical":

        return trial.suggest_categorical(
            name,
            spec["choices"]
        )

    raise ValueError(
        f"Unknown parameter type '{ptype}' for '{name}'"
    )


def suggest_basic(
    trial,
    arch,
    search_space=None
):
    """
    Suggest the basic HPO parameters.

    `search_space` can be supplied from the notebook.
    """

    space = (
        search_space
        if search_space is not None
        else DEFAULT_SEARCH_SPACE
    )

    arch_space = space[arch]

    basic_parameters = [
        "lr",
        "wd",
        "bs",
        "width",
        "n_branch_layers",
        "n_trunk_layers",
    ]

    if arch == "closure":

        basic_parameters += [
            "p",
            "sigma"
        ]

    else:

        basic_parameters += [
            "closure_pen"
        ]

    hp = {}

    for name in basic_parameters:

        hp[name] = _suggest_parameter(
            trial,
            name,
            arch_space[name]
        )

    return hp


def suggest_extended(
    trial,
    arch,
    hp,
    search_space=None
):
    """Add extended HPO parameters."""

    space = (
        search_space
        if search_space is not None
        else DEFAULT_SEARCH_SPACE
    )

    arch_space = space[arch]

    extended_parameters = [
        "p_drop",
        "kl_weight",
        "epochs",
        "patience",
        "var_weight",
        "phys_weight",
    ]

    if arch == "closure":
        extended_parameters.append("n_freq")

    hp = dict(hp)

    for name in extended_parameters:

        hp[name] = _suggest_parameter(
            trial,
            name,
            arch_space[name]
        )

    return hp


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(
    arch,
    hp,
    db,
    dt,
    p_drop=None,
    head="point",
    bayes_last=False
):

    drop = (
        hp.get("p_drop", 0.0)
        if p_drop is None
        else p_drop
    )

    if arch == "closure":

        return ClosureDeepONet(
            db,
            dt,

            p=hp.get("p", 128),

            width=hp.get(
                "width",
                192
            ),

            n_branch_layers=hp.get(
                "n_branch_layers",
                4
            ),

            n_trunk_layers=hp.get(
                "n_trunk_layers",
                4
            ),

            n_freq=hp.get(
                "n_freq",
                32
            ),

            sigma=hp.get(
                "sigma",
                0.3
            ),

            p_drop=drop,

            head=head,

            bayes_last=bayes_last
        )

    if arch == "baseline":

        return BaselineDeepONet(
            db,
            dt,

            width=hp.get(
                "width",
                128
            ),

            n_branch_layers=hp.get(
                "n_branch_layers",
                2
            ),

            n_trunk_layers=hp.get(
                "n_trunk_layers",
                8
            ),

            p_drop=drop,

            head=head,

            bayes_last=bayes_last
        )

    raise ValueError(
        f"Unknown architecture: {arch}"
    )


# =============================================================================
# TRAINING KWARGS
# =============================================================================

def train_kw_from(
    arch,
    hp,
    epochs=None
):

    common = {

        "lr": hp["lr"],

        "wd": hp["wd"],

        "var_weight": hp.get(
            "var_weight",
            True
        ),

        "kl_weight": hp.get(
            "kl_weight",
            0.05
        ),

        "phys_weight": hp.get(
            "phys_weight",
            0.0
        ),

        "patience": hp.get(
            "patience",
            30
        ),

        "verbose": False,
    }

    if arch == "closure":

        return {

            "epochs": (
                epochs
                if epochs is not None
                else hp.get("epochs", 80)
            ),

            "bs": hp.get(
                "bs",
                1024
            ),

            **common,
        }

    return {

        "epochs": (
            epochs
            if epochs is not None
            else hp.get("epochs", 140)
        ),

        "bs": hp.get(
            "bs",
            2048
        ),

        "closure_pen": hp.get(
            "closure_pen",
            1.0
        ),

        **common,
    }


# =============================================================================
# DATA
# =============================================================================

def make_data(
    protocol,
    mode="fixed",
    csv="All_dataset.csv",
    seed=42,
    fold=0
):

    df = load(
        csv,
        fix_injection=(mode == "fixed")
    )

    Xb, Xt, y, _, _ = build_features(
        df,
        mode=mode
    )

    groups = df["case"].values

    tr, va, te = make_split(
        protocol,
        len(y),
        groups,
        seed=seed,
        n_folds=5,
        fold=fold
    )

    scaler_b = StandardScaler().fit(
        Xb[tr]
    )

    scaler_t = StandardScaler().fit(
        Xt[tr]
    )

    def transform(mask):

        return (
            scaler_b.transform(
                Xb[mask]
            ).astype("float32"),

            scaler_t.transform(
                Xt[mask]
            ).astype("float32"),

            y[mask]
        )

    d = {}

    for name, mask in [
        ("tr", tr),
        ("va", va),
        ("te", te)
    ]:

        (
            d[f"Xb_{name}"],
            d[f"Xt_{name}"],
            d[f"y_{name}"]
        ) = transform(mask)

    Xp, signs = build_shift(
        df,
        mode=mode
    )

    for name, mask in [
        ("tr", tr),
        ("va", va),
        ("te", te)
    ]:

        d[f"Xt_plus_{name}"] = (
            scaler_t.transform(
                Xp[mask]
            ).astype("float32")
        )

        d[f"phys_sign_{name}"] = signs[mask]

    d["dims"] = (
        Xb.shape[1],
        Xt.shape[1]
    )

    d["groups_te"] = groups[te]

    d["t_te"] = (
        df["Time (years)"].values[te]
    )

    return d


# =============================================================================
# FIT
# =============================================================================

def _fit(
    arch,
    hp,
    d,
    epochs,
    seed,
    cb=None
):

    db, dt = d["dims"]

    seed_all(seed)

    model = build_model(
        arch,
        hp,
        db,
        dt
    )

    train_kw = train_kw_from(
        arch,
        hp,
        epochs=epochs
    )

    model, history = train_model(

        model,

        d["Xb_tr"],
        d["Xt_tr"],
        d["y_tr"],

        d["Xb_va"],
        d["Xt_va"],
        d["y_va"],

        head="point",

        Xt_plus_tr=d["Xt_plus_tr"],

        phys_sign_tr=d["phys_sign_tr"],

        epoch_callback=cb,

        **train_kw
    )

    return model, history


# =============================================================================
# OBJECTIVE
# =============================================================================

def objective_factory(
    arch,
    d,
    search_epochs,
    seed=42,
    prune=True,
    space="basic",
    search_space=None
):

    def objective(trial):

        hp = suggest_basic(
            trial,
            arch,
            search_space=search_space
        )

        if space == "extended":

            hp = suggest_extended(
                trial,
                arch,
                hp,
                search_space=search_space
            )

        def cb(ep, vloss):

            if not prune:
                return

            trial.report(
                vloss,
                ep
            )

            if trial.should_prune():

                raise optuna.TrialPruned()

        model, _ = _fit(
            arch,
            hp,
            d,
            search_epochs,
            seed,
            cb
        )

        pred = predict_mean(
            model,
            d["Xb_va"],
            d["Xt_va"]
        )

        r2 = accuracy(
            d["y_va"],
            pred
        )["R2_mean"]

        return (
            r2
            if np.isfinite(r2)
            else -1e3
        )

    return objective


# =============================================================================
# RUN STUDY
# =============================================================================

def run_study(
    arch,
    protocol,
    mode="fixed",
    n_trials=10,
    search_epochs=45,
    storage_dir="optuna",
    seed=42,
    timeout=None,
    csv="All_dataset.csv",
    space="basic",
    search_space=None
):

    os.makedirs(
        storage_dir,
        exist_ok=True
    )

    name = (
        f"{arch}_{protocol}_{mode}"
        + (
            "_ext"
            if space == "extended"
            else ""
        )
    )

    study = optuna.create_study(

        study_name=name,

        storage=(
            f"sqlite:///{storage_dir}/{name}.db"
        ),

        load_if_exists=True,

        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=6
        ),

        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=12
        )
    )

    d = make_data(
        protocol,
        mode=mode,
        csv=csv,
        seed=seed
    )

    if n_trials:

        study.optimize(

            objective_factory(
                arch,
                d,
                search_epochs,
                seed=seed,
                space=space,
                search_space=search_space
            ),

            n_trials=n_trials,

            timeout=timeout,

            catch=(
                RuntimeError,
                ValueError
            )
        )

    return study, d


# =============================================================================
# FINAL FIT
# =============================================================================

def final_fit(
    arch,
    hp,
    d,
    seed=42,
    epochs=None
):

    model, history = _fit(
        arch,
        hp,
        d,
        epochs,
        seed
    )

    pred = predict_mean(
        model,
        d["Xb_te"],
        d["Xt_te"]
    )

    metrics = accuracy(
        d["y_te"],
        pred
    )

    return model, metrics, pred


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

DEFAULTS = {

    "closure": {

        "lr": 1.5e-3,
        "wd": 1e-5,
        "bs": 1024,

        "width": 192,

        "n_branch_layers": 4,
        "n_trunk_layers": 4,

        "p": 128,

        "sigma": 0.3,
        "n_freq": 32,

        "p_drop": 0.0,
        "epochs": 80,
        "patience": 30,

        "var_weight": True,

        "phys_weight": 0.0,

        "kl_weight": 0.05,
    },

    "baseline": {

        "lr": 1.5e-3,
        "wd": 8.5e-5,
        "bs": 2048,

        "width": 128,

        "n_branch_layers": 2,
        "n_trunk_layers": 8,

        "closure_pen": 1.0,

        "p_drop": 0.0,
        "epochs": 140,
        "patience": 30,

        "var_weight": True,

        "phys_weight": 0.0,

        "kl_weight": 0.05,
    },
}


# =============================================================================
# PARAMETER IMPORTANCE
# =============================================================================

def importance(study):

    try:

        return optuna.importance.get_param_importances(
            study
        )

    except Exception:

        return {}