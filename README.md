# CO2STORETool

**A protocol-aware, uncertainty-aware deep learning framework for CO₂ storage trapping indices.**

CO2STORETool predicts the four CO₂ trapping mechanisms — structural (STTI), residual (RTI), solubility (STI) and mineral (MTI) — over a 0–200 year storage horizon from ten reservoir and operational parameters. It is built around two ideas that are usually left implicit in subsurface ML work:

- **Protocol awareness.** How rows are assigned to train/test decides what the reported accuracy means. Random row splitting on trajectory data measures interpolation along curves the model has already seen; case-level grouping measures what site screening actually needs.
- **Uncertainty awareness.** A point prediction is not a decision input. Every model can emit calibrated predictive intervals, and calibration quality is reported alongside accuracy rather than assumed.

> **Status:** code development and initial testing. Physics validation, a GUI, and publication are the next phase. APIs may change.

---

## Why this exists

Surrogate models for geological carbon storage are typically scored with a random train/test split and a single R². On a simulation ensemble — where each case contributes dozens of timesteps along a smooth trajectory — that number answers the wrong question, and it hides failure modes that only appear when the model must generalise to an unseen reservoir.

CO2STORETool makes the evaluation protocol an explicit, switchable part of the pipeline, enforces mass balance architecturally rather than by penalty, and ships the uncertainty and physics diagnostics needed to defend a surrogate in review.

---

## Features

### Two evaluation protocols
| | Protocol 1 — random rows | Protocol 2 — GroupKFold by case |
|---|---|---|
| Unit of assignment | individual timestep | whole simulation |
| Question answered | temporal interpolation | generalisation to an unseen reservoir |
| Use when | filling gaps in an existing ensemble | screening new sites |

Both are available through a single `make_split(protocol, ...)` call, with the pros and cons of each documented in code (`PROTOCOL_NOTES`) so they can be printed into a manuscript.

### Two architectures
- **Baseline DeepONet** — feature attention, spectral-normalised branch/trunk, cross-attention fusion, four independent sigmoid heads. Closure is encouraged by a penalty term.
- **Closure DeepONet** — random Fourier time embedding, a true bilinear branch⊗trunk contraction, and a **softmax head that makes STTI + RTI + STI + MTI = 1 and non-negativity hold exactly, by construction**, with no penalty weight to tune.

### Uncertainty quantification
Three interchangeable engines, all returning predictive samples of shape `(S, N, 4)`:

| Engine | Cost | Epistemic source |
|---|---|---|
| `mc_dropout` | 1 training run + S passes | dropout masks |
| `deep_ensemble` | M training runs | independent optima |
| `bayesian` | 1 training run | variational posterior on the final projection |

Aleatoric noise comes from the head: a **heteroscedastic Gaussian** for the baseline, and a **logit-normal (additive-log-ratio)** head for the closure model, so every predictive sample remains a valid composition. Variance decomposes into aleatoric and epistemic parts via the law of total variance.

Metrics: R², RMSE, MAE, **PICP, MPIW/NMPIW, CRPS, ECE**, reliability diagrams, and post-hoc variance recalibration fitted on validation.

### Physics diagnostics
A regime-aware monotonicity loss and violation-rate metric covering all four indices, with the constraint set derived from the ensemble rather than assumed:

```
t ≤ 30 yr  (injection):   ∂ₜSTTI ≥ 0,  ∂ₜSTI ≤ 0
t > 30 yr  (post shut-in): ∂ₜSTTI ≤ 0,  ∂ₜRTI ≥ 0,  ∂ₜSTI ≥ 0,  ∂ₜMTI ≥ 0
```

### Sensitivity and trend analysis
One-at-a-time parameter sweeps over 0–200 years with all other inputs at the dataset mean, in two plot styles (trajectory curves or time–parameter response surfaces). Model trends are checked against **both** a textbook physics prior and the partial correlations measured in the simulation ensemble, so "the model is wrong" can be distinguished from "the training data is wrong".

### Hyperparameter optimisation
Optuna studies with median pruning, a 7-parameter core space and a 13-parameter extended space, persistent SQLite storage for resumable searches, and fANOVA importance. The objective is computed on a validation partition that is case-disjoint from the test cases, so tuning cannot leak into the reported score.

---

## Installation

```bash
git clone https://github.com/<user>/CO2STORETool.git
cd CO2STORETool
pip install -r requirements.txt
```

Requires Python ≥ 3.10, PyTorch, scikit-learn, Optuna, SHAP, pandas, matplotlib, scipy.

---

## Quickstart

```python
import co2gcs as C
from sklearn.preprocessing import StandardScaler

df = C.load('All_dataset.csv', fix_injection=True)
Xb, Xt, y, *_ = C.build_features(df, mode='fixed')

tr, va, te = C.make_split('group', len(y), df['case'].values, n_folds=5, fold=0)
sb, st = StandardScaler().fit(Xb[tr]), StandardScaler().fit(Xt[tr])

model = C.ClosureDeepONet(Xb.shape[1], Xt.shape[1], sigma=0.3)
model, _ = C.train_model(model,
                         sb.transform(Xb[tr]), st.transform(Xt[tr]), y[tr],
                         sb.transform(Xb[va]), st.transform(Xt[va]), y[va])

pred = C.predict_mean(model, sb.transform(Xb[te]), st.transform(Xt[te]))
print(C.accuracy(y[te], pred))
```

Command line:

```bash
python run_sensitivity.py --param temperature --protocol both --mode fixed
python run_tuning.py --arch closure --protocol group --space basic --n-trials 25
```

---

## Repository layout

```
co2gcs/
  data.py         loading, physics feature engineering, splitting protocols
  models.py       BaselineDeepONet, ClosureDeepONet, Bayesian last layer
  train.py        training loop, loss functions, pruning hook
  uq.py           MC dropout / deep ensemble / Bayesian engines
  metrics.py      accuracy, interval and calibration metrics
  physics.py      regime-aware monotonicity loss and violation rate
  sensitivity.py  OAT sweeps, response surfaces, physics checks
  tune.py         Optuna search spaces and studies
  plots.py        publication figures
notebooks/
  01_DeepONet_two_protocols.ipynb
  02_ClosureDeepONet_two_protocols.ipynb
  03_DeepONet_two_protocols_UQ.ipynb
  04_ClosureDeepONet_two_protocols_UQ.ipynb
  05_Hyperparameter_Optimization.ipynb
run_sensitivity.py, run_tuning.py, make_arch_diagrams.py
```

---

## Data

The reference ensemble is **761 reservoir simulations**, each a continuous trajectory from 0.16 to 200 years (~62 stored timesteps, 47,425 rows). Inputs: Dykstra-Parsons coefficient, porosity, injection rate, Biot coefficient, Young's modulus, Poisson's ratio, perforation and reservoir thickness, aquifer salinity, reservoir temperature.

Two feature modes are supported. `mode='orig'` uses the ten raw columns as recorded. `mode='fixed'` restores each case's injection rate across the full trajectory (it is zeroed for `t > 30 yr` in the raw file) and adds physics-motivated derived features: cumulative injected mass, pore-volume proxy, fill fraction, completion ratio, and a solubility driver.

---

## Selected results

Closure DeepONet, mean R² across the four indices:

| Feature mode | Random rows | GroupKFold by case |
|---|---|---|
| `orig` | 0.926 | 0.748 |
| `fixed` | 0.992 | **0.972** |

- Closure MAE is **exactly 0.0000** in every closure-architecture run, predictive samples included.
- Deep ensembles give the best CRPS and raw ECE; the Bayesian last layer is the best value per training run. Raw PICP ranks the methods misleadingly — MC dropout appears best-covered while having the worst ECE.
- Hyperparameter search identifies the Fourier bandwidth as the most influential parameter, independently confirming a value chosen by matching predicted trajectory total variation to the simulations.

---

## Roadmap

- [x] Data pipeline, feature engineering, two splitting protocols
- [x] Baseline and Closure DeepONet architectures
- [x] UQ engines, calibration metrics, reliability diagnostics
- [x] Sensitivity / trend analysis and physics violation metrics
- [x] Optuna hyperparameter optimisation
- [ ] Physics validation against independent simulation cases
- [ ] Temporal extrapolation protocol (train early, predict late)
- [ ] GUI for interactive screening and risk-informed decision support
- [ ] Manuscript and release

---

## Citation

A manuscript is in preparation. Please open an issue if you intend to use CO2STORETool in published work so citation details can be provided.

## License

MIT (see `LICENSE`).

## Contact

Promise (Keshi) Longe — Ph.D. candidate, Chemical & Petroleum Engineering, University of Kansas.
