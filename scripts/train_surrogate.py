#!/usr/bin/env python3
"""Train surrogate models Cl(alpha, Re) and Cd(alpha, Re) on the CFD sweep.

Models: Gaussian Process Regression and a multilayer perceptron (scikit-learn).
Validation:
  1. Hold-out speed: train without U = 30 m/s, predict that whole curve (interpolation in Re)
  2. Hold-out angles: train without alpha = 2, 6, 10 deg, predict them at all speeds
  3. Leave-one-out over all cases
Writes results/surrogate_metrics.csv, results/figures/*.png and models/*.joblib.
"""
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.compose import TransformedTargetRegressor
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
SEED = 0

df = pd.read_csv(ROOT / "data" / "sweep_results.csv")
X = np.column_stack([df.alpha_deg, np.log10(df.Re)])
targets = {"Cl": df.Cl.values, "Cd": df.Cd.values}


def make(name):
    if name == "GPR":
        # length scales bounded (in standardised units) so the GP cannot collapse onto the grid points
        kernel = C(1.0) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.3, 100.0)) + WhiteKernel(1e-6, (1e-10, 1e-3))
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=SEED))
    mlp = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs",
        alpha=1e-4, max_iter=20000, random_state=SEED))
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


rows, preds = [], {}
holdout = (df.U_inf == 30.0).values
hold_a = df.alpha_deg.isin([2.0, 6.0, 10.0]).values
for tname, y in targets.items():
    for mname in ["GPR", "MLP"]:
        m = make(mname).fit(X[~holdout], y[~holdout])
        p_hold = m.predict(X[holdout])
        p_ang = make(mname).fit(X[~hold_a], y[~hold_a]).predict(X[hold_a])
        p_loo = cross_val_predict(make(mname), X, y, cv=LeaveOneOut())
        preds[(tname, mname)] = p_loo
        for split, true, pred in [("holdout U=30 m/s", y[holdout], p_hold),
                                  ("holdout alpha 2/6/10", y[hold_a], p_ang),
                                  ("leave-one-out", y, p_loo)]:
            err = np.abs(pred - true)
            rows.append({"target": tname, "model": mname, "validation": split,
                         "MAE": err.mean(), "max_abs_error": err.max(),
                         "mean_rel_error_pct": 100 * (err / np.abs(true).clip(1e-3)).mean()})
        joblib.dump(make(mname).fit(X, y), MODELS / f"{tname}_{mname}.joblib")

metrics = pd.DataFrame(rows)
metrics.to_csv(ROOT / "results" / "surrogate_metrics.csv", index=False)
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

# Parity plots (leave-one-out predictions)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a, t in zip(ax, ["Cl", "Cd"]):
    y = targets[t]
    for mname, mk in [("GPR", "o"), ("MLP", "^")]:
        a.scatter(y, preds[(t, mname)], marker=mk, s=28, alpha=0.75, label=mname)
    lo, hi = y.min(), y.max(); pad = 0.05 * (hi - lo)
    a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    a.set(xlabel=f"{t} CFD", ylabel=f"{t} surrogate (leave-one-out)", title=f"{t}: surrogate vs CFD")
    a.grid(alpha=0.3); a.legend()
fig.tight_layout(); fig.savefig(FIG / "parity.png", dpi=150)

# Hold-out curve: CFD points vs GPR prediction with uncertainty band
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
re30 = df.Re[holdout].iloc[0]
a_fine = np.linspace(df.alpha_deg.min(), df.alpha_deg.max(), 200)
Xf = np.column_stack([a_fine, np.full_like(a_fine, np.log10(re30))])
for a, t in zip(ax, ["Cl", "Cd"]):
    y = targets[t]
    m = make("GPR").fit(X[~holdout], y[~holdout])
    mu, sd = m[-1].predict(m[0].transform(Xf), return_std=True)
    a.fill_between(a_fine, mu - 2 * sd, mu + 2 * sd, alpha=0.25, label="GPR ±2σ")
    a.plot(a_fine, mu, lw=2, label="GPR (trained without this Re)")
    a.plot(df.alpha_deg[holdout], y[holdout], "ko", label="CFD (unseen)")
    a.set(xlabel="angle of attack [deg]", ylabel=t, title=f"{t} at Re = {re30:.2e} (held out)")
    a.grid(alpha=0.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "holdout_prediction.png", dpi=150)
# Sanity check of the final models between grid points: smooth curves expected
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a, t_ in zip(ax, ["Cl", "Cd"]):
    gpr = joblib.load(MODELS / f"{t_}_GPR.joblib")
    for u in sorted(df.U_inf.unique()):
        re_ = df.Re[df.U_inf == u].iloc[0]
        Xu = np.column_stack([a_fine, np.full_like(a_fine, np.log10(re_))])
        line, = a.plot(a_fine, gpr.predict(Xu), lw=1.5, label=f"GPR, Re = {re_:.2e}")
        d = df[df.U_inf == u]
        a.plot(d.alpha_deg, d[t_], "o", color=line.get_color(), ms=4)
    a.set(xlabel="angle of attack [deg]", ylabel=t_, title=f"{t_}: final surrogate (lines) vs CFD (dots)")
    a.grid(alpha=0.3); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "surrogate_curves.png", dpi=150)
print(f"models -> {MODELS.relative_to(ROOT)}, figures -> {FIG.relative_to(ROOT)}")
